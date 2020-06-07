import argparse
import numpy as np
import os
import json
from voc import parse_voc_annotation
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
import tensorflow as tf
import keras
from batch_generator import BatchGenerator
from yolo import create_yolov3_model, dummy_loss

def create_training_instances(train_annot_folder, train_image_folder,
                              train_cache,
                              valid_annot_folder, valid_image_folder,
                              valid_cache, labels):
    train_instances, train_labels = parse_voc_annotation(train_annot_folder,
                                                         train_image_folder,
                                                         train_cache,
                                                         labels)
    
    if os.path.exists(valid_annot_folder):
        valid_instances, valid_labels = parse_voc_annotation(valid_annot_folder,
                                                         valid_image_folder,
                                                         valid_cache,
                                                         labels)
    else:
        train_valid_split = int(0.85 * len(train_instances))
        np.random.seed(0)
        np.random.shuffle(train_instances)
        np.random.seed()
        
        valid_instances = train_instances[train_valid_split:]
        train_instances = train_instances[:train_valid_split]
        
        max_box_per_image = max([len(instance['object']) for instance 
                                   in (train_instances + valid_instances)])
        
        return train_instances, valid_instances, labels, max_box_per_image
    
def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stopping = EarlyStopping(
            monitor = 'loss',
            min_delta = 0.01,
            patience = 7,
            mode = 'min',
            verbose = 1
            )
    
    checkpoint = CustomModelCheckpoint(
            model_to_save = model_to_save,
            filepath = saved_weights_name,# + '{epoch:02d}.h5',
            monitor = 'loss',
            verbose = 1,
            save_best_only = True,
            mode = 'min',
            period = 1
            )
    
    reduce_on_plateau = ReduceLROnPlateau(
            monitor = 'loss',
            factor = 0.1,
            patience = 2,
            verbose = 1,
            mode = 'min',
            epsilon = 0.01,
            cooldown = 0,
            min_lr = 0
            )
    
    tensorboard = CustomTensorBoard(
            log_dir = tensorboard_logs,
            write_graph = True,
            write_images = True
            )
    
    return [early_stopping, checkpoint, reduce_on_plateau, tensorboard]

def create_model(num_of_classes, anchors, max_box_per_image, max_grid, 
                 batch_size, warmup_batches, ignore_thresh, 
                 saved_weights_name, lr, grid_scales, obj_scale,
                 noobj_scale, xywh_scale, class_scale):
    
    template_model, infer_model = create_yolov3_model(
            nb_class            = num_of_classes, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        ) 
    
    #pretrained or not
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)       
        
    train_model = template_model
    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
        
    #   Parse the annotations 
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    #   Create the generators   
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    #   Create the model 
    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = config['train']['learning_rate'],
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
    )
    
    #  Start the training
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)

    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'], 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 8
    )

    #   Run the evaluation 
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions))) 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
    

