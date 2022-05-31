
import os
import pathlib

test_record_fname = 'images/valid/hand.tfrecord'
train_record_fname = 'images/train/hand.tfrecord'
label_map_pbtxt_fname = 'images/train/hand_label_map.pbtxt'

MODELS_CONFIG = {
    'ssd-mobilenet' : {
        'model_name': 'ssd_mobilenet_v2',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
        'batch_size' : 16
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
        'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 16
    }
}

chosen_model = 'ssd-mobilenet'

num_steps = 40000
num_eval_steps = 500

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size']

pipeline_fname = 'models/' + base_pipeline_file
fine_tune_checkpoint = 'pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

num_classes = get_num_classes(label_map_pbtxt_fname)

import re

print('writing custom configuration file')

with open(pipeline_fname) as f:
    s = f.read()
with open('pipeline_file.config', 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    
    #fine-tune checkpoint type
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
    f.write(s)
'''
    #!python model_main_tf2.py --pipeline_config_path=pipeline_file.config --model_dir=models --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1 --num_eval_steps=500

    #python exporter_main_v2.py --trained_checkpoint_dir models --output_directory fine_model --pipeline_config_path pipeline_file.config

    #python export_tflite_graph_tf2.py --pipeline_config_path pipeline_file.config --trained_checkpoint_dir models --output_directory output

test_record_fname = '/content/drive/MyDrive/SignLanguageMobilenetV2/data/valid/Letters.tfrecord'
train_record_fname = '/content/drive/MyDrive/SignLanguageMobilenetV2/data/train/Letters.tfrecord'
label_map_pbtxt_fname = '/content/drive/MyDrive/SignLanguageMobilenetV2/data/train/Letters_label_map.pbtxt'
base_pipeline_file= 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config'

pipeline_fname = 'models/' + base_pipeline_file
fine_tune_checkpoint = 'pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'

'''
'''
saved_tflite_inference = "output/saved_model"
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 320, 320, 3)
      yield [data.astype(np.float32)]

import numpy as np
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_tflite_inference)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_quant_model = converter.convert()

tf_lite_model_path_without_metadata = "output/tflite/detect_without_metadata.tflite"
with tf.io.gfile.GFile(tf_lite_model_path_without_metadata, 'wb') as f:
  f.write(tflite_quant_model)
'''

'''
tf_lite_model_path_without_metadata = "output/tflite/detect_without_metadata.tflite"

from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

ObjectDetectorWriter = object_detector.MetadataWriter
_MODEL_PATH = tf_lite_model_path_without_metadata
_LABEL_FILE = "images/train/hand_label_map.pbtxt"
_SAVE_TO_PATH = "output/tflite/detect_with_metadata.tflite"

writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# Verify the populated metadata and associated files.
displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())

'''
'''
import tensorflow as tf
tf_lite_model_path_without_metadata = "output/tflite/detect_without_metadata.tflite"
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tf_lite_model_path_without_metadata)

print(interpreter.get_input_details()[0]['shape'])
print(interpreter.get_input_details()[0]['dtype'])

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])
print(interpreter.get_output_details()[0]['dtype'])
'''