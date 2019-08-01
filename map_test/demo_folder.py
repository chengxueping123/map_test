#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiBox.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiBox weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections
reload(sys)
sys.setdefaultencoding('utf-8')
# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import scipy as scp
import scipy.misc
import tensorflow as tf
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiBox.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiBox.')


default_run = 'KittiBox_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiBox_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiBox_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return
#--add code for save box detial to evaluation kittibox 2018-10-16 14:26:07
def savedata(rect,filename,path):#'car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic_sign', 'traffic_light'
    # Class = {'0':'Car', '1':'Van','2':'Truck','3':'Pedestrian','4':'Person_sitting','5':'Cyclist','6':'Tram','7':'Misc'}
    Class = {'0':'car', '1':'bus', '2':'person', '3':'bike', '4':'truck', '5':'motor', '6':'train', '7':'rider', '8':'traffic_sign', '9':'traffic_light'}
    fileout = path+filename.split('.')[0] + '.txt'
    with open(fileout,'a+') as f:
        f.write(str(Class[str(rect.classLabel)]))
        f.write(" ")
        f.write(str(rect.score))
        f.write(" ")
        f.write(str(rect.x1))
        f.write(" ")
        f.write(str(rect.y1))
        f.write(" ")
        f.write(str(rect.x2))
        f.write(" ")
        f.write(str(rect.y2))        
        f.write("\n")
    print("the box write to:",fileout,"done")

#-- add end

def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiBox')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for inputclassLabel
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # yang.
        image.set_shape([1, None, None, 3])
        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))
    # yang.
    for img in os.listdir(input_image):
        # print(input_image)
    # filelist = ['100k/train/a9c4268f-b97f4325.jpg','100k/train/1f416cf1-10a9e2ed.jpg','100k/train/4d22354a-6ce691da.jpg','100k/train/2786c637-b689c08a.jpg','100k/train/1c56c68b-b520ae6d.jpg','100k/train/7170cba6-dd72fe05.jpg','100k/train/71cf580c-d819c189.jpg','100k/train/70b5dbac-87ba4de7.jpg','100k/train/219da1a7-030143c1.jpg','100k/train/2f45c868-4399445f.jpg']
    # for img in filelist:
    #if img.split('.')[-1] == 'png':
    #if img.split('.')[-1] == 'png':
# Load and resize input image
        # image = scp.misc.imread('/home/yang/home/yang/MultiNet/submodules/KittiBox/DATA/KittiBox/' + img)
        image = scp.misc.imread(input_image + img)
       # if hypes["resize_image"]:
        #    image = scp.misc.imresize(image, (hypes["image_height"],hypes["image_width"]),interp='cubic')
        image = image[8:,:,:]
        # print(image.shape)
        # break
        # image = scp.misc.imread(input_image)
        
        feed = {image_pl: image}

        # Run KittiBox model on image
        pred_boxes = prediction['pred_boxes']
        pred_confidences = prediction['pred_confidences']
        # yang.
        pred_logits = prediction['pred_logits']
        # print(pred_boxes,pred_confidences,pred_logits)
        # yang.
        #print('----------------------demo.time------------------------')
        t0 = time.time()
        # (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
        #                                                 pred_confidences],
        #                                                 feed_dict=feed)
        (np_pred_boxes, np_pred_confidences, np_pred_logits) = sess.run([pred_boxes,
                                                        pred_confidences,
                                                        pred_logits],
                                                        feed_dict=feed)
        # print(np_pred_boxes,np_pred_confidences,np_pred_logits)
        print(time.time() - t0)
        # # print('--------------------------logits-----------------------')
        # print('-------------------------------------------------------')
        # print(np_pred_confidences)
        # # print(np_pred_logits)
        # print('-------------------------------------------------------')
        # break
        #print('-------------------------------------------------------')

        # Apply non-maximal suppression
        # and draw predictions on the image
        # yang.
        # print(pred_logits.shape)
        # break
        # output_image, rectangles = kittibox_utils.add_rectangles(
        #     hypes, [image], np_pred_confidences,
        #     np_pred_boxes, np_pred_logits, show_removed=False,
        #     use_stitching=True, rnn_len=1,
        #     min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))
        
        output_image, rectangles = kittibox_utils.add_rectangles(
            hypes, [image], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=1,
            min_conf=0.75, tau=hypes['tau'], color_acc=(0, 255, 0))

        threshold = 0.75
        accepted_predictions = []
        # removing predictions <= threshold
        for rect in rectangles:
            if rect.score >= threshold:
                accepted_predictions.append(rect)

        print('')
        logging.info("{} Cars detected".format(len(accepted_predictions)))

        # Printing coordinates of predicted rects.
        for i, rect in enumerate(accepted_predictions):
            logging.info("")
            logging.info("Coordinates of Box {}".format(i))
            logging.info("    class: {}".format(rect.classID))
            logging.info("    x1: {}".format(rect.x1))
            logging.info("    x2: {}".format(rect.x2))
            logging.info("    y1: {}".format(rect.y1))
            logging.info("    y2: {}".format(rect.y2))
            logging.info("    Confidence: {}".format(rect.score))
            savedata(rect,img,FLAGS.output_image)

        # save Image
        output_name = os.path.join(FLAGS.output_image, img.split('.')[0] + '_rects.jpg')
        #output_image = scp.misc.imresize(output_image, (512,640),interp='cubic')
#        scp.misc.imsave(output_name, output_image)
        logging.info("")
        logging.info("Output image saved to {}".format(output_name))
        

if __name__ == '__main__':
    tf.app.run()
