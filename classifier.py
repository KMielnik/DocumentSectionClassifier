#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from imageio import imsave
from tqdm import tqdm

from dh_segment.inference import LoadedModel
from dh_segment.post_processing import binarization


def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, kernel_size=5)
    return mask


def get_classes(file_path: str="classes.txt"):
    """
    Gets document classes from file.
    :param file: filepath with extension
    """

    f = open(file_path, "r")

    classes = []
    for line in f:
        cl = []
        for nbr in line.split():
            cl.append(int(nbr))
        classes.append(cl)

    return classes

def load_model(model_path: str="model"):
    """
    Loads model and starts session.
    :param model_path: folder path to the model
    """

    if not os.path.exists(model_path):
        print("Couldn't load model - path doesn't exist.")
        return
    
    sess = tf.compat.v1.Session()
    sess.__enter__()
    return (LoadedModel(model_path, predict_mode='filename'), sess)

def process_document(filename: str, model: LoadedModel, mask_only: bool=False, input_path: str="input/", output_path: str="output/"):
    os.makedirs(output_path, exist_ok=True)

    input_filepath = input_path + filename
    
    # For each image, predict each pixel's label
    prediction_outputs = model.predict(input_filepath)
    probs = prediction_outputs['probs'][0]
    original_shape = prediction_outputs['original_shape']
    classes = get_classes()

    img = Image.open(input_path + filename, 'r')
    pixels = img.load()

    if mask_only:
        for i in range(original_shape[::-1][1]):
            for j in range(original_shape[::-1][0]):
                pixels[j,i]=(255,255,255) #TODO: NAPRAW

    for p, cl in enumerate(classes, 1):
        prob = probs[:, :, p]  # Take only class 'p' (class 0 is the background, class 1 is the page)
        prob = prob / np.max(prob)  # Normalize to be in [0, 1]

        # Binarize the predictions
        page_bin = page_make_binary_mask(prob)

        # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
        bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                               tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

        cr, cg, cb = cl[0:3]

        if(mask_only):
            for i, row in enumerate(bin_upscaled):
                for j, col in enumerate(row):
                    # If pixel belongs to a class
                    if col == 1:
                        # Make class colors transparent
                        nr = cr
                        ng = cg
                        nb = cb
                        pixels[j, i] = (nr, ng, nb)
        else:
            # Mark each masked pixel with a transparent color
            for i, row in enumerate(bin_upscaled):
                for j, col in enumerate(row):
                    # If pixel belongs to a class
                    if col == 1:
                        r, g, b = img.getpixel((j, i))
                        # Make class colors transparent
                        nr = int((cr + r) / 2)
                        ng = int((cg + g) / 2)
                        nb = int((cb + b) / 2)
                        pixels[j, i] = (nr, ng, nb)
    # Save output
    basename = os.path.basename(filename).split('.')
    if(mask_only):
        imsave(os.path.join(output_path, 'mask_{}.{}'.format(basename[0], basename[1])), img)
    else:
        imsave(os.path.join(output_path, '{}.{}'.format(basename[0], basename[1])), img)

def process_all_documents(model: LoadedModel, mask_only: bool=False, input_path: str="input/", output_path: str="output/"):
    input_files = glob(input_path + "*")

    for filename in tqdm(input_files, desc='Processed files'):
        process_document(filename, model, mask_only, input_path, output_path)