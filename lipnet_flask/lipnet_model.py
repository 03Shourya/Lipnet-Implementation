# This is a placeholder. Replace with your actual LipNet logic.
from werkzeug.utils import secure_filename
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler







from model import predict_and_evaluate

def run_lipnet_model(video_path):
    # Extract the filename from the video path
    
    filename = os.path.basename(video_path)
    # Create a directory to save the frames
    
    #swith mp4 with mpg
    file_name = filename.replace('.mp4', '.mpg') 
    file_name = file_name.replace('clips','s1')
    
    file_path = os.path.join(filename)
    
    print(file_path)
    
    actual_words = []
    predicted_words = []
    accuracy = 0.0
    
    actual_words,predicted_words,accuracy= predict_and_evaluate(file_path)
    print(actual_words,predicted_words,accuracy)
    return actual_words, predicted_words, accuracy
