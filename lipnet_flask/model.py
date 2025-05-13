import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set memory growth for GPU to avoid consuming all GPU memory
physical_device = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_device[0], True)
except:
    pass


# Load video frames
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


# Define vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Keras StringLookup for character-to-number and number-to-character mappings
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


# Load alignment tokens, ignoring silence tokens
def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':  # Skip silence tokens
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


# Load data from video and alignment files
def load_data(path: str):
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments




def tensor_to_list(tensor):
    return [tensor.numpy().decode() for tensor in tensor]





# Function for model prediction and accuracy calculation
def predict_and_evaluate(input_path: str):
    # Load data
    sample = load_data(input_path)

    # Load the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(256, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(75, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool3D((1, 2, 2)))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(tf.keras.layers.Dropout(.5))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(tf.keras.layers.Dropout(.5))

    model.add(tf.keras.layers.Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights('models/checkpoint').expect_partial()

    # Make predictions
    yhat = model.predict(tf.expand_dims(sample[0], axis=0))

    # Decode the predictions
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

    # Decode the true alignments (ground truth)
    true_alignment = sample[1]

    # Convert predictions and ground truth to readable text
    decoded_text = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
    true_alignment_text = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [true_alignment]]

    # Calculate accuracy (number of common words)
    true_words = true_alignment_text[0].numpy().decode('utf-8').split()
    predicted_words = decoded_text[0].numpy().decode('utf-8').split()

    common_words = len(set(true_words) & set(predicted_words))
    accuracy = common_words / len(true_words) if len(true_words) > 0 else 0

    
    a = tensor_to_list(true_alignment_text)
    a = a[0].split()
    
    b = tensor_to_list(decoded_text)
    b = b[0].split()
    return a,b , accuracy


