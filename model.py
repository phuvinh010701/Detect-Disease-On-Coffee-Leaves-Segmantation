import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Convolution2DTranspose, Input
from tensorflow.keras.models import Model
import numpy as np


def downsample_block(block_input, num_filter, is_first=False):
  if is_first:
    conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(block_input)
    conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(conv1)
    return [block_input, conv1, conv2]
  else:
    maxpool = MaxPooling2D(pool_size=2)(block_input)
    conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(maxpool)
    conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(conv1)
    return [maxpool, conv1, conv2]

def upsample_block(block_input, block_counterpart, num_filter, is_last=False):
  uppool1 = Convolution2DTranspose(num_filter, kernel_size=2, strides=2)(block_input)
  concat = Concatenate(axis=-1)([block_counterpart, uppool1])
  conv1 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(concat)
  conv2 = Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(conv1)

  if is_last:
    conv3 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='softmax')(conv2)
    return [concat, conv1, conv2, conv3]
  return [uppool1, concat, conv1, conv2]

def unet(input_size=(256, 256, 3)):
  ds_block1 = downsample_block(Input(shape=input_size), num_filter=64, is_first=True)
  ds_block2 = downsample_block(ds_block1[-1], num_filter=128)
  ds_block3 = downsample_block(ds_block2[-1], num_filter=256)
  ds_block4 = downsample_block(ds_block3[-1], num_filter=512)
  ds_block5 = downsample_block(ds_block4[-1], num_filter=1024)

  us_block4 = upsample_block(ds_block5[-1], ds_block4[-1], num_filter=512)
  us_block3 = upsample_block(us_block4[-1], ds_block3[-1], num_filter=256)
  us_block2 = upsample_block(us_block3[-1], ds_block2[-1], num_filter=128)
  us_block1 = upsample_block(us_block2[-1], ds_block1[-1], num_filter=64, is_last=True)

  model = Model(inputs=ds_block1[0], outputs=us_block1[-1])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model

