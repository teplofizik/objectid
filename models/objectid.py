#######################################################################################
# 
# ObjectID network (based on FaceId)
# https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.py
# 
#######################################################################################

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from tensorflow.keras.activations import relu, softmax
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

def squeeze_model():
  img_input=Input(shape=(96,96,4))
  x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

  x = fire(x, squeeze=16, expand=16)
  x = fire(x, squeeze=16, expand=16)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

  x = fire(x, squeeze=32, expand=32)
  x = fire(x, squeeze=32, expand=32)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

  x = fire(x, squeeze=48, expand=48)
  x = fire(x, squeeze=48, expand=48)
  x = fire(x, squeeze=64, expand=64)
  x = fire(x, squeeze=64, expand=64)

  x = Dropout(0.2)(x)
  x = Convolution2D(512, (1, 1), padding='same')(x)
  out = Activation('relu')(x)

  return Model(img_input, out)

def objectid_model():
  modelsqueeze = squeeze_model()
  im_in = Input(shape=(96,96,4))

  x1 = modelsqueeze(im_in)
  x1 = Flatten()(x1)
  x1 = Dense(512, activation="relu")(x1)
  x1 = Dropout(0.2)(x1)
  feat_x = Dense(128, activation="linear")(x1)

  return Model(inputs = [im_in], outputs = feat_x)

def objectid_train_model(objectid_model):
  im_in1 = Input(shape=(96,96,4))
  im_in2 = Input(shape=(96,96,4))
  
  feat_x1 = objectid_model(im_in1)
  feat_x2 = objectid_model(im_in2)

  # L2 Normalization in final layer, tfjs not support Lambda python code layer =)
  # Need perform normalization in pure js
  feat_x1 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x1)
  feat_x2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x2)

  lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
  return Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
