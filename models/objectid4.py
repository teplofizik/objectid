#######################################################################################
# 
# ObjectID network (based on FaceId)
# https://github.com/albertogaspar/keras-face-id/blob/master/models/facenet.py
# 
#######################################################################################

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, Concatenate, Conv2D
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

def Inception1x1(input, conv_1x1=64, strides_1x1=(1,1)):
    x = Conv2D(conv_1x1, 1, strides=strides_1x1, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Inception3x3(input, conv_1x1=96, conv_3x3=128, strides_1x1 =(1,1), strides_3x3 =(1,1)):
    x = Inception1x1(input, conv_1x1, strides_1x1=strides_1x1)
    x = Conv2D(conv_3x3, 3, strides=strides_3x3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
    
def Inception5x5(input, conv_1x1=16, conv_5x5=32, strides_1x1 =(1,1), strides_5x5 =(1,1)):
    x = Inception1x1(input, conv_1x1, strides_1x1=strides_1x1)
    x = Conv2D(conv_5x5, 5, strides=strides_5x5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
    

def InceptionPooling(input, conv_1x1=32, strides=(1,1), pool_type='max'):
    if pool_type == 'max':
        x = MaxPooling2D(pool_size=3, strides=strides, padding='same')(input)
    elif pool_type == 'l2':
        x = AveragePooling2D(pool_size=3, strides=strides, padding='same')(input)
    else:
        raise NotImplementedError('pool_type = {0}. '
                                  'This type of pooling is not available.'.format(pool_type))
    if conv_1x1:
        x = Inception1x1(x, conv_1x1=conv_1x1, strides_1x1=strides)
    return x
    
def InceptionLayer(input, conv_1x1, conv3x3_reduce, conv_3x3, conv_5x5_reduce, conv_5x5, pool_proj):
    to_concatenate = []
    if conv_1x1:
        inception_1x1 = Inception1x1(input, conv_1x1=conv_1x1[0], strides_1x1= conv_1x1[1])
        to_concatenate.append(inception_1x1)
    if conv_3x3:
        inception_3x3 = Inception3x3(input, conv_1x1=conv3x3_reduce[0], conv_3x3=conv_3x3[0],
                                     strides_1x1 =conv3x3_reduce[1], strides_3x3 =conv_3x3[1])
        to_concatenate.append(inception_3x3)
    if conv_5x5:
        inception_5x5 = Inception5x5(input, conv_1x1=conv_5x5_reduce[0], conv_5x5=conv_5x5[0],
                                     strides_1x1 =conv_5x5_reduce[1], strides_5x5 =conv_5x5[1])
        to_concatenate.append(inception_5x5)
    if pool_proj:
        inception_pool = InceptionPooling(input, conv_1x1=pool_proj[1], strides=pool_proj[2], pool_type=pool_proj[0])
        to_concatenate.append(inception_pool)
    inception = Concatenate()(to_concatenate)
    return inception
    
def objectid_model():
    input = Input(shape=(96,96,4))

    x = Conv2D(64, 7, strides=(2,2), padding='same')(input)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=(2,2), padding='same')(x)

    x = Inception3x3(x, conv_1x1=64, conv_3x3=192)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding='same')(x)


    inception_3a = InceptionLayer(x, conv_1x1=(64,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(128,(1,1)),
                                  conv_5x5_reduce=(16,(1,1)), conv_5x5=(32,(1,1)), pool_proj=('max',32,1))
    inception_3b = InceptionLayer(inception_3a, conv_1x1=(64,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(128,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',64,1))
    inception_3c = InceptionLayer(inception_3b, conv_1x1=None, conv3x3_reduce=(128,(1,1)), conv_3x3=(256,(2,2)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(2,2)), pool_proj=('max',None,2))

    inception_4a = InceptionLayer(inception_3c, conv_1x1=(256,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(192,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4b = InceptionLayer(inception_4a, conv_1x1=(224,(1,1)), conv3x3_reduce=(112,(1,1)), conv_3x3=(224,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4c = InceptionLayer(inception_4b, conv_1x1=(192,(1,1)), conv3x3_reduce=(128,(1,1)), conv_3x3=(256,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4d = InceptionLayer(inception_4c, conv_1x1=(160,(1,1)), conv3x3_reduce=(144,(1,1)), conv_3x3=(288,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4e = InceptionLayer(inception_4d, conv_1x1=None, conv3x3_reduce=(160,(1,1)), conv_3x3=(256,(2,2)),
                                  conv_5x5_reduce=(64,(1,1)), conv_5x5=(128,(2,2)), pool_proj=('max',None,2))

    inception_5a = InceptionLayer(inception_4e, conv_1x1=(384,(1,1)), conv3x3_reduce=(192,(1,1)), conv_3x3=(384,(1,1)),
                                  conv_5x5_reduce=(48,(1,1)), conv_5x5=(128,(1,1)), pool_proj=('l2',128,1))
    inception_5b = InceptionLayer(inception_5a, conv_1x1=(384,(1,1)), conv3x3_reduce=(192,(1,1)), conv_3x3=(384,(1,1)),
                                  conv_5x5_reduce=(48,(1,1)), conv_5x5=(128,(1,1)), pool_proj=('max',128,1))

    x = GlobalAveragePooling2D()(inception_5b)
    x = Dense(128)(x)
    
    return Model(inputs = [input], outputs = x)

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
