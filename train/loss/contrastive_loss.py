#######################################################################################
# 
# Contrastive loss:
# https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.py
# 
#######################################################################################

from tensorflow.keras import backend as K

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
    