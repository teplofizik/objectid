from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from objectid.utils.dataset import ObjectDataset, DatasetSequence
from tensorflow.keras import backend as K

#example
# trainer = Trainer(final_model)
# trainer.earlyStopping()
# trainer.modelCheckpoint('best_model.h5')
# trainer.reduceLROnPlateau()
# trainer.run(train, val)

class Trainer:
  def __init__(self,model):
    self.history = []
    self.model = model
    self.earlystopping = None
    self.modelcheckpoint = None
    self.reducelronplateau = None
    self.val_gen = None
    self.epochs = 50
    self.batchsize = 200
    self.trainsize = 15000
    self.valsize = 1500
    # LearningRateScheduler
    
  def setLR(self,lr):
    K.set_value(self.model.optimizer.lr, lr)
    
  def earlyStopping(self):
    self.earlystopping = EarlyStopping(monitor='val_loss', patience=4)

  def modelCheckpoint(self,filename):
    self.modelcheckpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True)

  def reduceLROnPlateau(self):
    self.reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

  def run(self, train, val = None):
    gen = DatasetLongSequence(train,self.trainsize,self.batchsize)
    if val is not None:
      del self.val_gen
      self.val_gen = DatasetSequence(val,self.valsize,200)
    callbacks = []
    if self.earlystopping is not None:
      callbacks.append(self.earlystopping)
    if self.modelcheckpoint is not None:
      callbacks.append(self.modelcheckpoint)
    if self.reducelronplateau is not None:
      callbacks.append(self.reducelronplateau)

    outputs = self.model.fit_generator(gen, epochs=self.epochs, validation_data = self.val_gen, callbacks=callbacks)