
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from objectid.utils.dataset import ObjectDataset, DatasetSequence

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
    # LearningRateScheduler

  def earlyStopping(self):
    self.earlystopping = EarlyStopping(monitor='val_loss', patience=4)

  def modelCheckpoint(self,filename):
    self.modelcheckpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True)

  def reduceLROnPlateau(self):
    self.reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

  def run(self, train, val):
    gen = DatasetLongSequence(train,15000,200)
    val_gen = DatasetSequence(val,1500,100)
    callbacks = []
    if self.earlystopping is not None:
      callbacks.append(self.earlystopping)
    if self.modelcheckpoint is not None:
      callbacks.append(self.modelcheckpoint)
    if self.reducelronplateau is not None:
      callbacks.append(self.reducelronplateau)

    outputs = self.model.fit_generator(gen, steps_per_epoch=20, epochs=50, validation_data = val_gen, validation_steps=20,callbacks=callbacks)