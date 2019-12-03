
from keras.callbacks import EarlyStopping, ModelCheckpoint
from objectid.utils.dataset import ObjectDataset, DatasetSequence

#example
# trainer = Trainer(final_model)
# trainer.earlyStopping()
# trainer.modelCheckpoint('best_model.h5')
# trainer.run(train, val)

class Trainer:
  def __init__(self,model):
    self.history = []
    self.model = model
    self.earlystopping = None
    self.modelcheckpoint = None

  def earlyStopping(self):
    self.earlystopping = EarlyStopping(monitor='val_loss', patience=4)

  def modelcheckpoint(self,filename):
    self.modelcheckpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', save_best_only=True)

  def run(self, train, val):
    gen = DatasetLongSequence(train,15000,200)
    val_gen = DatasetSequence(val,1500,100)
    callbacks = []
    if self.earlystopping is not None:
      callbacks.append(self.earlystopping)
    if self.modelcheckpoint is not None:
      callbacks.append(self.modelcheckpoint)

    outputs = self.model.fit_generator(gen, steps_per_epoch=20, epochs=50, validation_data = val_gen, validation_steps=20,callbacks=callbacks)