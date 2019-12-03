
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Trainer:
  def __init__(self,model):
    self.history = []
    self.model = model
    self.earlystopping = None

  def earlyStopping(self):
    self.earlystopping = EarlyStopping(monitor='val_loss', patience=2)

  def train(self, train, val):
    gen = DatasetLongSequence(train,15000,200)
    val_gen = DatasetSequence(val,1500,100)
    callbacks = []
    if self.earlystopping is not None:
      callbacks.append(self.earlystopping)

    outputs = self.model.fit_generator(gen, steps_per_epoch=20, epochs=50, validation_data = val_gen, validation_steps=20,callbacks=callbacks)