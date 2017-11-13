import keras


class Save_Valloss(keras.callbacks.Callback):
    def __init__(self, mode):
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.loss = 0
        self.val_loss = 0
        self.epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        f = open('%s.txt' % (self.mode), 'w')
        f.write('%d\n' % (epoch+1))
        f.write('loss: %f\n' % (logs.get('loss')))
        f.write('validation loss: %f\n' % (logs.get('val_loss')))
        f.close()

    def on_train_end(self, logs={}):
        return
