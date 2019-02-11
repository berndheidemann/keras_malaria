import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
from drawnow import drawnow

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.counter=0

        self.fig = plt.figure()
        self.logs = []

    def __init__(self, slowlyCutBeginning=True):
        self.slowlyCutBeginning=slowlyCutBeginning

    def paintPlot(self):
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.acc, label="val_acc")
        plt.legend()
       # plt.show();

    def on_epoch_end(self, epoch, logs={}):
        self.counter+=1
        if self.slowlyCutBeginning and self.counter%10==0:
            self.logs=self.logs[1:]
            self.x=self.x[1:]
            self.losses=self.losses[1:]
            self.val_losses=self.val_losses[1:]
            self.acc=self.acc[1:]
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        drawnow(self.paintPlot)

