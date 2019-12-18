from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import os


# TODO: define dataset x and y
x = np.array([[1,2],
                [2,5],
                [2,6],
                [2,9],
                [20,4],
                [23,36],
                [26,2],
                [9,23]])
y = np.array([3,7,8,11,24,59,28,32])

# TODO: define constants
N_FEATURES = 2
lr = 0.01
epochs = 100


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def linear_model():
    model = Sequential()
    model.add(Dense(1, input_dim=N_FEATURES, kernel_initializer='normal', activation=None))
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())
    return model

class PlotLosses(Callback):

    def __init__(self, path):
        self.losses = []
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 1 or (epoch > 1 and logs.get('loss') < min(self.losses)):
            copyfile("runs/model_{}.hdf5".format(epoch), "runs/model_best.hdf5")

        self.losses.append(logs.get('loss'))


create_path("plots")
create_path("runs")

checkpointer = ModelCheckpoint(filepath="runs\\model_{epoch}.hdf5", verbose=1)
plot_losses = PlotLosses("plots\\")
model = linear_model()
model.fit(x=x, y=y, epochs=epochs, steps_per_epoch=x.shape[0], callbacks=[checkpointer, plot_losses])

plt.figure()
plt.plot(plot_losses.losses)
plt.title('loss on epochs')
plt.savefig(plot_losses.path + 'loss.svg', format='svg')
plt.show()

print(model.layers[0].get_weights())