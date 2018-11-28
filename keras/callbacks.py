import keras, math
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from pathlib import Path

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias'):
            if layer.bias and hasattr(layer, 'bias_initializer'): layer.bias.initializer.run(session=session)

class CycleLearner(keras.callbacks.Callback):
    def __init__(self, lr, nb, n_cycle, cycle_len=1, cycle_mult=1,
                 snapshots=False, n_snapshots=5,
                 snapshots_name='temp_model', snapshots_folder='snapshots'):
        super().__init__()
        self.lr = lr
        self.nb = nb*cycle_len
        self.n_cycle = n_cycle
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.snapshots = snapshots
        self.n_snapshots = n_snapshots
        self.start_snaps = n_cycle - n_snapshots
        self.snapshots_name = snapshots_name
        self.snapshots_folder = Path(snapshots_folder)
        self.lr_log = []
        self.losses = []
        self.iterations = []
        if snapshots: self.snapshots_folder.mkdir(exist_ok=True)

    def on_train_begin(self, logs={}):
        self.iteration,self.epoch = 0,0
        self.cycle_iter,self.cycle_count=0,0
        self.update_lr()
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.lr_log.append(K.eval(self.model.optimizer.lr))
        loss = logs.get('loss')
        self.losses.append(loss)
        self.iterations.append(self.iteration)
        self.update_lr()
        if self.cycle_count == self.n_cycle:
            self.model.stop_training = True
    
    def on_cycle_end(self):
        self.cycle_iter = 0
        self.nb *= self.cycle_mult
        self.cycle_count += 1
        if self.snapshots and self.cycle_count>self.start_snaps:
            file = self.snapshots_folder / f'{self.snapshots_name}_{self.cycle_count}.hdf5'
            self.model.save_weights(file, overwrite=True)
    
    def update_lr(self):
        new_lr = self.calc_lr()
        K.set_value(self.model.optimizer.lr, new_lr)

    def calc_lr(self):
        if self.iteration<self.nb/20:
            self.cycle_iter += 1
            return self.lr/100.

        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.on_cycle_end()
            
        return self.lr / 2 * cos_out        
        
    def plot_loss(self, skip=10):
        plt.plot(self.iterations[skip:], self.losses[skip:])

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lr_log)


class LrFinder(keras.callbacks.Callback):
    def __init__(self, nb, start_lr=1e-5, end_lr=10):
        super().__init__()
        self.start_lr = start_lr
        self.lr = start_lr
        self.lr_mult = (end_lr/start_lr)**(1/nb)
        self.lr_log = []
        self.losses = []
        self.iterations = []
        
    def on_train_begin(self, logs={}):
        self.best = 1e9
        self.iteration = 0
        self.update_lr()

    def on_batch_end(self, batch, logs={}):
        self.iteration += 1
        self.lr_log.append(K.eval(self.model.optimizer.lr))
        loss = logs.get('loss')
        self.losses.append(loss)
        self.iterations.append(self.iteration)

        if math.isnan(loss) or loss>self.best*4:
            self.model.stop_training = True
        if loss<self.best:
            self.best=loss
        self.update_lr()
    
    def update_lr(self):
        new_lr = self.start_lr * (self.lr_mult**self.iteration)
        K.set_value(self.model.optimizer.lr, new_lr)
        
    def plot_loss(self):
        plt.plot(self.iterations[10:], self.losses[10:])

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lr_log)
        
    def plot(self, n_skip_start=2, xlim=None, ylim=None):
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lr_log[n_skip_start:], self.losses[n_skip_start:])
        plt.xscale('log')
        if ylim is not None: plt.ylim(top=ylim)
        if xlim is not None: plt.xlim(right=xlim)
        