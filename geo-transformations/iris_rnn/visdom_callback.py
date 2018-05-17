import time

from keras.callbacks import Callback
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import visdom


class PlotVisdom(Callback):
    def __init__(self, autoencoder,
                 get_train_generator, get_val_generator):
        self.autoencoder = autoencoder
        self.train_generator = get_train_generator()
        self.val_generator = get_val_generator()
    
    def on_train_begin(self, logs={}):
        self.vis = visdom.Visdom(port=21002)
        startup_sec = 1
        while not self.vis.check_connection()\
                and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1

        assert self.vis.check_connection(),\
            'No connection could be formed quickly'

    def plot_figures(self, generator, epoch, logs):

        for images, _ in generator:
            decoded_images = self.autoencoder.predict(images)
            
            fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20, 10))
            
            for row in range(8):
                axes[row, 0].axis("off")
                axes[row, 1].axis("off")
                axes[row, 0].set_title("Input")
                axes[row, 0].imshow(images[row].reshape(64, 64))
                axes[row, 1].set_title("Output")
                axes[row, 1].imshow(decoded_images[row].reshape(64, 64))
            break
        
        title = 'Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(
                        epoch, logs.get('loss'), logs.get('val_loss'))

        self.vis.matplot(fig, opts=dict(title=title))

    def on_epoch_end(self, epoch, logs={}):
        self.plot_figures(self.train_generator, epoch, logs)
        self.plot_figures(self.val_generator, epoch, logs)
