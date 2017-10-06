import matplotlib.pyplot as plt
import pylab

def Plot_history(history, dir):

    plt.plot(history['loss'], "o-", label="loss",)
    plt.plot(history['val_loss'], "o-", label="val_loss",)
    plt.title('train history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='center right')
    plt.savefig(dir)
    #pylab.savefig(dir)
