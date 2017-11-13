import matplotlib.pyplot as plt
import pylab

def Plot_history(history, file):

    plt.plot(history['loss'], "o-", ls='-', label="loss",)
    plt.plot(history['val_loss'], "o-", ls='-', label="val_loss",)
    plt.title('train history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='center right')
    plt.ylim([0,0.5])
    plt.savefig(file)
    plt.figure()
    #pylab.savefig(file)
