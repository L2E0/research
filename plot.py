import matplotlib.pyplot as plt
import pylab

def Plot_history(history, folder):

    plt.plot(history['loss'], "o-", label="loss",)
    plt.plot(history['val_loss'], "o-", label="val_loss",)
    #plt.plot(history['mean_squared_error'], "o-", label="mse",)
    #plt.plot(history['val_mean_squared_error'], "o-", label="val_mse")
    plt.title('train history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='center right')
    plt.savefig(folder)
    plt.show()
    #pylab.savefig(folder)
