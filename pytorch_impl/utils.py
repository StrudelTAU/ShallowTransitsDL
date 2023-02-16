import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def make_scatter(lc_in, lc_out, target, loss=0):
    fig, axes = plt.subplots(3,1, figsize=(8,10))
    axes[0].scatter(np.arange(0, len(lc_in)), lc_in, s=0.5)
    axes[0].set_title('input')
    axes[1].scatter(np.arange(0, len(lc_in)), lc_out, s=0.5)
    axes[1].set_title('output')
    axes[2].scatter(np.arange(0, len(lc_in)), target, s=0.5)
    axes[2].set_title('target - dice: ' + str(1-loss))
    return fig



def draw_auc_chart(y_pred, y_test):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='pytorch (area = {:.5f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Classifier ROC curve')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlim(0, 0.02)
    return plt.gcf()


def draw_snr_dice_chart(dice, snr, zoom=False):
    plt.hist2d(snr, dice, bins=(100, 100), cmap=plt.cm.Greys)
    plt.colorbar()
    plt.xlabel('SNR')
    plt.ylabel('Dice Coeff.')
    plt.xscale('log')
    plt.xticks([1,2,5,10,20,50,100,200], ['1','2','5','10','20','50','100','200'])
    if zoom: 
        plt.xlim(1, 125)
        plt.ylim(0.75, 1)
    return plt.gcf()

