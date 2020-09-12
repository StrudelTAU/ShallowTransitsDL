import os
import itertools
import matplotlib
import numpy as np
from tqdm import tqdm
from keras import backend as K
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import signal, fftpack
from keras.models import load_model
from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score

def intersection_loss(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    return 1-dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    #mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_np(y_true,y_pred):
    mask_true = y_true != 0 #K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = y_pred > np.mean(y_pred, keepdims=True, axis=1) #K.cast(K.not_equal(y_pred, 0), K.floatx())
    intersection = np.sum(mask_true * mask_true_2, axis=1)
    return (2. * intersection) / (np.sum(mask_true, axis=1) + np.sum(mask_true_2, axis=1)+0.00001)



def iterate_efficiently(input1, input2, output, chunk_size):
    # create an empty array to hold each chunk
    # the size of this array will determine the amount of RAM usage
    holder_1 = np.zeros([chunk_size,20610], dtype='float32')
    holder_2 = np.zeros([chunk_size,20610], dtype='float32')
    # iterate through the input, replace with ones, and write to output
    for i in tqdm(range(input1.shape[0])):
        if i % chunk_size == 0:
            holder_1[:] = input1[i:i+chunk_size] # read in chunk from input
            holder_2[:] = input2[i:i+chunk_size] # read in chunk from input
    
            output[i:i+chunk_size] = np.array(dice_coef_np(holder_1,holder_2)) # write chunk to output

    return output

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def load_data_now(header='', batch_size=128):
    x = np.expand_dims(np.load('./total_x_sim.npy', mmap_mode='r'), axis=1)
    x = np.pad(x, ((0, 0), (0, 0), (0, 30 + 96), (0, 0)), 'constant', constant_values=(0, 0))
    print('X loaded')

    params = np.load('./total_params_sim.npy')
    print('Params loaded')
    return x, None, params

def load_data(header='', batch_size=128):
    x = np.load('./total_x_sim_train.npy')
    x = np.expand_dims(x,axis=1)
    x = np.pad(x, ((0, 0), (0, 0), (0, 126), (0, 0)), 'constant', constant_values=(0.0, 0.0))
    print('X loaded')
    y = np.load('./total_transit_sim_train.npy')
    y = y[:, :, 0]
    print(y.shape)
    y_norm_max = np.max(y, axis=1, keepdims=True)
    y_norm_min = np.min(y, axis=1, keepdims=True)
    y = (y - y_norm_min) / (y_norm_max - y_norm_min)
    y = np.expand_dims(y, axis=-1)
    y = np.expand_dims(y, axis=1)
    y = np.pad(y, ((0, 0), (0, 0), (0, 30 + 96), (0, 0)), 'constant', constant_values=(0, 0))
    print(y.shape)
    print('Y loaded')
    print(x.shape, y.shape)
    has_transit = np.load('./total_params_sim_train.npy')[:,1] != 0
    print(np.sum(has_transit), has_transit.shape)
    print('Params loaded')
    print("Finished Loading Data")
    return x, y, has_transit

def process_transit(args):
    transit = args[0]
    peakind, _ = signal.find_peaks(transit, distance=args[2])
    peakind = peakind[args[0][np.array(peakind, dtype='int32')] > np.mean(args[0])]
    median_periods = []
    possible_periods = np.asarray([float(abs(combo[1]-combo[0]))/720.0 for combo in itertools.combinations(peakind, 2)])
    binned, _ = np.histogram(possible_periods, bins=[i for i in range(21)])
    for i in range(3):
        top_possible_periods = possible_periods[possible_periods <= np.argmax(binned)+1] 
        top_possible_periods = top_possible_periods[top_possible_periods > np.argmax(binned)] 
        binned[np.argmax(binned)] = 0
        curr_best = np.mean(top_possible_periods)
        if np.isnan(curr_best): curr_best = 100000000
        else:
            for j in range(4):
                test_next = int(curr_best * (j + 2))
                temp_periods = possible_periods[possible_periods <= test_next+1] 
                temp_periods = temp_periods[temp_periods > test_next] / (j+2)
                top_possible_periods = np.concatenate((top_possible_periods, temp_periods))
            curr_best = np.mean(top_possible_periods)
        median_periods.append((curr_best))
    min_arg = np.argmin(np.abs(np.asarray(median_periods - args[1])))
    best_found = median_periods[min_arg]
    return [best_found, args[1]]

def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    res_list = []
    for i, res in enumerate(p.imap_unordered(func, args)): res_list.append(res)
    p.close()
    p.join()
    return res_list

def p_epsilon_chart(p_test, p_pred):
    percentages = []
    auc_p = 0
    epsilon_range = np.linspace(0, 1, 10000)
    for epsilon in epsilon_range:
        current_correct = p_pred[np.abs(1 - (p_pred / p_test)) < epsilon]
        percentages.append(float(current_correct.shape[0]) / float(p_pred.shape[0]))
        auc_p += float(percentages[-1]) / 10000
    return auc_p, percentages, epsilon_range


