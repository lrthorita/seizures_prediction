#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# *************************************************************************** #
# Author:  LUIZ RICARDO TAKESHI HORITA
# Master's Degree - Electrical and Computer Engineering
# University of Sao Paulo - Sao Carlos, SP, Brazil
# *************************************************************************** #
import numpy as np
import scipy.io
import os
from sklearn.externals import joblib

# Global constants
IEEG_SAMPLING_RATE = 400
SAMPLING_PERIOD = 1./400.
N_SAMPLES_SEGMENT = 240000


def load_matdata(filename):
    """Take the mat file and convert it in a dictionary variable with the 
    following keys: data, label (if it is a training data)."""
    matdata = scipy.io.loadmat(filename)
    dataStruct = matdata['dataStruct']
    data = {}
    data['data'] = dataStruct['data'][0,0]

    if "train" in filename:
        if "0." in filename:
            data['label'] = 0
        elif "1." in filename:
            data['label'] = 1
    return data


def get_fft(data):
    """Compute the Fast Fourier Transform on the signal of each channel of the
    data, and return the histogram of frequencies energy."""
    fft_data = np.fft.fft(data['data'],axis=0)
    frequencies = np.fft.fftfreq(N_SAMPLES_SEGMENT,SAMPLING_PERIOD)

    return fft_data, frequencies


#def get_feature(data):
#    """Get the frequency range where alpha, beta and mu waves are situated.
#    These waves are the waves that can, possibly, advise a future seizure 
#    event. The frequency range is between 8 Hz and 30 Hz."""
#    lower_index = np.int(8 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)
#    higher_index = np.int(30 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE) + 1
#
#    fft_data, frequencies = get_fft(data)
#
#    freq_range = frequencies[lower_index:higher_index]
#    energy_range = fft_data[lower_index:higher_index]
#
#    fft_feature = {'frequency':freq_range, 'energy':energy_range}
#
#    return fft_feature

def get_and_filter_fft(data):
    """Apply a band-pass filter in the range where alpha, beta and mu waves are
    situated. These waves are the waves that can, possibly, advise a future 
    seizure event. The frequency range is between 8 Hz and 30 Hz."""
    fft_data, frequencies = get_fft(data)

    for i in xrange(len(frequencies)):
        freq = frequencies[i]
        if (freq < -30) or ((freq > -8) and (freq < 8)) or (freq > 30):
            fft_data[i] = 0
    
    return fft_data, frequencies


def get_inverse_fft(fft_data):
    """Recover the signal by applying the inverse Fourier Transform on the
    fft_data."""
    filtered_data = np.fft.ifft(fft_data,axis=0)
    return filtered_data
    

def get_train_dataset():
    """Get all the training dataset and return it as a dictionary of all the 
    examples and labels. It will return a dictionary with dictionaries as
    elements. Each element corresponds to a unique .mat file. So, the structure
    of the returned variable is as follows:
        train_dataset
            | ['1_1_0.mat']
            |              |_ ['data']
            |_             |_ ['label']
            | [1_1_1.mat']
            |              |_ ['data']
            |_             |_ ['label']
            ...
    """
    for i in xrange(2,3):
        fft_train_dataset = {}
        directory = '../Seizures_Dataset/train_%s/' % i
        list_files = os.listdir(directory)
        for j in xrange(len(list_files)):
#        for j in xrange(1550,1600):
            path = directory + list_files[j]
            try:
                data = load_matdata(path)
                fft_feature = get_feature(data)
                fft_train_dataset[list_files[j]] = {
                                    'energy': fft_feature['energy'],
                                    'label': data['label']}
            except KeyboardInterrupt:
                break
            except:
                print "The file " + list_files[j] + " is corrupted."
                continue
            if j%50 == 0: print j
#            print j
            fft_train_dataset['frequency'] = fft_feature['frequency']
    
        # Save the dataset
        filename = 'fft_train_dataset%s.pkl' % i
        filename = joblib.dump(fft_train_dataset,filename)
        print "The training dataset was saved as: " + filename[0]
        # Use the following command to load the training dataset:
        """    joblib.load('fft_train_dataset.pkl')    """
    return fft_train_dataset


# ================================ MAIN ================================== #
#train_dataset = get_train_dataset()


