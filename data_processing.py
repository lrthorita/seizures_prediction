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

# Global constants
IEEG_SAMPLING_RATE = 400
SAMPLING_PERIOD = 1./400.
N_SAMPLES_SEGMENT = 240000


def load_matdata(filename):
    """Take the mat file and convert it in a dictionary variable with the 
    following keys: data, iEEsamplingRate, nSamplesSegment, channelIndices."""
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
    fft_data = abs(np.fft.fft(data['data'],axis=0))/(N_SAMPLES_SEGMENT/2)
    frequencies = np.fft.fftfreq(N_SAMPLES_SEGMENT,SAMPLING_PERIOD)

    return fft_data, frequencies


def get_feature(data):
    """Get the frequency range where alpha, beta and mu waves are situated.
    These waves are the waves that can, possibly, advise a future seizure 
    event. The frequency range is between 8 Hz and 30 Hz."""
    lower_index = np.int(8 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)
    higher_index = np.int(30 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)

    fft_data, frequencies = get_fft(data)

    freq_range = frequencies[lower_index:higher_index]
    fft_data_range = fft_data[lower_index:higher_index]

    return fft_data_range, freq_range


def get_train_dataset():
    train_dataset = {}
    for i in xrange(1,3):
        directory = '../Seizures_Dataset/train_%s/' % i
        list_files = os.listdir(directory)
        for j in xrange(len(list_files)):
            path = directory + list_files[j]
            try:
                train_dataset[list_files[j]] = load_matdata(path)
            except:
                print "The file " + list_files[j] + " is corrupted."
                continue
            if j%100 == 0: print j
    return train_dataset









