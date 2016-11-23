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
    """Take the mat file and convert it in a numpy matrix variable with the 
    datas from 16 channels (columns) and 240,000 elements (rows) each."""
    matdata = scipy.io.loadmat(filename)
    data = matdata['dataStruct']['data'][0,0]
    return data


def get_fft(data):
    """Compute the Fast Fourier Transform on the signal of each channel of the
    data, and return the histogram of frequencies energy."""
    fft_data = np.fft.fft(data,axis=0)
    frequencies = np.fft.fftfreq(N_SAMPLES_SEGMENT,SAMPLING_PERIOD)

    return fft_data, frequencies


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
    return filtered_data.real


def train_cross_correlation():
    """Compute the cross-correlation function for all the training dataset,
    creating 16 correlation matrices. It returns a dictionary with keys as
    'channel_i', where i is the channel number (1 to 16), and the respective
    correlation matrix."""
    datas = np.loadtxt('../Seizures_Dataset/train_and_test_data_labels_safe.csv',
                       dtype='str', delimiter=',', skiprows=1, usecols=(0,1))
    datas = datas[0:6042,:] # Eliminate the test datas
    train_xcorr = {}
    corrupted_list = []
    try:
        for c in xrange(16):
            key = "channel_%s" % (c+1)
            train_xcorr[key] = np.zeros((6042,6042))
            x = y = 0
            for i in xrange(6042):
                if corrupted_list.count(i) == 0:
                    # File not corrupted
                    corrupted_i = False
                    
                    # Load the i-th sample
                    file_dir = "../Seizures_Dataset/train_%s/" % datas[i,0][0]
                    file_path = file_dir + datas[i,0]
                    try:
                        sample_i = load_matdata(file_path)
                        # Band-pass filter (8-30 Hz)
                        fft_sample_i, freq_sample_i = get_and_filter_fft(sample_i)
                        filtered_sample_i = get_inverse_fft(fft_sample_i)
                    except KeyboardInterrupt:
                        print "Interrupted!"
                        break
                    except:
                        print "The file " + datas[i,0] + " is corrupted."
                        # The file must be corrupted
                        corrupted_i = True
                        corrupted_list.append(i)
                        x = x - 1
                        continue
                    
                    if corrupted_i == False: # Check if the i-th file is corrupted
                        j = i
                        for j in xrange(i,6042):
                            if corrupted_list.count(j) == 0:
                                # Load the j-th sample
                                file_dir = "../Seizures_Dataset/train_%s/" % datas[j,0][0]
                                file_path = file_dir + datas[j,0]
                                try:
                                    sample_j = load_matdata(file_path)
                                    # Band-pass filter (8-30 Hz)
                                    fft_sample_j, freq_sample_j = get_and_filter_fft(sample_j)
                                    filtered_sample_j = get_inverse_fft(fft_sample_j)
                                    
                                    # Correlation between the i-th sample and the j-th sample
                                    print "Correlation between:"
                                    print str(i)+"-th sample: " + datas[i,0]
                                    print str(j)+"-th sample: " + datas[j,0] + "\n"
                                    corr = np.correlate(filtered_sample_i[:,c],
                                                        filtered_sample_j[:,c])
                                    
                                    # The cross-correlation matrix is symmetric
                                    train_xcorr[key][x,y] = train_xcorr[y,x] = corr
                                except KeyboardInterrupt:
                                    print "Interrupted!"
                                    break
                                except:
                                    print "The file " + datas[j,0] + " is corrupted."
                                    corrupted_list.append(j)
                                    y = y - 1
                                    continue
                            y = y + 1
                x = x + 1
                    
        # Save the cross-correlation matrices
        joblib.dump(train_xcorr,'train_cross_correlation.pkl')
    except KeyboardInterrupt:
        print "Interrupted!"
    
    return train_xcorr


# ================================ MAIN ================================== #
train_xcorr = train_cross_correlation()


