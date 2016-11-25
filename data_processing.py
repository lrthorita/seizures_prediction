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
N_TRAINING_SAMPLES = 6042
N_TEST_SAMPLES = 6126
N_NEW_TEST_SAMPLES = 1908

def load_matdata(filename):
    """Take the mat file and convert it in a numpy matrix variable with the 
    datas from 16 channels (columns) and 240,000 elements (rows) each."""
    matdata = scipy.io.loadmat(filename)
    data = matdata['dataStruct']['data'][0,0]
    return data


def get_dataset(mode):
    """Take all the training or test dataset and concatenate in 16 matrices in the
    feature space (FFT + Filter(0~30Hz))."""
    datas = np.loadtxt('../Seizures_Dataset/train_and_test_data_labels_safe.csv',
                       dtype='str', delimiter=',', skiprows=1, usecols=(0,1))
    
    if mode == 'train':
        N_SAMPLES = N_TRAINING_SAMPLES
        datas = datas[0:N_SAMPLES,:] # Eliminate the test datas
        directory = "../Seizures_Dataset/train_"
    elif mode == 'test':
        N_SAMPLES = np.size(datas,0) - N_TRAINING_SAMPLES
        datas = datas[N_TRAINING_SAMPLES:np.size(datas,0),:] # Eliminate the test datas
        directory = "../Seizures_Dataset/test_"
    
    
    feature_dataset = {}
    corrupted_list = []
    for c in xrange(16):
        key = "channel_%s" % (c+1)
        feature_dataset[key] = np.zeros((13201,N_SAMPLES))
    for i in xrange(N_SAMPLES):
        if corrupted_list.count(i) == 0:
            # Load the i-th sample
            file_dir = directory + "%s/" % datas[i,0][0]
            file_path = file_dir + datas[i,0]
            try:
                sample = load_matdata(file_path)
                # Get spectrum in the range 8-30 Hz
                fft_feature = get_feature(sample)
                feature = abs(fft_feature['energy'])/N_SAMPLES_SEGMENT

                for c in xrange(16):
                    key = "channel_%s" % (c+1)
                    feature_dataset[key][:,i] = feature[:,c]
                
                print str(i) + "-th feature added."
                    
            except KeyboardInterrupt:
                print "Interrupted!"
                break
            except:
                print "The file " + datas[i,0] + " is corrupted."
                corrupted_list.append(i)
                continue
    
    # Save the cross-correlation matrices
    filename = "feature_" + mode + "_dataset.pkl"
    joblib.dump(feature_dataset,filename)
    
    return feature_dataset


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
#        if (freq < -30) or ((freq > -8) and (freq < 8)) or (freq > 30):
        if (freq < -30) or (freq > 30):
            fft_data[i] = 0
    
    return fft_data, frequencies


def get_feature(data):
    """Get the frequency range where alpha, beta and mu waves are situated.
    These waves are the waves that can, possibly, advise a future seizure 
    event. The frequency range is between 8 Hz and 30 Hz."""
#    fft_data, frequencies = get_fft(data)
#    fft_feature = np.zeros((31,16))
#    for i in xrange(31):
#        # Sample spectrum for integer values of frequency from 0 to 30 Hz
#        # The index of i Hz
#        i_index = np.int(i * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)
#        fft_feature[i,:] = abs(fft_data[i_index,:])/N_SAMPLES_SEGMENT
#
#    return fft_feature
    lower_index = np.int(8 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)
    higher_index = np.int(30 * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE) + 1

    fft_data, frequencies = get_fft(data)

    freq_range = frequencies[lower_index:higher_index]
    energy_range = fft_data[lower_index:higher_index]

    fft_feature = {'frequency':freq_range, 'energy':energy_range}

    return fft_feature


def get_inverse_fft(fft_data):
    """Recover the signal by applying the inverse Fourier Transform on the
    fft_data."""
    filtered_data = np.fft.ifft(fft_data,axis=0)
    return filtered_data.real


def fft_train_cross_correlation():
    """Compute the cross-correlation function for all the fft training dataset,
    creating 16 correlation matrices. It returns a dictionary with keys as
    'channel_i', where i is the channel number (1 to 16), and the respective
    correlation matrix."""
    datas = np.loadtxt('../Seizures_Dataset/train_and_test_data_labels_safe.csv',
                       dtype='str', delimiter=',', skiprows=1, usecols=(0,1))
    datas = datas[0:N_TRAINING_SAMPLES,:] # Eliminate the test datas
    fft_train_xcorr = {}
    corrupted_list = []
    for c in xrange(16):
        key = "channel_%s" % (c+1)
        fft_train_xcorr[key] = np.zeros((3,3))
    
    for i in xrange(3):
        if corrupted_list.count(i) == 0:
            # File not corrupted
            corrupted_i = False
            
            # Load the i-th sample
            file_dir = "../Seizures_Dataset/train_%s/" % datas[i,0][0]
            file_path = file_dir + datas[i,0]
            try:
                sample_i = load_matdata(file_path)
                # Get spectrum in the range 8-30 Hz
                fft_feature_i = get_feature(sample_i)
                feature_i = fft_feature_i['energy'].real/N_SAMPLES_SEGMENT
            except KeyboardInterrupt:
                print "Interrupted!"
                break
            except:
                print "The file " + datas[i,0] + " is corrupted."
                # The file must be corrupted
                corrupted_i = True
                corrupted_list.append(i)
                continue
            
            if corrupted_i == False: # Check if the i-th file is corrupted
                for j in xrange(i,3):
                    if corrupted_list.count(j) == 0:
                        # Load the j-th sample
                        file_dir = "../Seizures_Dataset/train_%s/" % datas[j,0][0]
                        file_path = file_dir + datas[j,0]
                        try:
                            sample_j = load_matdata(file_path)
                            # Get spectrum in the range 8-30 Hz
                            fft_feature_j = get_feature(sample_j)
                            feature_j = fft_feature_j['energy'].real/N_SAMPLES_SEGMENT
                            for c in xrange(16):
                                key = "channel_%s" % (c+1)
                                # Correlation between the i-th sample and the j-th sample
                                print key + " - Correlation between:"
                                print str(i)+"-th sample: " + datas[i,0]
                                print str(j)+"-th sample: " + datas[j,0] + "\n"
                                corr = np.correlate(feature_i[:,c],
                                                    feature_j[:,c])
                                # The cross-correlation matrix is symmetric
                                fft_train_xcorr[key][i,j] = corr[0]
                                fft_train_xcorr[key][j,i] = corr[0]
                        except KeyboardInterrupt:
                            print "Interrupted!"
                            break
                        except:
                            print "The file " + datas[j,0] + " is corrupted."
                            corrupted_list.append(j)
                            continue
            
    # Save the cross-correlation matrices
    joblib.dump(fft_train_xcorr,'fft_train_xcorr.pkl')
    
    return fft_train_xcorr


def train_cross_correlation():
    """Compute the cross-correlation function for all the training dataset,
    creating 16 correlation matrices. It returns a dictionary with keys as
    'channel_i', where i is the channel number (1 to 16), and the respective
    correlation matrix."""
    datas = np.loadtxt('../Seizures_Dataset/train_and_test_data_labels_safe.csv',
                       dtype='str', delimiter=',', skiprows=1, usecols=(0,1))
    datas = datas[0:N_TRAINING_SAMPLES,:] # Eliminate the test datas
    train_xcorr = {}
    corrupted_list = []
    for c in xrange(16):
        key = "channel_%s" % (c+1)
        train_xcorr[key] = np.zeros((3,3))
    
    for i in xrange(3):
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
                continue
            
            if corrupted_i == False: # Check if the i-th file is corrupted
                for j in xrange(i,3):
                    if corrupted_list.count(j) == 0:
                        # Load the j-th sample
                        file_dir = "../Seizures_Dataset/train_%s/" % datas[j,0][0]
                        file_path = file_dir + datas[j,0]
                        try:
                            sample_j = load_matdata(file_path)
                            # Band-pass filter (8-30 Hz)
                            fft_sample_j, freq_sample_j = get_and_filter_fft(sample_j)
                            filtered_sample_j = get_inverse_fft(fft_sample_j)
                            
                            for c in xrange(16):
                                key = "channel_%s" % (c+1)
                                # Correlation between the i-th sample and the j-th sample
                                print key + " - Correlation between:"
                                print str(i)+"-th sample: " + datas[i,0]
                                print str(j)+"-th sample: " + datas[j,0] + "\n"
                                corr = np.correlate(filtered_sample_i[:,c],
                                                    filtered_sample_j[:,c])
                                # The cross-correlation matrix is symmetric
                                train_xcorr[key][i,j] = corr[0]
                                train_xcorr[key][j,i] = corr[0]
                        except KeyboardInterrupt:
                            print "Interrupted!"
                            break
                        except:
                            print "The file " + datas[j,0] + " is corrupted."
                            corrupted_list.append(j)
                            continue
            
    # Save the cross-correlation matrices
    joblib.dump(train_xcorr,'train_xcorr.pkl')
    
    return train_xcorr

def get_correlation_matrix():
    feature_dataset = joblib.load('feature_train_dataset.pkl')
    xcorr = {}
    for c in xrange(1,17):
        key = "channel_%s" % c
        print "Computing correlation matrix of " + key
        xcorr[key] = np.dot(np.transpose(feature_dataset[key]), feature_dataset[key])
        print "Finished the " + key
    print "Finished all! Now, saving..."
    joblib.dump(xcorr, 'train_xcorrelation_matrix.pkl')
    print "Saved in train_xcorrelation_matrix.pkl!"
    return True

# ================================ MAIN ================================== #
#fft_train_xcorr = fft_train_cross_correlation()
#train_xcorr = train_cross_correlation()

#feature_dataset = get_dataset('train')
xcorr_success = get_correlation_matrix()
