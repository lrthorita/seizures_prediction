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
from sklearn import svm
from scipy import signal
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


def butterworth_bandpass_filter(data, order, f_min, f_max):
    """Apply a Butterworth bandpass filter with the cut frequencies in
    (f_min, f_max) on the data signal."""
    # The Nyquist frequency
    f_nyquist = 0.5 * IEEG_SAMPLING_RATE    # 200Hz
    # The f_nyquist fractions, where there are cut frequencies
    lower = f_min / f_nyquist
    higher = f_max / f_nyquist
    # Numerator and denominator of the bandpass filter
    n, d = signal.butter(order, [lower, higher], btype="band")
    # Apply filter
    return signal.lfilter(n, d, data, axis=0)


def crop_spectrum(data, min_freq, max_freq):
    """Get the frequency range where alpha, beta and mu waves are situated.
    These waves are the waves that can, possibly, advise a future seizure 
    event. The frequency range is between 8 Hz and 30 Hz."""
    lower_index = np.int(min_freq * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE)
    higher_index = np.int(max_freq * N_SAMPLES_SEGMENT / IEEG_SAMPLING_RATE) + 1

    fft_data, frequencies = get_fft(data)

    freq_range = frequencies[lower_index:higher_index]
    fft_range = fft_data[lower_index:higher_index]

    return fft_range, freq_range


def compute_energy(fft_data):
    """Compute the spectral energy of a signal.
                E = |X(f)|²"""
    abs_fft_data = abs(fft_data)/(N_SAMPLES_SEGMENT/2)
    energy = np.zeros(16)
    for c in xrange(16):
        energy[c] = np.dot(np.transpose(abs_fft_data[:,c]),abs_fft_data[:,c])
    return energy


def get_feature(data, half_bw=2):
    """For each channel of the data, apply 10 Butterworth bandpass filters in
    different ranges of frequencies (from 8 Hz to 35 Hz) and concatenate the 
    energy of these spectrals in a unique 10-dimension feature vector. Also, concatenate
    the feature vectors of the 16 channels in just one vector, resulting in a
    160-dimension feature vector."""
    feature = np.zeros((1,160))
    # Get the feature vector for each of the 16 channels
    for i in xrange(10):
        # Apply Butterworth bandpass filter
        f_center = 3*i + 8
        f_min = f_center - half_bw
        f_max = f_center + half_bw
        filtered_data = butterworth_bandpass_filter(data, 5, f_min, f_max)
        
        # Crop the spectral signal on the bandwidth region
        min_freq = f_min - 3
        max_freq = f_max + 3
        fft_cropped, freq_cropped = crop_spectrum(filtered_data, min_freq, max_freq)
        
        # Compute the approximated spectral energy
        energy = compute_energy(fft_cropped)
        
        # Save energy to the feature vector
        for c in xrange(16):
            feature[0, 10*c + i] = energy[c]
        
    return feature
    
    
def get_features_dataset(mode):
    """Compute feature for every samples in the training dataset."""
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
    
    
    features_dataset = {}
    for i in xrange(N_SAMPLES):
        # Load the i-th sample
        file_dir = directory + "%s/" % datas[i,0][0]
        file_path = file_dir + datas[i,0]
        try:
            # Get the data
            data = load_matdata(file_path)
            # Compute the feature vector
            feature = get_feature(data,2)
            
            if features_dataset.has_key('features'):
                features_dataset['features'] = np.r_[features_dataset['features'], feature]
                features_dataset['filenames'].append(datas[i,0])
                features_dataset['labels'].append(datas[i,1])
            else:
                features_dataset['features'] = feature
                features_dataset['filenames'] = [datas[i,0]]
                features_dataset['labels'] = [datas[i,1]]
            
            print str(i) + "-th feature added."
                
        except KeyboardInterrupt:
            print "Interrupted!"
            break
        except:
            if features_dataset.has_key('corrupted'):
                features_dataset['corrupted'].append(datas[i,0])
            else:
                features_dataset['corrupted'] = [datas[i,0]]
            print "The file " + datas[i,0] + " is corrupted."
            continue
    
    # Save the cross-correlation matrices
    filename = "features_" + mode + "_dataset.pkl"
    print "Finished all! Now, saving..."
    joblib.dump(features_dataset, filename)
    print "Saved in " + filename +" file."
    
    return features_dataset


def train_svm(dataset):
    # Create an object of SVC class
    classifier = svm.SVC(C=0.1, kernel='poly', coef0=0.0, degree=2, tol = 0.001)
    
    # Divide the dataset in examples and labels
    examples = dataset['features']
    labels = dataset['labels']
    
    # Train the SVM Classifier to fit the dataset
    classifier.fit(examples, labels)
    
    # Save the cross-correlation matrices
    filename = "trained_classifier.pkl"
    print "Finished all! Now, saving..."
    joblib.dump(classifier, filename)
    print "Saved in " + filename +" file."
    
    return classifier


def test_svm(dataset, classifier):
    # Divide the dataset in examples and labels
    examples = dataset['features']
    labels = dataset['labels']

    # Test classifier
    training_accuracy = classifier.score(examples, labels)
    print "The achieved training accuracy is: " + str(training_accuracy)
    return training_accuracy



# ================================ MAIN ================================== #
features_train_dataset = get_features_dataset('train')


