#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# *************************************************************************** #
# Author:  LUIZ RICARDO TAKESHI HORITA
# Master's Degree - Electrical and Computer Engineering
# University of Sao Paulo - Sao Carlos, SP, Brazil
# *************************************************************************** #
import numpy as np
import scipy.io

def load_matdata(filename):
    """Take the mat file and convert it in a dictionary variable with the 
    following keys: data, iEEsamplingRate, nSamplesSegment, channelIndices."""
    matdata = scipy.io.loadmat(filename)
    dataStruct = matdata['dataStruct']
    data = {}
    data['data'] = dataStruct['data'][0,0]
    data['iEEGsamplingRate'] = dataStruct['iEEGsamplingRate'][0,0][0,0]
    data['nSamplesSegment'] = np.int(dataStruct['nSamplesSegment'][0,0][0,0])
    data['channelIndices'] = dataStruct['channelIndices'][0,0]
    return data

def get_fft(data):
    """Compute the Fast Fourier Transform on the signal of each channel of the
    data, and return the histogram of frequencies energy."""
    samplingPeriod = 1/data['iEEGsamplingRate']
    nSamples = data['nSamplesSegment']

    fft_data = abs(np.fft.fft(data['data'],axis=0))/(nSamples/2)
    frequencies = np.fft.fftfreq(nSamples,samplingPeriod)

    return fft_data, frequencies

def get_feature(data):
    """Get the frequency range where alpha, beta and mu waves are situated.
    These waves are the waves that can, possibly, advise a future seizure 
    event. The frequency range is between 8 Hz and 30 Hz."""
    lower_index = np.int(8 * data['nSamplesSegment'] / data['iEEGsamplingRate'])
    higher_index = np.int(30 * data['nSamplesSegment'] / data['iEEGsamplingRate'])

    fft_data, frequencies = get_fft(data)

    freq_range = frequencies[lower_index:higher_index]
    fft_data_range = fft_data[lower_index:higher_index]

    return fft_data_range, freq_range












