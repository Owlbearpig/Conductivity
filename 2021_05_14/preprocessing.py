import numpy as np
from scipy.signal import windows, butter, filtfilt, freqz
import matplotlib.pyplot as plt

def rotate(t):
    tc = np.copy(t)
    if np.size(t,0) > 1: 
        for i, temp in enumerate(t):
            tc[i] = temp[-1::-1]
    else:
        t[0] = t[0][-1::-1]
    return tc

def offset(t,a, range):
    tc = np.copy(t)
    ac = np.copy(a)
    if len(t.shape) == 1: # only a single time vector
        idx = t <= t[0]+range
        if len(a.shape) == 1: # if np.size(a,0) == 1:
            ac -= np.mean(a[idx]) # offset correction
        else:
            for i, temp in enumerate(a):
                ac[i] -= np.mean(temp[idx]) # offset correction
    else: # more time vectors
        if np.size(a,0) == np.size(t,0): # then the number of time vectors should equal amplitude vectors
            i = 0
            for temp1, temp2 in zip(t,a):
                idx = temp1 <= temp1[0]+range
                ac[i] -= np.mean(temp2[idx]) # offset correction
                i += 1
        else:
            print('Warning: number of time vectors does not match number of amplitude vectors by more than one time vector')
    return tc, ac

def fft(t, a, n = None, windowing = True, alpha = 0.05, ignore_nan = False):
    freq = []
    amp = []
    ac = np.copy(a)
    if len(t.shape) == 1: # only a single time vector
        dt = np.mean(np.diff(t))
        if len(ac.shape) == 1:
            if ignore_nan:
                idx = np.isnan(ac)
                ac = ac[idx]
            if n == None:
                n = len(ac)
            if windowing:
                window = windows.tukey(len(ac), alpha, sym=False)
                ac *= window
            amp.append(np.fft.fft(ac,n))
            freq.append(np.fft.fftfreq(n)/dt)
        else:
            if n == None:
                n = len(ac[0])
            if windowing and not ignore_nan:
                window = windows.tukey(len(ac[0]), alpha, sym=False) # if there is only a single time vector, then all amplitude vector should be of same length
            for temp in ac:
                if ignore_nan:
                    idx = np.isnan(temp)
                    temp = temp[~idx]
                if windowing: 
                    if ignore_nan:
                        window = windows.tukey(len(temp), alpha, sym=False)                        
                    temp *= window
                amp.append(np.fft.fft(temp,n))
            freq.append(np.fft.fftfreq(n)/dt) # only one time vector
    else: # more time vectors
        if ignore_nan:
            print('ignore_none is not implemented for multiple time vecotrs!')
            float('0,df')
        if len(ac.shape) == len(t.shape): # then the number of time vectors should equal amplitude vectors
            i = 0 # counter makes sense only for lists. Numpy array have to have the same length either way
            for temp1, temp2 in zip(t,ac):
                if n == None:
                    n = len(ac[i])
                dt = np.mean(np.diff(temp1))
                if windowing:
                    window = windows.tukey(len(temp2), alpha, sym=False)
                    temp2 *= window
                amp.append(np.fft.fft(temp2,n))
                freq.append(np.fft.fftfreq(n)/dt )
                i += 1
        else:
            print('Warning: number of time vectors does not match number of amplitude vectors by more than one time vector')
    return np.asarray(freq), np.asarray(amp)

def plot_freq_response(b, a, fs, worN = 8000):
    # Plot the frequency response.
    w, h = freqz(b, a, worN=worN)
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    #plt.plot(highcut, 0.5*np.sqrt(2), 'ko')
    #plt.axvline(highcut, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Filter Frequency Response")
    plt.xlabel('Frequency (THz)')
    plt.grid()
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
     
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, plot = False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    f = filtfilt (b, a, data)
    if plot:
        plot_freq_response(b,a, fs)   
    return f

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5, plot=False, ignore_nan=False):
    b, a = butter_lowpass(cutoff, fs, order=order)
    if ignore_nan:
        if len(data.shape) == 1:
            idx = np.isnan(data)
            temp = data[~idx]
            temp = filtfilt(b, a, temp)
            y = np.copy(data)
            y[~idx] = temp
        else:
            y =  []
            for temp in data:
                idx = np.isnan(temp)
                tmp = temp[~idx]
                tmp =  filtfilt(b, a, tmp)
                tmp2 = np.copy(temp)
                tmp2[~idx] = tmp
                y.append(tmp2)
            y = np.asanyarray(y)
    else:
        y = filtfilt(b, a, data)
    if plot:
        plot_freq_response(b,a, fs)            
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5, plot=False, ignore_nan=False):
    b, a = butter_highpass(cutoff, fs, order=order)
    if ignore_nan:
        if len(data.shape) == 1:
            idx = np.isnan(data)
            temp = data[~idx]
            temp = filtfilt(b, a, temp)
            y = np.copy(data)
            y[~idx] = temp
        else:
            y =  []
            for temp in data:
                idx = np.isnan(temp)
                tmp = temp[~idx]
                tmp =  filtfilt(b, a, tmp)
                tmp2 = np.copy(temp)
                tmp2[~idx] = tmp
                y.append(tmp2)
            y = np.asanyarray(y)
    else:
        y = filtfilt(b, a, data)
    if plot:
        plot_freq_response(b,a, fs)            
    return y
    
