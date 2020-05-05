#!/usr/bin/env python

from __future__ import division
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import cv2
import wave
import struct
from scipy import signal

img = cv2.imread("example_image.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

first_row = img[50:51,][0]
#first_row = np.concatenate((first_row, first_row, first_row))

first_row = np.add(first_row, -np.mean(first_row), casting="unsafe")

first_row *= 15#np.std(first_row)
sampled = signal.resample(first_row, int(len(first_row)*(2000/44100)))
data = sampled#np.tile(sampled, 1000)
print(data)
#samples = (np.sin(2*np.pi*np.arange(44100*1)*440/44100)).astype(np.float32)
obj = wave.open('sound.wav','w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(44100)
for i in range(len(data)):
   data_point = struct.pack('<h', data[i])
   obj.writeframesraw(data_point)
obj.close()

'''
w = np.fft.fft(data)
freqs = np.fft.fftfreq(len(w))
plt.plot(freqs)
plt.show()
print(freqs.min(), freqs.max())
# (-0.5, 0.499975)

# Find the peak in the coefficients
idx = np.argmax(np.abs(w))
freq = freqs[idx]
freq_in_hertz = abs(freq)
print(freq_in_hertz)
'''
fourier = np.absolute(fft(first_row)) / 10000
print fourier
plt.plot(first_row)
#plt.plot(sampled)
plt.show()
