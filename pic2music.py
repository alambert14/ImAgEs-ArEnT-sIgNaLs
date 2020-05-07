#!/usr/bin/env python

from __future__ import division
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import cv2
import os
import wave
import struct
from scipy import signal
from sklearn.cluster import KMeans

class MusicalPicture(object):
    
    SAMPLE_RATE = 44100
    NUM_PITCHES = 7
    
    #          F      F#/Gb  G      G#/Ab  A      A#/Bb  B      C      C#/Db  D      D#/Eb  E
    PITCHES = [349.2, 370.0, 392.0, 415.3, 440.0, 466.2, 493.9, 523.2, 554.4, 587.3, 622.2, 659.3]

    def __init__(self, filename, duration, num_sections):
        self.image = cv2.imread(filename)
        print(self.image.shape)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]
        self.duration = duration
        self.num_sections = num_sections
        self.sections = self.generate_sections() # List of len num_sections
        self.dominant_colors = self.find_dominant_colors() # List of tuples, len num_sections (frequency, hue, saturation, value)
        self.hue_to_pitch = self.calculate_hue_to_pitch()
        self.pitches = self.calculate_pitches()

    def generate_sections(self):
        sections = []
        section_size = self.width//self.num_sections
        for i in range(self.num_sections-2):
            new_section = self.image[:,i*section_size:(i+1)*section_size,:]
            sections.append(new_section)
        # The last section will likely be different width
        sections.append(self.image[:,(self.num_sections-1)*section_size:,:].astype(np.uint8))
        return sections

    
    def generate_single_wav(self, section_number):
        """
        Generates a wav file of a single period of a sound shaped by the value of the colors
        in a particular section of the image
        
        Args:
            section_number: the section of the image to sonify
        Returns:
            None
        """

        if section_number >= self.num_sections:
            raise IndexError("Invalid section number")
        section = self.sections[section_number]
        grayscale = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
        grayscale = grayscale.reshape((grayscale.shape[0], grayscale.shape[1]))
        averaged = np.mean(grayscale, axis=1)
        averaged = np.add(averaged, -np.mean(averaged), casting="unsafe")
        print(averaged.shape)
        sampled = signal.resample(averaged.astype(float), int(len(averaged)*(self.width/MusicalPicture.SAMPLE_RATE)))

        dir_name = "section_{}".format(section_number)

        try:
            # Create target Directory
            os.mkdir(dir_name)
            print("Directory " , dir_name ,  " Created ")
        except FileExistsError:
            print("Directory " , dir_name ,  " already exists")

        output_wav = "{}/instrument.wav".format(dir_name)
        obj = wav.open(output_wav, 'w')
        obj.setnchannels(1)
        obj.setframerate(MusicalPicture.SAMPLE_RATE)
        
        for i in range(len(sampled)):
            data_point = struct.pack('<h', data[i])
            obj.writeframesraw(data_point)
            obj.close()

    def generate_all_wavs(self):
        for i in range(self.num_sections):
            self.generate_single_wav(i)


    def find_dominant_colors(self):
        
        '''
        Adapted from: https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
        '''

        section_results = []
        for section in self.sections:
            print(section.shape)
            section = cv2.cvtColor(section, cv2.COLOR_BGR2HSV)
            image = section.reshape((section.shape[0] * section.shape[1], 3)) # see if this can just be clustered by hue
            clt = KMeans(n_clusters = MusicalPicture.NUM_PITCHES)
            clt.fit(image)
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins = numLabels)
            # normalize the histogram, such that it sums to one
            hist = hist.astype("float")
            hist /= hist.sum()

            frequencies = []
            hues = []
            saturations = []
            values = []
            for (percent, color) in zip(hist, clt.cluster_centers_):
                frequencies.append(percent)
                hues.append(color[0])
                saturations.append(color[1])
                values.append(color[2])

            section_results.append((frequencies, hues, saturations, values))
        return section_results

    def calculate_hue_to_pitch(self):
        hue_to_pitch = {}
        multiple = round(180/len(MusicalPicture.PITCHES), 1)
        for i in range(len(MusicalPicture.PITCHES)):
            hue_to_pitch[i*multiple] = MusicalPicture.PITCHES[i]
        return hue_to_pitch

    def closest_pitch(self, hue):
        hues = list(self.hue_to_pitch.keys())
        # find the closest hue by minimizing squared error
        closest_ind = np.argmin((hues-hue)**2) # check the syntax on this
        return hues[closest_ind]
    
    def calculate_pitches(self):
        """
        Resources: 
            - https://cycling74.com/forums/patch-for-pitch-to-color
            - https://www.flutopedia.com/sound_color.htm
        """
        pitches = []
        for section in self.dominant_colors:
            section_pitches = []
            for i in range(len(section[1])): # hues
                hue = section[1][i]
                saturation = section[2][i]
                central_pitch = self.closest_pitch(hue)
                if saturation > 50: # can modify to have more octave range
                    central_pitch *= 2
                section_pitches.append(central_pitch)
            pitches.append(section_pitches)
        return pitches
    
    
    def write_pitches(self, section):
        if section >= self.num_sections:
            raise IndexError("Invalid section number")
        dir_name = "section_{}".format(section)
 
        try:
            # Create target Directory
            os.mkdir(dir_name)
            print("Directory " , dir_name ,  " Created ") 
        except OSError:
            print("Directory " , dir_name ,  " already exists")
        filename = "{}/pitches.txt".format(dir_name)

        output = "/pitches = " + str(self.pitches)
        with open(filename, 'w+') as f:
            f.write(output)

if __name__ == "__main__":
    yay = MusicalPicture("example_image_small.jpg", 120, 5)
    #yay.generate_all_wavs()
    yay.write_pitches(0)
    yay.write_pitches(1)
    yay.write_pitches(2)
    yay.write_pitches(3)
    yay.write_pitches(4)
