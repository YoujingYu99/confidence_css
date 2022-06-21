"""Extracts features from a single audio file.

This script extracts useful features for confidence assessment from a single
audio file.

Classes
--------
SingleFileFeatureExtraction: Extract features from a single audio file.

"""

import os
import sklearn
import scipy
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


def audio_path_in_dir(folder_path_list):
    """
    :param folder_path_list: The path of the parent-parent folder of audio.
    :return: Clean list containing the absolute filepaths of all audio files.
    """
    file_path_list = []
    # for filename in os.listdir(folder_path_list):
    #     if filename.endswith("mp3"):
    #         file_path_list.append(os.path.join(folder_path, filename))
    for folder_path in folder_path_list:
        for filename in os.listdir(folder_path ):
            if filename.endswith("mp3"):
                file_path_list.append(os.path.join(folder_path, filename))
    return file_path_list


# Normalising the spectral centroid
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


class SingleFileFeatureExtraction:
    """
    This class extracts the features deemed relevant for confidence assessment
    in an audio.

    Public methods
    --------------
    load_audio(self, audio_path): Load audio from path provided.
    get_energy(self): Calculate energy of audio.
    get_spectral_centroids(self): Calculate all spectral centroids.
    get_zero_crossings(self) Calculate the zero-crossing rate.
    get_mfcc(self): Get the MFCCs.
    get_pitch(self): Get pitch data.
    write_features_to_csv(self): Write all features extracted to a csv file.
    """

    def __init__(self, audio_path, feature_csv_folder_path):
        self.audio_path = audio_path
        self.feature_csv_folder_path = feature_csv_folder_path
        self.feature_dataframe = pd.DataFrame()
        self.frame_length = 1024
        self.hop_length = self.frame_length / 2
        self.eps = 0.000000001
        self.audio_array = None
        self.sr = None

        # Audio features
        self.energy = None
        self.spectral_centroids = None
        self.spectral_spread = None
        self.spectral_entropy = None
        self.zero_crossing_rate = None
        self.mfcc = None
        self.pitches = None

    def load_audio(self):
        """Load audio file according to path."""
        x, sr = librosa.load(self.audio_path)
        self.audio_array = x
        self.sr = sr

    def get_energy(self):
        """Get rms of audio"""
        energy = librosa.feature.rms(
            y=self.audio_array,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        self.energy = energy

    def get_spectral_centroids(self):
        """Get all spectral centroids and spectral spread."""
        # https: // github.com / novoic / surfboard / blob / 700
        # d1b26e80fbd06d2a9c682260e6c635d6c0d40 / surfboard / spectrum.py  # L8
        frame = librosa.util.frame(
            self.audio_array, frame_length=self.frame_length, hop_length=self.hop_length
        )
        S = np.abs(
            librosa.stft(
                self.audio_array,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                center=False,
            )
        )
        sc = librosa.feature.spectral_centroid(
            self.audio_array,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            center=False,
        )

        freq = np.arange(0, len(S), 1) * (self.sr / (2.0 * (len(S) - 1)))

        # spectral spread
        spread = np.zeros((1, frame.shape[1]))
        for i in range(frame.shape[1]):
            spread[:, i] = np.sqrt(
                (sum((freq - sc[:, i]) ** 2 * S[:, i])) / sum(S[:, i])
            )

        self.spectral_centroids = sc
        self.spectral_spread = spread

    def get_spectral_entropy(self):
        """Calculate normalised spectral entropy."""
        _, psd = scipy.signal.periodogram(self.audio_array, self.sr)

        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        se /= np.log2(psd_norm.size)
        self.spectral_entropy = se

    def get_zero_crossings(self):
        """Get the rate of zero crossings."""
        zero_crossings = librosa.zero_crossings(
            self.audio_array,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            pad=False,
        )
        self.zero_crossing_rate = zero_crossings / len(self.audio_array)

    def get_mfcc(self):
        """Get mfccs."""
        mfccs = librosa.feature.mfcc(
            self.audio_array,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        self.mfcc = mfccs

    def get_pitch(self):
        """Get pitches"""
        pitches, magnitudes = librosa.core.piptrack(
            y=self.audio_array,
            sr=self.sr,
            fmin=75,
            fmax=1600,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        # get indexes of the maximum value in each time slice
        max_indexes = np.argmax(magnitudes, axis=0)
        # get the pitches of the max indexes per time slice
        pitches = pitches[max_indexes, range(magnitudes.shape[1])]
        self.pitches = pitches

    def write_features_to_csv(self):
        self.load_audio()
        self.get_energy()
        self.get_spectral_centroids()
        self.get_spectral_entropy()
        self.get_zero_crossings()
        self.get_mfcc()
        self.get_pitch()
        self.feature_dataframe["energy"] = self.energy
        self.feature_dataframe["spectral_centroids"] = self.spectral_centroids
        self.feature_dataframe["spectral_spread"] = self.spectral_spread
        self.feature_dataframe["spectral_entropy"] = self.spectral_entropy
        self.feature_dataframe["zero_crossing_rate"] = self.zero_crossing_rate
        self.feature_dataframe["mfcc"] = self.mfcc
        self.feature_dataframe["pitches"] = self.pitches

        audio_name = self.audio_path.split("/")[-1] + ".csv"
        feature_csv_path = os.path.join(self.feature_csv_folder_path, audio_name)
        self.feature_dataframe.to_csv(feature_csv_path)


#
# for audio_path in audio_path_list:
#     # loading audio files
#     x, sr = librosa.load(audio_path)
# print(type(x), type(sr))#<class 'numpy.ndarray'> <class 'int'>
# sr is 22050
# print(x.shape, sr)
# #plot the waveform
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()

# # plot the spectrogram
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# # using log scale
# #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()

# #.spectral_centroids will return an array with columns equal to a number of frames present in your sample.
# spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# # Computing the time variable for visualization
# plt.figure(figsize=(12, 4))
# frames = range(len(spectral_centroids))
# t = librosa.frames_to_time(frames)
# # Plotting the Spectral Centroid along the waveform
# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_centroids), color='b')
# plt.show()
#
#
# # Spectral bandwidth
# spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
# spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=3)[0]
# spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=4)[0]
# plt.figure(figsize=(15, 9))
# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_bandwidth_2), color='r')
# plt.plot(t, normalize(spectral_bandwidth_3), color='g')
# plt.plot(t, normalize(spectral_bandwidth_4), color='y')
# plt.legend(('p = 2', 'p = 3', 'p = 4'))

# # Zero-crossing rate
# n0 = 9000
# n1 = 9100
# zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
# print(sum(zero_crossings))

# # MFCCS
# mfccs = librosa.feature.mfcc(x, sr=sr)
# print(mfccs.shape)
# # Displaying  the MFCCs:
# plt.figure(figsize=(15, 7))
# librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# plt.show()

# # Chroma Features
# hop_length = 512
# chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

# # Spectral Rolloff
# # Compute the time variable for visualisation
# spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# frames = range(len(spectral_centroids))
# t = librosa.frames_to_time(frames)
#
# spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_rolloff), color='r')

# # Spectral Contrast
# spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
# # spectral_contrast.shape
# plt.imshow(normalize(spectral_contrast, axis=1), aspect='auto', origin='lower', cmap='coolwarm')
# plt.show()

# # Tonnetz
# tonnetz = np.mean(librosa.feature.tonnetz(y=x, sr=sr))
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(tonnetz, y_axis="tonnetz")
# plt.show()
