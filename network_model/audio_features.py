"""Extracts features from a single audio file.

This script extracts useful features for confidence assessment from a single
audio file.

Classes
--------
SingleFileFeatureExtraction: Extract features from a single audio file.
"""

import os
import scipy
import numpy as np
import pandas as pd
import librosa
import librosa.display
import timbral_models
import sklearn


class SingleFileFeatureExtraction:
    """
    This class extracts the features deemed relevant for confidence assessment
    in an audio.

    Public methods
    --------------
    load_audio(self, audio_path): Load audio from path provided.
    get_pos_of_item(self, elem, string_list): Get a list of all positions of
                                        an item in the list.
    get_transcription(self): Get the transcribed sentence.
    get_interjecting frequency: Get the frequency of interjecting sounds.
    get_energy(self): Calculate energy of audio.
    get_energy_entropy(self): Calculate the entropy of energy of audio.
    get_spectral_centroids(self): Calculate all spectral centroids.
    get_spectral_entropy(self): Calculate normalised spectral entropy.
    get_spectral_rolloff(self): Calculate all spectral rolloffs.
    get_spectral_contrast(self): Calculate all spectral contrasts.
    get_zero_crossings(self) Calculate the zero-crossing rate.
    get_mfcc(self): Get the MFCCs.
    get_autocorrelation(self): Get the autocorrelation array of aidio.
    get_pitch(self): Get pitch data.
    get_tonnetz(self): Get tonnetz data.
    get_sharp_rough(self): Get sharpness and roughness of audio.
    get_pause_ratio(self): Get the ratio of pausing to speaking of audio.
    get_repetition_rate(self): Get the number of successive repetition of words.
    write_features_to_csv(self): Write all features extracted to a csv file.
    """

    def __init__(
        self, home_dir, audio_path, feature_csv_folder_path, target_sampling_rate
    ):
        # Load paths
        self.home_dir = home_dir
        self.audio_path = audio_path
        self.feature_csv_folder_path = feature_csv_folder_path

        # Set parameters
        self.target_sampling_rate = target_sampling_rate
        self.frame_length = 1024
        self.hop_length = 512
        self.eps = 0.000000001
        self.n_scontrast_bands = 12
        self.n_mfcc = 12
        self.autocorrelation_max_size = 1000
        self.interjecting_sounds_list = [
            "Hmm",
            "eh",
            "oh",
            "ooh",
            "oops",
            "whoa",
            "wow",
            "well",
            "well well well",
        ]
        self.silent_threshold_ratio = 0.01

        # Audio properties
        self.audio_array = None
        self.sr = None
        self.audio_name = None
        self.audio_name_without_extension = None
        self.audio_folder_name = None
        self.audio_start_time = None
        self.transcript = None

        # Audio features
        self.interjecting_frequency = None
        self.energy = None
        self.feature_length = None
        self.energy_entropy = None
        self.spectral_centroids = None
        self.spectral_spread = None
        self.spectral_entropy = None
        self.spectral_rolloff = None
        self.spectral_contrast = None
        self.zero_crossing_rate = None
        self.mfcc = None
        self.autocorrelation = None
        self.pitches = None
        self.tonnetz = None
        self.pause_ratio = None
        self.repetition_rate = None
        self.single_feature_list = [
            "interjecting_frequency",
            "energy_entropy",
            "spectral_entropy",
            "pause_ratio",
            "repetition_rate",
        ]
        # self.sharpness = None
        # self.roughness = None

    def load_audio(self):
        """Load audio file according to path."""
        x, sr = librosa.load(self.audio_path)
        self.audio_array = x
        self.sr = sr

    def get_pos_of_item(self, elem, string_list):
        """
        Get position of all duplicates in a list.
        :return: a list of all positions for an element.
        """
        if elem in string_list:
            counter = 0
            elem_pos_list = []
            for i in string_list:
                if i == elem:
                    elem_pos_list.append(counter)
                counter = counter + 1
            return elem_pos_list

    def get_transcription(self):
        """
        Get the transcription of the audio clip.
        :return: assign a string of the sentence to self.
        """
        self.audio_folder_name = self.audio_path.split("/")[-2]
        confidence_dataframe_name = (
            "confidence_dataframe_" + self.audio_folder_name + ".csv"
        )
        self.audio_name = self.audio_path.split("/")[-1]
        self.audio_name_without_extension = self.audio_name.split("_")[-2]
        self.audio_start_time = self.audio_name.split("_")[-1][:-4]
        dataframe_path = os.path.join(
            self.home_dir,
            "data_sheets",
            "confidence_dataframes",
            confidence_dataframe_name,
        )
        dataframe = pd.read_csv(dataframe_path)
        filename_list = dataframe["filename"].tolist()
        pure_filename_list = [
            item.split("/")[-1].split(".")[0] for item in filename_list
        ]
        # Get positions of all entries with this audio file.
        elem_pos_list = self.get_pos_of_item(
            self.audio_name_without_extension, pure_filename_list
        )
        start_time_list = dataframe["start_time"].tolist()
        start_time_list = [s.replace("s", "") for s in start_time_list]
        sentence_list = dataframe["sentence"].tolist()
        for pos in elem_pos_list:
            start_time = start_time_list[pos].split(".")[0]
            # If same start time; only check until 0 dp
            if start_time == self.audio_start_time.split(".")[0]:
                self.transcript = sentence_list[pos]

    def get_interjecting_frequency(self):
        """
        Get the number of interjecting sounds per word.
        :return: assign floating number of frequency to self.
        """
        count = 0
        transcript_list = self.transcript.split()
        for interjecting_sound in self.interjecting_sounds_list:
            if interjecting_sound in transcript_list:
                new_count = transcript_list.count(interjecting_sound)
                count += new_count

        self.interjecting_frequency = count / len(transcript_list)

    def get_energy(self):
        """
        Get energy of audio.
        :return: assign single element numpy array of energy to self.
        """
        rms_energy = librosa.feature.rms(
            self.audio_array,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        # Square to get total energy
        energy = np.power(rms_energy, 2)
        self.energy = energy
        self.feature_length = len(self.energy[0])

    def get_energy_entropy(self):
        """
        Get energy entropy.
        :return: assign a floating number of entropy to self.
        """
        # Normalise energies to probabilities
        energy_norm = np.divide(self.energy, self.energy.sum())
        ee = -np.multiply(energy_norm, np.log2(energy_norm)).sum()
        self.energy_entropy = ee

    def get_spectral_centroids(self):
        """
        Get all spectral centroids and spectral spread.
        :return: assign a single element numpy array of spectral centroids to self.
        :return: assign a single element numpy array of spectral spread to self.
        """
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
        """
        Calculate normalised spectral entropy.
        :return: Assign a floating number of spectral entropy to self.
        """
        _, psd = scipy.signal.periodogram(self.audio_array, self.sr)

        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        se /= np.log2(psd_norm.size)
        self.spectral_entropy = se

    def get_spectral_rolloff(self):
        """
        Calculate spectral rolloff of audio.
        :return: assign a single-element array spectral rolloff to self.
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(
            self.audio_array + 0.01,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        self.spectral_rolloff = spectral_rolloff

    def get_spectral_contrast(self):
        """
        Calculate the spectral contrast for each subband.
        :return: assign np.ndarray [shape=(…, n_bands + 1, t)] to self.
        """
        spectral_contrast = librosa.feature.spectral_contrast(
            y=self.audio_array,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        self.spectral_contrast = spectral_contrast

    def get_zero_crossings(self):
        """
        Get the rate of zero crossings.
        :return: assign single element numpy array of zero crossing rate to self.
        """
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            self.audio_array, frame_length=self.frame_length, hop_length=self.hop_length
        )
        self.zero_crossing_rate = zero_crossing_rate

    def get_mfcc(self):
        """
        Get mfccs with the specified number of entries.
        :return: assign an array of subarray of mfccs to self.
        """
        mfccs = librosa.feature.mfcc(
            self.audio_array,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        self.mfcc = mfccs

    def get_autocorrelation(self):
        """
        Calculate the autocorrelation of the signal.
        :return: assign an array of autocorrelation to self.
        """
        if self.feature_length:
            max_auto_size = self.feature_length
        else:
            max_auto_size = self.autocorrelation_max_size

        r = librosa.autocorrelate(self.audio_array, max_size=max_auto_size)
        self.autocorrelation = r

    def get_pitch(self):
        """
        Get pitches.
        :return: assign an array of pitches to self.
        """
        pitches, magnitudes = librosa.core.piptrack(
            self.audio_array,
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

    def get_tonnetz(self):
        """
        Calculate tonnetz of the audio.
        :return: assign np.ndarray [shape(…, 6, t)] to self.
        """
        tonnetz = librosa.feature.tonnetz(y=self.audio_array, sr=self.sr)
        self.tonnetz = tonnetz

    def get_sharp_rough(self):
        """
        Calculate the sharpness of the audio file.
        :return: assign sharpness floating number to self.
        """
        # TODO: soundfile failing to load atm
        timbre_dict = timbral_models.timbral_extractor(self.audio_path)
        self.sharpness = timbre_dict["sharpness"]
        self.roughness = timbre_dict["roughness"]

    def get_pause_ratio(self):
        """
        Calculate the period of pausing versus speaking rate.
        :return: assign a floating number of pause to speaking ratio to self.
        """
        num_pause = (self.audio_array < np.amax(self.audio_array) * 0.01).sum()
        num_speaking = self.audio_array.size - num_pause
        self.pause_ratio = float(num_pause / num_speaking)

    def get_repetition_rate(self):
        """
        Count the number of successive repetitions.
        :return: assign an integer of repetition rate to self.
        """
        count = 0
        transcript_list = self.transcript.split()
        for i in range(len(transcript_list) - 1):
            if transcript_list[i + 1] == transcript_list[i]:
                count += 1
        self.repetition_rate = count / len(transcript_list)

    def normalize(self, list):
        """Normalise the list column-wise."""
        max_value = max(list)
        min_value = min(list)
        result = [(i - min_value) / (max_value - min_value) for i in list]

        return result

    def write_features_to_csv(self):
        """Extract all features and write to a csv."""
        # Extract audio features
        frames = []

        # Audio array
        self.load_audio()
        audio_df = pd.DataFrame(self.audio_array.tolist(), columns=["audio_array"])
        frames.append(audio_df)

        # Text
        self.get_transcription()
        text_df = pd.DataFrame([self.transcript], columns=["text"])
        frames.append(text_df)

        # Interjecting frequency
        self.get_interjecting_frequency()
        frequency_df = pd.DataFrame(
            [self.interjecting_frequency], columns=["interjecting_frequency"]
        )
        frames.append(frequency_df)

        # Energy
        self.get_energy()
        energy_scaled = self.normalize(self.energy[0].tolist())
        energy_df = pd.DataFrame(energy_scaled, columns=["energy"])
        frames.append(energy_df)

        # Energy Entropy
        self.get_energy_entropy()
        ee_df = pd.DataFrame([self.energy_entropy], columns=["energy_entropy"])
        frames.append(ee_df)

        # Spectral centroids/spectral spread
        self.get_spectral_centroids()
        sc_scaled = self.normalize(self.spectral_centroids[0].tolist())
        sc_df = pd.DataFrame(sc_scaled, columns=["spectral_centroids"])
        frames.append(sc_df)
        ss_scaled = self.normalize(self.spectral_spread[0].tolist())
        ss_df = pd.DataFrame(ss_scaled, columns=["spectral_spread"])
        frames.append(ss_df)

        # Spectral entropy
        self.get_spectral_entropy()
        se_df = pd.DataFrame([self.spectral_entropy], columns=["spectral_entropy"])
        frames.append(se_df)

        # Spectral rolloff
        self.get_spectral_rolloff()
        sr_scaled = self.normalize(self.spectral_rolloff[0].tolist())
        sr_df = pd.DataFrame(sr_scaled, columns=["spectral_rolloff"])
        frames.append(sr_df)

        # Spectral contrast
        self.get_spectral_contrast()
        # Special care taken for the 13 scontrasts
        scontrast_list = self.spectral_contrast.tolist()
        for i in range(len(scontrast_list)):
            data_column = self.normalize(scontrast_list[i])
            column_name = "spectral_contrast" + str(i)
            scontrast_indiv_df = pd.DataFrame(data_column, columns=[column_name])
            frames.append(scontrast_indiv_df)

        # Zero crossing rate
        self.get_zero_crossings()
        zcr_scaled = self.normalize(self.zero_crossing_rate[0].tolist())
        zcr_df = pd.DataFrame(zcr_scaled, columns=["zero_crossing_rate"])
        frames.append(zcr_df)

        # mfccs
        self.get_mfcc()
        # Special care taken for the 12 mfccs
        mfcc_list = self.mfcc.tolist()
        for i in range(len(mfcc_list)):
            data_column = self.normalize(mfcc_list[i])
            column_name = "mfcc" + str(i)
            mfcc_indiv_df = pd.DataFrame(data_column, columns=[column_name])
            frames.append(mfcc_indiv_df)

        # Autocorrelation
        self.get_autocorrelation()
        auto_scaled = self.normalize(self.autocorrelation.tolist())
        autocorrelation_df = pd.DataFrame(auto_scaled, columns=["autocorrelation"])
        frames.append(autocorrelation_df)

        # Pitches
        self.get_pitch()
        pitches_scaled = self.normalize(self.pitches.tolist())
        pitches_df = pd.DataFrame(pitches_scaled, columns=["pitches"])
        frames.append(pitches_df)

        # Tonnetz
        self.get_tonnetz()
        # Special care taken for the 6 tonnetz dimensions
        tonnetz_list = self.tonnetz.tolist()
        for i in range(len(tonnetz_list)):
            data_column = self.normalize(tonnetz_list[i])
            column_name = "tonnetz" + str(i)
            tonnetz_indiv_df = pd.DataFrame(data_column, columns=[column_name])
            frames.append(tonnetz_indiv_df)

        # Pause ratio
        self.get_pause_ratio()
        pr_df = pd.DataFrame([self.pause_ratio], columns=["pause_ratio"])
        frames.append(pr_df)

        self.get_repetition_rate()
        rr_df = pd.DataFrame([self.repetition_rate], columns=["repetition_rate"])
        frames.append(rr_df)

        # # Sharpness/ roughness
        # self.get_sharp_rough()
        # sharp_df = pd.DataFrame([self.sharpness],
        #                      columns=["sharpness"])
        # frames.append(sharp_df)
        # rough_df = pd.DataFrame([self.roughness],
        #                         columns=["roughness"])
        # frames.append(rough_df)

        total_df = pd.concat(frames, axis=1)
        feature_csv_path = os.path.join(
            self.feature_csv_folder_path, str(self.audio_name[:-4] + ".csv")
        )
        total_df.to_csv(feature_csv_path)


def audio_path_in_dir(folder_path_list):
    """
    :param folder_path_list: The path of the parent-parent folder of audio.
    :return: Clean list containing the absolute filepaths of all audio files.
    """
    file_path_list = []
    for folder_path in folder_path_list:
        for filename in os.listdir(folder_path):
            if filename.endswith("mp3"):
                file_path_list.append(os.path.join(folder_path, filename))
    return file_path_list


# Normalising the spectral centroid
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


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