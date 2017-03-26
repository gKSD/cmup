#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re
import os
import numpy


from loader import Loader
from config import Config
from storage import Storage

class Classifier:
    def __init__(self, storage, directory = None):
        self._storage = storage
        self._loader = Loader(self._storage)

        self._resultFeaturesDirectory = directory if directory else Config.get_instance().classifierAudioFeaturesResultDirectory

    def _loadDataProcess_impl_(self, input_data):
        config = Config.get_instance()

        self._loader.load(input_data)
        self._loader.printData()
        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory)

    def _loadDataProcess_(self, data, file_name):
        if data:
            self._loadDataProcess_impl_(data)
        if file_name:
            data = ''
            with open(file_name, 'r') as datafile:
                data = datafile.read().replace('\n', '')
            self._loadDataProcess_impl_(data)

    def loadData(self, data):
        self._loadDataProcess_(data, None)

    def loadDataFromFile(self, file_name):
        self._loadDataProcess_(None, file_name)

    def __addFeature__(self, valid_features, valid_types, feature, value, result_features):
        if not (valid_features == "all" or "all" in valid_features or feature in valid_features):
            return result_features

        aggregated_features = [
                                "spectral_complexity",
                                "silence_rate_20dB",
                                "erbbands_spread",
                                "spectral_kurtosis",
                                "barkbands_kurtosis",
                                "spectral_strongpeak",
                                "spectral_spread",
                                "spectral_rms",
                                "erbbands",
                                "zerocrossingrate",
                                "spectral_contrast_coeffs",
                                "dissonance",
                                "spectral_energyband_high",
                                "spectral_skewness",
                                "spectral_flux",
                                "silence_rate_30dB",
                                "spectral_energyband_middle_high",
                                "barkbands_spread",
                                "spectral_centroid",
                                "pitch_salience",
                                "erbbands_skewness",
                                "erbbands_crest",
                                "melbands",
                                "spectral_entropy",
                                "spectral_rolloff",
                                "barkbands",
                                "melbands_flatness_db",
                                "melbands_skewness",
                                "barkbands_skewness",
                                "silence_rate_60dB",
                                "spectral_energyband_low",
                                "spectral_energyband_middle_low",
                                "melbands_kurtosis",
                                "spectral_decrease",
                                "erbbands_kurtosis",
                                "melbands_crest",
                                "gfcc",
                                "melbands_spread",
                                "spectral_energy",
                                "mfcc",
                                "spectral_contrast_valleys",
                                "barkbands_flatness_db",
                                "erbbands_flatness_db",
                                "hfc",
                                "barkbands_crest",
                                "hpcp_entropy",
                                "chords_strength",
                                "hpcp",
                                "bpm_histogram_second_peak_bpm",
                                "bpm_histogram_second_peak_spread",
                                "beats_loudness",
                                "bpm_histogram_first_peak_spread",
                                "bpm_histogram_first_peak_weight",
                                "beats_loudness_band_ratio",
                                "bpm_histogram_second_peak_weight",
                                "bpm_histogram_first_peak_bpm",
                                "audio_properties",
                                "version",
                                "tags"
                              ]
        simple_features = [
                                "average_loudness",
                                "dynamic_complexity",
                                "thpcp",
                                "tuning_diatonic_strength",
                                "chords_number_rate",
                                "key_strength",
                                "key_scale",
                                "key_key",
                                "chords_changes_rate",
                                "chords_scale",
                                "tuning_nontempered_energy_ratio",
                                "tuning_equal_tempered_deviation",
                                "chords_histogram",
                                "chords_key",
                                "tuning_frequency",
                                "beats_count",
                                "bpm",
                                "danceability",
                                "onset_rate"
                                #"beats_position" - produce arrays with different length
                          ]

        if feature in simple_features:
            result_features = numpy.append(result_features, value)

        elif feature in aggregated_features:
            for type in value:
                if valid_types == "all" or "all" in valid_types or type in valid_types:
                    result_features = numpy.append(result_features, value[type])

                    #if isinstance(value[type], list):
                    #    print "[DEBUG] " + feature + " ==> " + str(len(value[type]))

        return result_features

    def readFeaturesToMem(self):
        directory = self._resultFeaturesDirectory

        if not os.path.exists(directory):
            raise Exception("[ERROR] directory: '" + directory +"' doesn't exist")

        label_directories = os.listdir(directory)

        labels = dict()

        config = Config.get_instance()

        for label_dir in label_directories:

            print "\033[1;36m[CLASSIFIER] label: " + label_dir + "\033[0;0m"

            #labels[label_dir] = []
            label_features = numpy.array([])

            feature_files = os.listdir(directory + "/" + label_dir)
            for feature_file in feature_files:
                #print "[DEBUG] feature_file: " + feature_file

                if not feature_file.endswith(".json"):
                    continue

                full_feature_file = directory + "/" + label_dir + "/"
                data = ''
                with open(full_feature_file + feature_file, 'r') as datafile:
                    data = datafile.read().replace('\n', '')

                data = json.loads(data)

                file_features = numpy.array([])

                for section in data:
                    #print "[DEBUG] Section => " + section

                    for feature in data[section]:
                        #print "[DEBUG] feature: " + feature

                        valid_features = ''
                        if section == "lowlevel":
                            valid_features = config.emotionClassifierLowlevelAudioFeatures
                        elif section == "tonal":
                            valid_features = config.emotionClassifierTonalAudioFeatures
                        elif section == "rhythm":
                            valid_features = config.emotionClassifierRhythmAudioFeatures
                        elif section == "metadata":
                            break #do not process metadata section
                        else:
                            print "[WARNING] unknown section: '"+ section+"' found, skip"
                            break

                        file_features = self.__addFeature__(valid_features,
                                                              config.emotionClassifierAudioFeatureTypes,
                                                              feature,
                                                              data[section][feature],
                                                              file_features)

                if len(label_features) == 0:                              # append feature vector
                    label_features = file_features
                else:
                    label_features = numpy.vstack((label_features, file_features))

            labels[label_dir] = label_features

            print "[CLASSIFIER] n samples: " + str(labels[label_dir].shape[0])
            print "[CLASSIFIER] n features: " + str(labels[label_dir].shape[1])
