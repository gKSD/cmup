#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re
from jsonschema import validate

from lib.audio_feature_extraction.audioBasicIO import *
from audio_feature_extracter import AudioFeatureExtracter
from storage import Storage
from config import Config
from lib.fingerprint.dejavu.dejavu import decoder as fingerprintDecoder
from lib.fingerprint.dejavu.dejavu import fingerprint as fingerprint

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Loader:
    def __init__(self,
                 storage,
                 result_directory = None,
                 valid_lowlevel_features = None,
                 valid_tonal_features = None,
                 valid_rhythm_features = None,
                 valid_audio_features_type = None):

        self.__storage = storage

        self.__data = []

        self._audioFeatureExtracter = AudioFeatureExtracter()

        self._features = {}

        self._resultFeaturesDirectory = result_directory if result_directory else Config.get_instance().classifierAudioFeaturesResultDirectory

        self._validLowlevelFeatures = valid_lowlevel_features if valid_lowlevel_features else "all"

        self._validTonalFeatures = valid_tonal_features if valid_tonal_features else "all"

        self._validRhythmFeatures = valid_rhythm_features if valid_rhythm_features else "all"

        self._validAudioFeatureTypes = valid_audio_features_type if valid_audio_features_type else "all"


    def load(self, data, ids = None):
        print "[DEBUG] input data => " + data

        try:
            self.__data = json.loads(data)
        except ValueError, e:
            return False

        if ids:
            ids_array = re.sub(r'\s', '', ids).split(',')
            keys = self.__data.keys()
            for key in keys:
                if key in ids_array:
                    del self.__data[key]

        return True

    def reload(self, input_data):
        self.__data = []
        self.load(input_data)

    def printData(self):
        print json.dumps(self.__data, separators=(',',':'))

    def data(self):
        return self.__data

    def extractFeatures(self):
        if not self.__data:
            raise Exception("Errror: input data is empty or error occured while parsing")

        user_ids = self.__data.keys()

        print "[LOADER] feature exctraction for IDs: "+ ', '.join(user_ids)

        config = Config.get_instance()

        for user_id in user_ids:
            print "[LOADER] feature exctraction for ID: " + user_id

            features = numpy.array([])
            fileNames = []

            if "dirs" in self.__data[user_id]:

                dir_names = self.__data[user_id]["dirs"]

                print "[LOADER] feature exctraction for dirs: '" + ', '.join(dir_names) + "'"

                try:
                    [features, fileNames] = self._audioFeatureExtracter.processDirs(features,
                                                                                    fileNames,
                                                                                    dir_names,
                                                                                    config.midTermSize,
                                                                                    config.midTermStep,
                                                                                    config.shortTermSize,
                                                                                    config.shortTermStep,
                                                                                    config.computeBeat)
                    #self._audioFeatureExtracter.normalizeFeatures(features)
                except Exception as e:
                    print "[LOADER] error while processing dirs: " + ', '.join(dir_names) + ", skip [Error:" + str(e) +"]"

            if "files" in self.__data[user_id]:
                print "[LOADER] feature extraction for files: " + ', '.join(self.__data[user_id]["files"])

                files = self.__data[user_id]["files"]

                for i, file_name in enumerate(files):
                    print "[LOADER] Analyzing file {0:d} of {1:d}: {2:s}".format(i + 1, len(files), file_name.encode('utf-8'))

                    try:
                        #[Fs, audio] = readAudioFile(file_name)
                        fileFeatures = self._audioFeatureExtracter.processFile(file_name,
                                                                          config.midTermSize,
                                                                          config.midTermStep,
                                                                          config.shortTermSize,
                                                                          config.shortTermStep,
                                                                          config.computeBeat)
                    except Exception as e:
                        print "[LOADER] error while processing file: " + file_name + ", skip [Error:" + str(e) +"]"
                        continue
                    # TODO: save them to storage and them make the references??

                    if len(features) == 0:                              # append feature vector
                        features = fileFeatures
                    else:
                        features = numpy.vstack((features, fileFeatures))
                    fileNames.append(file_name)

                self._features[user_id] = [features, fileNames]

            # TODO: store file to _self._resultFeaturesDirectory
            res_file_name = str(user_id) + '_res_file_name.txt'
            f = open(res_file_name, 'w')
            for i in range(0, len(fileNames)):
                j=json.dumps({fileNames[i]:features[i]},cls=NumpyAwareJSONEncoder)
                f.write(j)
                f.write("\n")
            f.close()


    def extractFeaturesEssentia(self, result_directory = None):

        if not self.__data:
            raise Exception("Errror: input data is empty or error occured while data parsing")

        user_ids = self.__data.keys()

        print "[LOADER] essentia feature exctraction for IDs: "+ ', '.join(user_ids)

        config = Config.get_instance()

        res_dir = result_directory if result_directory else config.audioFeaturesResultDirectory

        for user_id in user_ids:
            print "[LOADER] essentia feature exctraction for ID: " + user_id

            user_res_dir = res_dir + "/" + user_id

            features = numpy.array([])
            fileNames = []

            if "dirs" in self.__data[user_id]:

                dir_names = self.__data[user_id]["dirs"]

                print "[LOADER] essentia feature exctraction for dirs: '" + ', '.join(dir_names) + "'"

                for dir_name in dir_names:
                    try:
                        self._audioFeatureExtracter.processDirEssentia(dir_name, user_res_dir)
                    except Exception as e:
                        print "[LOADER] essentia error while processing dir: " + ', '.join(dir_name) + ", skip [Error:" + str(e) +"]"
                        continue

            if "files" in self.__data[user_id]:
                print "[LOADER] essentia feature extraction for files: " + ', '.join(self.__data[user_id]["files"])

                files = self.__data[user_id]["files"]

                for i, file_name in enumerate(files):
                    print "[LOADER] essentia analyzing file {0:d} of {1:d}: {2:s}".format(i + 1, len(files), file_name.encode('utf-8'))

                    try:
                        self._audioFeatureExtracter.processFileEssentia(file_name, user_res_dir)
                    except Exception as e:
                        print "[LOADER] essentia error while processing file: " + file_name + ", skip [Error:" + str(e) +"]"
                        continue


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


    def readFeaturesToMem(self,
                          features_directory = None,
                          valid_lowlevel_features = None,
                          valid_tonal_features = None,
                          valid_rhythm_features = None,
                          valid_audio_features_type = None):

        directory = features_directory if features_directory else self._resultFeaturesDirectory

        valid_lowlevel_features = valid_lowlevel_features if valid_lowlevel_features else self._validLowlevelFeatures
        valid_tonal_features = valid_tonal_features if valid_tonal_features else self._validTonalFeatures
        valid_rhythm_features = valid_rhythm_features if valid_rhythm_features else self._validRhythmFeatures
        valid_audio_features_type = valid_audio_features_type if valid_audio_features_type else self._validAudioFeatureTypes

        if not os.path.exists(directory):
            raise Exception("[ERROR] directory: '" + directory +"' doesn't exist")

        label_directories = os.listdir(directory)

        labels = dict()

        for label_dir in label_directories:

            print "\033[1;36m[LOADER] label: " + label_dir + "\033[0;0m"

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
                            valid_features = valid_lowlevel_features
                        elif section == "tonal":
                            valid_features = valid_tonal_features
                        elif section == "rhythm":
                            valid_features = valid_rhythm_features
                        elif section == "metadata":
                            break #do not process metadata section
                        else:
                            print "[WARNING] unknown section: '" + section + "' found, skip"
                            break

                        file_features = self.__addFeature__(valid_features,
                                                            valid_audio_features_type,
                                                            feature,
                                                            data[section][feature],
                                                            file_features)

                if len(label_features) == 0:                              # append feature vector
                    label_features = file_features
                else:
                    label_features = numpy.vstack((label_features, file_features))

            labels[label_dir] = label_features

            print "[LOADER] n features: " + str(labels[label_dir].shape[1])
            print "[LOADER] m samples: " + str(labels[label_dir].shape[0])

        self._features = labels


    def features(self):
        return self._features


    def normalizeFeatures(self):

        if not self._features:
            self.readFeaturesToMem()

    
    def fingerprint(self, filename, limit=None, song_name=None):
        print "[LOADER] fingerprinting filename: " + filename
        try:
            filename, limit = filename
        except ValueError:
            pass

        songname, extension = os.path.splitext(os.path.basename(filename))
        song_name = song_name or songname
        channels, Fs, file_hash = fingerprintDecoder.read(filename, limit)
        result = set()
        channel_amount = len(channels)
        print "channel_amount => "
        print channel_amount

        for channeln, channel in enumerate(channels):
            # TODO: Remove prints or change them into optional logging.
            print("Fingerprinting channel %d/%d for %s" % (channeln + 1,
                                                           channel_amount,
                                                           filename))
            #hashes = fingerprint.fingerprint(channel, Fs=Fs)
            #print("Finished channel %d/%d for %s" % (channeln + 1, channel_amount,
            #                                         filename))
            #result |= set(hashes)

        print "result => "
        print result
        print "file_hash => "
        print file_hash

        return song_name, result, file_hash


    def saveData(self):
        pass

# TODO: learn how to detect similar music
# TODO: detect similar - DHT
