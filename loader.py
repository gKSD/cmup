#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re
from jsonschema import validate
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from lib.audio_feature_extraction.audioBasicIO import *
from audio_feature_extracter import AudioFeatureExtracter
from storage import Storage
from config import Config
from lib.fingerprint.dejavu.dejavu import decoder as fingerprintDecoder
from lib.fingerprint.dejavu.dejavu import fingerprint as fingerprint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from enum import Enum

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Loader:
    class ScalingType(Enum):
        ST_NONE    = 0,
        ST_MINMAX  = 1,
        ST_ZSCORE  = 2

    class FeatureSelectionType(Enum):
        FST_NONE = 0,
        FST_GENERIC_UNIVARIATIVE = 1,
        FST_RECURSIVE = 2,
        FST_PCA = 3,
        FST_L1BASED = 4,
        FST_TREE_BASED = 5

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

        self._audioFeaturesByIdentifier = {}

        self._featuresMatrix = [] # matrix X n x m (n features, m samples)
        self._featuresLabels = [] # vector Y m x 1 (m samples)
        self._featuresMatrixAndLabelsComputed = False

        self._resultFeaturesDirectory = result_directory if result_directory != None else Config.get_instance().classifierAudioFeaturesResultDirectory

        self._validLowlevelFeatures = valid_lowlevel_features if valid_lowlevel_features != None else "all"

        self._validTonalFeatures = valid_tonal_features if valid_tonal_features != None else "all"

        self._validRhythmFeatures = valid_rhythm_features if valid_rhythm_features != None else "all"

        self._validAudioFeatureTypes = valid_audio_features_type if valid_audio_features_type != None else "all"


        # different feature scalers
        self._featuresStandardScaler = None # Z-score sclaing
        self._featuresMinMaxScaler = None # min max scaling
        self._featuresScalingType = Loader.ScalingType.ST_NONE


        # label encoding
        self._featuresEncoder = None #TODO: don't use
        self._labelsEncoder = None
        self._labelsEncoded = False


        # polynomial features createSC_r
        self._polynomialFeaturesMaker = None
        self._polynomialFeatures = None


        # different feature selectors
        self._featuresSelectorType = Loader.FeatureSelectionType.FST_NONE
        self._genericUnivariateSelector = None
        self._recursiveSelector = None
        self._PCASelector = None
        self._L1BasedSelector = None   # using SelectFromModel
        self._treeBasedSelector = None # using SelectFromModel


    def labelsEncoder(self):
        return self._labelsEncoder


    def scalingTypeToString(self, stype):
        if stype == Loader.ScalingType.ST_NONE:
            return "None"
        if stype == Loader.ScalingType.ST_ZSCORE:
            return "Z-score"
        if stype == Loader.ScalingType.ST_MINMAX:
            return "Min max"
        return "Unknown"


    def featureSelectorTypeToString(self, fstype):
        if fstype == Loader.FeatureSelectionType.FST_NONE:
            return "None"
        if fstype == Loader.FeatureSelectionType.FST_GENERIC_UNIVARIATIVE:
            return "Generic univariative"
        if fstype == Loader.FeatureSelectionType.FST_RECURSIVE:
            return "Recursive"
        if fstype == Loader.FeatureSelectionType.FST_PCA:
            return "PCA"
        if fstype == Loader.FeatureSelectionType.FST_L1BASED:
            return "L1 based"
        return "Unknown"


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

    def extractFeatures(self, data):
        if not data:
            raise Exception("Errror: input data is empty or error occured while parsing")

        user_ids = data.keys()

        print "[LOADER] feature exctraction for IDs: "+ ', '.join(user_ids)

        config = Config.get_instance()

        for user_id in user_ids:
            print "[LOADER] feature exctraction for ID: " + user_id

            features = numpy.array([])
            fileNames = []

            if "dirs" in data[user_id]:

                dir_names = data[user_id]["dirs"]

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

            if "files" in data[user_id]:
                print "[LOADER] feature extraction for files: " + ', '.join(data[user_id]["files"])

                files = data[user_id]["files"]

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

                self._audioFeaturesByIdentifier[user_id] = [features, fileNames]

            # TODO: store file to _self._resultFeaturesDirectory
            res_file_name = str(user_id) + '_res_file_name.txt'
            f = open(res_file_name, 'w')
            for i in range(0, len(fileNames)):
                j=json.dumps({fileNames[i]:features[i]},cls=NumpyAwareJSONEncoder)
                f.write(j)
                f.write("\n")
            f.close()


    def extractFeaturesEssentia(self, result_directory, data):

        if data is None:
            raise Exception("Errror: input data is empty or error occured while data parsing")

        if not result_directory:
            raise Exception("Error: result directory must be set")

        identifiers = data.keys()

        print "[LOADER] essentia feature exctraction for IDs: "+ ', '.join(identifiers)

        config = Config.get_instance()

        res_dir = result_directory

        for identifier in identifiers:
            print "[LOADER] essentia feature exctraction for ID: " + identifier

            user_res_dir = res_dir + "/" + identifier

            if not os.path.exists(user_res_dir):
                os.makedirs(user_res_dir)


            features = numpy.array([])
            fileNames = []

            if "dirs" in data[identifier]:

                dir_names = data[identifier]["dirs"]

                print "[LOADER] essentia feature exctraction for dirs: '" + ', '.join(dir_names) + "'"

                for dir_name in dir_names:
                    try:
                        self._audioFeatureExtracter.processDirEssentia(dir_name, user_res_dir)
                    except Exception as e:
                        print "[LOADER] essentia error while processing dir: " + ', '.join(dir_name) + ", skip [Error:" + str(e) +"]"
                        continue

            if "files" in data[identifier]:
                print "[LOADER] essentia feature extraction for files: " + ', '.join(data[identifier]["files"])

                files = data[identifier]["files"]

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
                          features_directory,
                          valid_lowlevel_features = None,
                          valid_tonal_features = None,
                          valid_rhythm_features = None,
                          valid_audio_features_type = None):

        directory = features_directory

        valid_lowlevel_features = valid_lowlevel_features if valid_lowlevel_features != None else self._validLowlevelFeatures
        valid_tonal_features = valid_tonal_features if valid_tonal_features != None else self._validTonalFeatures
        valid_rhythm_features = valid_rhythm_features if valid_rhythm_features != None else self._validRhythmFeatures
        valid_audio_features_type = valid_audio_features_type if valid_audio_features_type != None else self._validAudioFeatureTypes

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

            if labels[label_dir].size == 0:
                print "[LOADER] no data found"
            else:
                print "[LOADER] n features: " + str(labels[label_dir].shape[1])
                print "[LOADER] m samples: " + str(labels[label_dir].shape[0])

        #self._audioFeaturesByIdentifier = labels
        return labels


    def __checkFeaturesByIdentifier__(self, featuresByIdentifier):

        if featuresByIdentifier is None:
            raise Exception("Error, features is None")


    def __checkFeatureMatrix__(self, features):

        if features is None:
            raise Exception("Error feature matrix is None")


    def __checkFeatureMatrixAndLables__(self, features, labels):

        self.__checkFeatureMatrix__(features)

        if features is None:
            raise Exception("Error feature labels is None")


    def makeFeatureMatrixAndLabels(self, featuresByIdentifier, needRandomShuffle = False):

        print "[LOADER] making feature matrix and feature labels vector"

        self.__checkFeaturesByIdentifier__(featuresByIdentifier)

        x = numpy.array([])
        y = []

        for identifier in featuresByIdentifier:

            m_samples = featuresByIdentifier[identifier].shape[0]

            if m_samples == 0:
                print "[WARNING] no files found in " + identifier
                continue

            if len(x) == 0:                              # append feature vector
                x = featuresByIdentifier[identifier]
            else:
                x = numpy.vstack((x, featuresByIdentifier[identifier]))

            y = numpy.append(y, numpy.full(m_samples, identifier, dtype='|S2'))

        self._featuresMatrixAndLabelsComputed = True

        if x.size == 0:
            print "[LOADER] feature matrix and labels are empty"
        else:
            print "[LOADER] feature matrix shape: " + str(x.shape[0]) + " x " + str(x.shape[1])
            print "[LOADER] feature labels shape: " + str(y.shape[0])

        return [x, y]


    def runFeatureStandardization(self, data, print_matrixes = False):

        """
        Standardization of datasets is a common requirement for many machine learning estimators
        implemented in scikit-learn; they might behave badly if the individual features do not
        more or less look like standard normally distributed data: Gaussian with zero mean and
        unit variance.

        In practice we often ignore the shape of the distribution and just transform the data to center
        it by removing the mean value of each feature, then scale it by dividing non-constant features
        by their standard deviation.

        For instance, many elements used in the objective function of a learning algorithm
        (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models)
        assume that all features are centered around zero and have variance in the same order.
        If a feature has a variance that is orders of magnitude larger than others, it might dominate
        the objective function and make the estimator unable to learn from other features correctly as expected.

        Scaled data has zero mean and unit variance:
        >>> X_scaled = preprocessing.scale(X)
        >>> X_scaled.mean(axis=0)
        array([ 0.,  0.,  0.])
        >>> X_scaled.std(axis=0)
        array([ 1.,  1.,  1.])

        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale

        It is Z-score normalization

        TODO: if your data contains many outliers, scaling using the mean and variance
        of the data is likely to not work very well. In these cases, you can use
        robust_scale and RobustScaler as drop-in replacements instead.
        They use more robust estimates for the center and range of your data.

        NB! to perform the same standartization both for traing and for testing datasets use
            Loader().runFeatureStandardScaler() function
        """

        self.__checkFeatureMatrix__(data.X())

        print "[LOADER] perform feature standartization (using sklearn.preprocessing.scale function)"

        if print_matrixes:
            print data.X()

        #self._featuresMatrix =
        preprocessing.scale(data.X(), copy=False)
        data.setScalingType(Loader.ScalingType.ST_ZSCORE)

        if print_matrixes:
            print data.X()

        #return features

    def runFeatureMinMaxNormalization(self, data, print_matrixes = False):

        """
        An alternative standardization is scaling features to lie between a given minimum
        and maximum value, often between zero and one, or so that the maximum absolute value
        of each feature is scaled to unit size.
        This can be achieved using MinMaxScaler or MaxAbsScaler, respectively.

        The motivation to use this scaling include robustness to very small standard deviations
        of features and preserving zero entries in sparse data.

        Formula:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        """

        self.__checkFeatureMatrix__(data.X())

        print "[LOADER] perform min max feature scaling (values between 0 and 1)"

        if data.scalingType() == Loader.ScalingType.ST_MINMAX:
            print "[WARNING] features are already scaled"
            return

        if self._featuresMinMaxScaler is None:
            self._featuresMinMaxScaler = preprocessing.MinMaxScaler(copy=False)

        if print_matrixes:
            print data.X()

        #self._featuresMatrix =
        self._featuresMinMaxScaler.fit_transform(data.X())
        data.setScalingType(Loader.ScalingType.ST_MINMAX)

        if print_matrixes:
            print data.X()


    def runFeatureStandardScaler(self, data):

        """
        The preprocessing module further provides a utility class StandardScaler
        that implements the Transformer API to compute the mean and standard deviation
        on a training set so as to be able to later reapply the same transformation
        on the testing set.
        This class is hence suitable for use in the early steps of a sklearn.pipeline.Pipeline.

        The scaler instance can then be used on new data to transform it the same way it did on the training set:
        >>> scaler.transform([[-1.,  1., 0.]])
        array([[-2.44...,  1.22..., -0.26...]])

        NB! this scaler perform Z-score sclaing as preprocessing.scale,
            but this allows to use the same scaler both for training and for testing datasets
        """

        self.__checkFeatureMatrix__(data.X())

        print "[LOADER] perform feature standartization (using sklearn.preprocessing.StandardScaler class)"

        if data.scalingType() == Loader.ScalingType.ST_ZSCORE:
            print "[WARNING] features are already scaled"
            return

        if self._featuresStandardScaler is None:
            self._featuresStandardScaler = preprocessing.StandardScaler()
            self._featuresStandardScaler.fit(data.X())

        print data.X()
        data.setX( self._featuresStandardScaler.transform(data.X()) )
        print data.X()

        data.setScalingType(Loader.ScalingType.ST_ZSCORE)


    def unscaleFeatures(self, data):
        """
        Function return original features as before any type of scaling
        """

        print "[LOADER] unscaling features [used scaling type: " + self.scalingTypeToString(data.scalingType()) + "]"

        if data.scalingType() == Loader.ScalingType.ST_NONE:
            return

        if data.scalingType() == Loader.ScalingType.ST_ZSCORE:

            if self._featuresStandardScaler is None:
                print "[WARNING] feature StandardScaler is not set, do nothing"
                return

            data.setX( self._featuresStandardScaler.inverse_transform(data.X()) )
            print data.X()
            return

        if data.scalingType() == Loader.ScalingType.ST_MINMAX:

            if self._featuresMinMaxScaler is None:
                print "[WARNING] feature MinMaxScaler is not set, do nothing"
                return

            data.setX( self._featuresMinMaxScaler.inverse_transform(data.X()) )
            return

        raise Exception("Unknown scaling type, can't unscale features!")


    def generatePolynomialFeatures (self, data, degree = 2):

        """
        Transforming features from (X1, X2) to (1, X1, X2, X1^2, X2^2, X1*X2)

        Param: degree - degree of generated polynomial features

        has no inverse transformation
        """
        #TODO
        self.__checkFeatureMatrix__(data.X())

        print "[LOADER] creating polynomial features [degree: " + str(degree) + "]"

        if _self._polynomialFeaturesMaker is None:
            _self._polynomialFeaturesMaker = PolynomialFeatures(degree)
            _self._polynomialFeaturesMaker.fit(data.X())

        data.setXpoly( _self._polynomialFeaturesMaker.transform(data.X()) )


    def labelEncoding(self, data):
        """
        from sklearn.preprocessing import LabelEncoder
        >> le=LabelEncoder()
        """

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform label encoding"

        if data.labelsEncoded():
            print "[LOADER] labels are already encoded"
            return

        #print data.Y()

        if self._labelsEncoder is None:
            self._labelsEncoder = preprocessing.LabelEncoder()
            self._labelsEncoder.fit(data.Y())

        data.setY( self._labelsEncoder.transform(data.Y()) )

        data.setLabelsEncoded(True)

        #print data.Y()


    def labelDecoding(self):

        print "[LOADER] perform label decoding"

        if not data.labelsEncoded():
            print "[LABEL] labels are already decoded or are original"
            return

        if self._labelsEncoder is None:
            print "[WARNING] label encoder is null, return original features"
            return

        data.setY( self._labelsEncoder.inverse_transform(data.Y()) )

        data.setLabelsEncoded(False)


    def performUnvariatefeatureSelection(self, data, mode="k_best", score_func_name="f_classif", k = 10):
        """
        mode = ["percentile", "k_best", "fpr", "fdr", "fwe"]
        score_func = ["chi2", "f_classif", "mutual_info_classif"]
        k - amount of features to leave
        """

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform unvariate feature selction [score_func: " + score_func_name + ", k: " + str(k) + "]"

        if data.selectionType() == Loader.FeatureSelectionType.FST_GENERIC_UNIVARIATIVE:
            print "[WARNING] unvariate feature selction is already done, perform univariative inverse transorm"
            data.setX( self._genericUnivariateSelector.inverse_transform(data.X()) )
        elif data.selectionType() != Loader.FeatureSelectionType.FST_NONE:
            raise Exception("Can't perform Univariative selection, feature selection '" + self.featureSelectorTypeToString(data.selectionType()) + "' is done")

        if not(mode in ["percentile", "k_best", "fpr", "fdr", "fwe"]):
            raise Exception("Unexpected mode for unvariate feature selection!")

        score_func=None
        if score_func_name == "chi2":
            print "[WARNING] features should be non-negative"
            score_func = chi2
        elif score_func_name == "f_classif":
            score_func = f_classif
        elif score_func_name == "mutual_info_classif":
            score_func = mutual_info_classif
        else:
            raise Exception("Unexpected score function for unvariate feature selection!")

        if self._genericUnivariateSelector is None:
            #selector = SelectKBest(score_func, k)
            self._genericUnivariateSelector = GenericUnivariateSelect(score_func, "k_best", k)
        #else:
            #self._genericUnivariateSelector.set_params(score_func=score_func, mode="k_best", param=k)
            # непосредственно анализирует признаки и определяет их стоимости
            self._genericUnivariateSelector.fit(data.X(), data.Y())

        # summarize scores
        numpy.set_printoptions(precision=3)
        # выводит стоимость каждого признака для принятия результата
        print(self._genericUnivariateSelector.scores_)

        # применяет посичтанные стоимости к матрице признаков
        # summarize selected features
        data.setX( self._genericUnivariateSelector.transform(data.X()) )
        print data.X()[0:5,:]

        # setting class params
        data.setSelectionType(Loader.FeatureSelectionType.FST_GENERIC_UNIVARIATIVE)


    def performRecursiveFeatureSelection(self, data, k = 10, step=1):
        """
        k - amount of features to leave
        step - int or float, optional (default=1)
               If greater than or equal to 1, then step corresponds to the
               (integer) number of features to remove at each iteration.
               If within (0.0, 1.0), then step corresponds to the percentage
               (rounded down) of features to remove at each iteration.
        """

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform recursive feature selction [step: " + str(step) + ", k: " + str(k) + "]"

        if data.selectionType() == Loader.FeatureSelectionType.FST_RECURSIVE:
            print "[WARNING] recursive feature selection is already done, perform inverse rescursive transform"
            data.setX( self._recursiveSelector.inverse_transform(data.X()) )
        elif data.selectionType() != Loader.FeatureSelectionType.FST_NONE:
            raise Exception("Can't perform recursive selection, feature selection '" + self.featureSelectorTypeToString(data.selectionType()) + "' is done")

        if self._recursiveSelector is None:
            svc = SVC(kernel="linear", C=1)
            self._recursiveSelector = RFE(estimator=svc, n_features_to_select=k, step=step, verbose=1)
        #else:
        #    self._recursiveSelector.set_params(n_features_to_select=k, step=step)
            self._recursiveSelector.fit(data.X(), data.Y())

        numpy.set_printoptions(precision=3)

        # ranking_ : array of shape [n_features]
        # The feature ranking, such that ranking_[i] corresponds to the ranking
        # position of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1.
        print self._recursiveSelector.ranking_
        data.setX( self._recursiveSelector.transform(data.X()) )
        print data.X()

        # setting class params
        data.setSelectionType(Loader.FeatureSelectionType.FST_RECURSIVE)


    def plotReducedFeatures(self, data):

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] ploting features (dimension reduction - use PCA with n_components = 2)"

        pca = PCA(2)
        fit = pca.fit(data.X())
        features = fit.transform(data.X())

        self.labelEncoding(data)

        figure = plt.figure(figsize=(27, 9))

        x_min, x_max = features[:, 0].min() - .5, features[:, 0].max() + .5
        y_min, y_max = features[:, 1].min() - .5, features[:, 1].max() + .5

        h = .02 # step size in the mesh (сетка)

        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

        #cm = plt.cm.RdBu
        #cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        ax = plt.subplot(2, 1, 1)
        ax.axis('tight')
        ax.set_title("Input data")
        # Plot the training points

        #ax.scatter(features[:, 0], features[:, 1], c=y_train, cmap=cm_bright)
        for i in numpy.unique(data.Y()):
            print "class: " + str(i)
            idx = numpy.where(data.Y() == i)
            color=None
            if i == 0:
                color='y'
            elif i == 1:
                color='b'
            elif i == 2:
                color='r'
            elif i == 3:
                color='g'
            ax.scatter(features[idx, 0], features[idx, 1], c=color, cmap=plt.cm.Paired, label=self._labelsEncoder.classes_[i])
        # TODO: вынести все преобразователи в переменные класса и например использовать для label в легенде
        # and testing points
        #plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend()

        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("dataset centroids")
        for i in numpy.unique(data.Y()):
            print "class: " + str(i)
            idx = numpy.where(data.Y() == i)
            print "idx"
            print idx[0]
            #print features[idx, :]
            centroid = numpy.mean(features[idx[0], :], 0)
            print "Centroid"
            print centroid
            color=None
            if i == 0:
                color='y'
            elif i == 1:
                color='b'
            elif i == 2:
                color='r'
            elif i == 3:
                color='g'
            ax2.scatter(centroid[0], centroid[1], c=color, cmap=plt.cm.Paired, label=self._labelsEncoder.classes_[i])
        # TODO: вынести все преобразователи в переменные класса и например использовать для label в легенде
        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())
        ax2.set_xticks(())
        ax2.set_yticks(())
        ax2.legend()

        plt.show()


    def performPCAFeatureSelection(self, data, n_components = 3):

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform PCA feature selction [n_components: " + str(n_components) + "]"

        if data.selectionType() == Loader.FeatureSelectionType.FST_PCA:
            print "[WARNING] PCA feature selction is already done, perform PCA inverse transorm"
            data.setX( self._PCASelector.inverse_transform(data.X()) )
        elif data.selectionType() != Loader.FeatureSelectionType.FST_NONE:
            raise Exception("Can't perform PCA selection, feature selection '" + self.featureSelectorTypeToString(data.selectionType()) + "' is done")

        if self._PCASelector is None:
            self._PCASelector = PCA(n_components)
        #else:
        #    self._PCASelector.set_params(n_components = n_components)
            self._PCASelector.fit(data.X())

        data.setX( self._PCASelector.transform(data.X()) )

        # summarize components
        print("[LABEL] PCA Explained Variance: %s") % self._PCASelector.explained_variance_ratio_
        print(data.X()[0:5,:])

        # setting class params
        data.setSelectionType( Loader.FeatureSelectionType.FST_PCA )


    def performL1BasedFeatureSelection(self, data, C=0.005):
        """
        C regulates feature count
        L1 - l1 norm
        """

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform L1 based feature selction"

        if data.selectionType() == Loader.FeatureSelectionType.FST_L1BASED:
            print "[WARNING] L1 Based feature selction is already done, perform L1 Based inverse transorm"
            return
            #self._featuresMatrix = self._PCASelector.inverse_transform(self._featuresMatrix)
        elif data.selectionType() != Loader.FeatureSelectionType.FST_NONE:
            raise Exception("Can't perform L1 Based selection, feature selection '" + self.featureSelectorTypeToString(data.selectionType()) + "' is done")

        lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(data.X(), data.Y())
        self._L1BasedSelector = SelectFromModel(lsvc, prefit=True)

        data.setX( self._L1BasedSelector.transform(data.X()) )

        # summarize components
        print data.X().shape
        print(data.X()[0:5,:])

        # setting class params
        data.setSelectionType( Loader.FeatureSelectionType.FST_L1BASED )


    def performTreeBasedFeatureSelection(self, data):
        """
        do not know how to regulate
        """

        self.__checkFeatureMatrixAndLables__(data.X(), data.Y())

        print "[LOADER] perform Tree based feature selction"

        if data.selectionType() == Loader.FeatureSelectionType.FST_TREE_BASED:
            print "[WARNING] Tree Based feature selction is already done, perform Tree Based inverse transorm"
            return
            #self._featuresMatrix = self._PCASelector.inverse_transform(self._featuresMatrix)
        elif data.selectionType() != Loader.FeatureSelectionType.FST_NONE:
            raise Exception("Can't perform Tree Based selection, feature selection '" + self.featureSelectorTypeToString(data.selectionType()) + "' is done")

        clf = ExtraTreesClassifier()
        clf.fit(data.X(), data.Y())
        print "[LOADER] Tree based feature selection, feature importances: "
        print clf.feature_importances_
        self._treeBasedSelector = SelectFromModel(clf, prefit=True)

        print data.X()

        data.setX( self._treeBasedSelector.transform(data.X()) )

        # summarize components
        print data.X().shape
        print(data.X()[0:5,:])

        # setting class params
        data.setSelectionType(Loader.FeatureSelectionType.FST_TREE_BASED)


    def makeOriginalFeaturesSetAfterFeatureSelection(self, data):
        if data.selectionType() == Loader.FeatureSelectionType.FST_NONE:
            return

        if data.selectionType() == Loader.FeatureSelectionType.FST_GENERIC_UNIVARIATIVE:
            if not(self._genericUnivariateSelector is None):
                data.setX( self._genericUnivariateSelector.inverse_transform(data.X()) )
                return

        if data.selectionType() == Loader.FeatureSelectionType.FST_RECURSIVE:
            if not(self._recursiveSelector is None):
                data.setX( self._recursiveSelector.inverse_transform(data.X()) )
                return

        if data.selectionType() == Loader.FeatureSelectionType.FST_PCA:
            if not(self._PCASelector is None):
                data.setX( self._PCASelector.inverse_transform(data.X()) )
                return

        if data.selectionType() == Loader.FeatureSelectionType.FST_L1BASED:
            if not(self._L1BasedSelector is None):
                data.setX( self._L1BasedSelector.inverse_transform(data.X()) )
                return

        if data.selectionType() == Loader.FeatureSelectionType.FST_TREE_BASED:
            if not(self._treeBasedSelector is None):
                data.setX( self._treeBasedSelector.inverse_transform(data.X()) )
                return

        # fall here if invalid type found or appropriate selector is None
        raise Exception("Unknow feature selection state")


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
