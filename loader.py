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
        NONE    = 0,
        MINMAX  = 1,
        ZSCORE = 2


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
        self._featuresScalingType = Loader.ScalingType.NONE

        # label encoding
        self._featuresEncoder = None #TODO: don't use
        self._labelsEncoder = None
        self._labelsEncoded = False

        # polynomial features creater
        self._polynomialFeaturesMaker = None
        self._polynomialFeatures = None

        # different feature selectors


    def scalingTypeToString(self):
        if self._featuresScalingType == Loader.ScalingType.NONE:
            return "None"
        if self._featuresScalingType == Loader.ScalingType.ZSCORE:
            return "Z-score"
        if self._featuresScalingType == Loader.ScalingType.MINMAX:
            return "Min max"
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

                self._audioFeaturesByIdentifier[user_id] = [features, fileNames]

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

        res_dir = result_directory if result_directory != None else config.audioFeaturesResultDirectory

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

            print "[LOADER] n features: " + str(labels[label_dir].shape[1])
            print "[LOADER] m samples: " + str(labels[label_dir].shape[0])

        self._audioFeaturesByIdentifier = labels


    def __checkFeatures__(self):

        if not self._audioFeaturesByIdentifier:
            self.readFeaturesToMem()

        if not self._audioFeaturesByIdentifier:
            raise Exception("[ERROR] empty feature vector!!!")

    def __checkFeatureMatrixAndLables__(self):

        self.__checkFeatures__()

        if not self._featuresMatrixAndLabelsComputed:
            print "[WARNING] making feature matrix and labels vector from loaded features"

            self._featuresMatrix = []
            self._featuresLabels = []

            self.makeFeatureMatrixAndLabels()


    def makeFeatureMatrixAndLabels(self, needRandomShuffle = False):

        print "[LOADER] making feature matrix and feature labels vector"

        self.__checkFeatures__()

        for identifier in self._audioFeaturesByIdentifier:

            m_samples = self._audioFeaturesByIdentifier[identifier].shape[0]

            if len(self._featuresMatrix) == 0:                              # append feature vector
                self._featuresMatrix = self._audioFeaturesByIdentifier[identifier]
            else:
                self._featuresMatrix = numpy.vstack((self._featuresMatrix, self._audioFeaturesByIdentifier[identifier]))

            self._featuresLabels = numpy.append(self._featuresLabels, numpy.full(m_samples, identifier, dtype='|S2'))

        self._featuresMatrixAndLabelsComputed = True

        print "[LOADER] feature matrix shape: " + str(self._featuresMatrix.shape[0]) + " x " + str(self._featuresMatrix.shape[1])
        print "[LOADER] feature labels shape: " + str(self._featuresLabels.shape[0])


    def features(self):

        self.__checkFeatures__()

        return self._audioFeaturesByIdentifier


    def Xmatrix(self):

        self.__checkFeatureMatrixAndLables__()

        return self._featuresMatrix


    def Yvector(self):

        self.__checkFeatureMatrixAndLables__()

        return self._featuresLabels


    def runFeatureStandardization(self, print_matrixes = False):

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

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform feature standartization (using sklearn.preprocessing.scale function)"

        if print_matrixes:
            print self._featuresMatrix

        #self._featuresMatrix =
        preprocessing.scale(self._featuresMatrix, copy=False)
        self._featuresScalingType = Loader.ScalingType.ZSCORE

        if print_matrixes:
            print self._featuresMatrix

    def runFeatureMinMaxNormalization(self, print_matrixes = False):

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

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform min max feature scaling (values between 0 and 1)"

        if self._featuresScalingType == Loader.ScalingType.MINMAX:
            print "[WARNING] features are already scaled"
            return

        if self._featuresMinMaxScaler is None:
            self._featuresMinMaxScaler = preprocessing.MinMaxScaler(copy=False)

        if print_matrixes:
            print self._featuresMatrix

        #self._featuresMatrix =
        self._featuresMinMaxScaler.fit_transform(self._featuresMatrix)
        self._featuresScalingType = Loader.ScalingType.MINMAX

        if print_matrixes:
            print self._featuresMatrix


    def runFeatureStandardScaler(self, features = None):

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

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform feature standartization (using sklearn.preprocessing.StandardScaler class)"

        if self._featuresScalingType == Loader.ScalingType.ZSCORE:
            print "[WARNING] features are already scaled"
            return

        if self._featuresStandardScaler is None:
            self._featuresStandardScaler = preprocessing.StandardScaler().fit(self._featuresMatrix)

        print self._featuresMatrix
        self._featuresMatrix = self._featuresStandardScaler.transform(self._featuresMatrix)
        print self._featuresMatrix

        self._featuresScalingType = Loader.ScalingType.ZSCORE


    def unscaleFeatures(self):
        """
        Function return original features as before any type of scaling
        """

        print "[LOADER] unscaling features [used scaling type: " + self.scalingTypeToString() + "]"

        if self._featuresScalingType == Loader.ScalingType.NONE:
            return

        if self._featuresScalingType == Loader.ScalingType.ZSCORE:

            if self._featuresStandardScaler is None:
                print "[WARNING] feature StandardScaler is not set, do nothing"
                return

            self._featuresMatrix = self._featuresStandardScaler.inverse_transform(self._featuresMatrix)
            print self._featuresMatrix
            return

        if self._featuresScalingType == Loader.ScalingType.MINMAX:

            if self._featuresMinMaxScaler is None:
                print "[WARNING] feature MinMaxScaler is not set, do nothing"
                return

            self._featuresMatrix = self._featuresMinMaxScaler.inverse_transform(self._featuresMatrix)
            return

        raise Exception("Unknown scaling type, can't unscale features!")


    def generatePolynomialFeatures (self, degree = 2):

        """
        Transforming features from (X1, X2) to (1, X1, X2, X1^2, X2^2, X1*X2)

        Param: degree - degree of generated polynomial features

        has no inverse transformation
        """
        #TODO
        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] creating polynomial features [degree: " + str(degree) + "]"

        if not(_self.polynomialFeatures is None):
            return

        if _self._polynomialFeaturesMaker is None:
            _self._polynomialFeaturesMaker = PolynomialFeatures(degree)

        _self._polynomialFeaturesMaker.fit(self._featuresMatrix)

        self._polynomialFeatures = _self._polynomialFeaturesMaker.transform(self._featuresMatrix)


    def labelEncoding(self):
        """
        from sklearn.preprocessing import LabelEncoder
        >> le=LabelEncoder()
        """

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform label encoding"

        if self._labelsEncoded:
            print "[LOADER] labels are already encoded"
            return

        print self._featuresLabels

        if self._labelsEncoder is None:
            self._labelsEncoder = preprocessing.LabelEncoder()

        self._labelsEncoder.fit(self._featuresLabels)

        self._featuresLabels = self._labelsEncoder.transform(self._featuresLabels)

        print self._featuresLabels


    def labelDecoding(self):

        print "[LOADER] perform label decoding"

        if self._labelsEncoded:
            print "[LABEL] labels are already decoded or are original"
            return

        if self._labelsEncoder is None:
            print "[WARNING] label encoder is null, return original features"
            return

        self._featuresLabels = self._labelsEncoder.inverse_transform(self._featuresLabels)

    def performUnvariatefeatureSelection(self, mode="k_best", score_func_name="f_classif", k = 10):
        """
        mode = ["percentile", "k_best", "fpr", "fdr", "fwe"]
        score_func = ["chi2", "f_classif", "mutual_info_classif"]
        k - amount of features to leave
        """

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform unvariate feature selction [score_func: " + score_func_name + ", k: " + str(k) + "]"

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

        #selector = SelectKBest(score_func, k)
        selector = GenericUnivariateSelect(score_func, "k_best", k)

        fit = selector.fit(self._featuresMatrix, self._featuresLabels) # непосредственно анализирует признаки и определяет их стоимости
        # summarize scores
        numpy.set_printoptions(precision=3)
        print(fit.scores_) # выводит стоимость каждого признака для принятия результата
        features = fit.transform(self._featuresMatrix) # применяет посичтанные стоимости к матрице признаков
        # summarize selected features
        print(features[0:5,:])


    def performRecursiveFeatureSelection(self, k = 10, step=1):
        """
        k - amount of features to leave
        step - int or float, optional (default=1)
               If greater than or equal to 1, then step corresponds to the
               (integer) number of features to remove at each iteration.
               If within (0.0, 1.0), then step corresponds to the percentage
               (rounded down) of features to remove at each iteration.
        """

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform recursive feature selction [step: " + str(step) + ", k: " + str(k) + "]"

        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=k, step=step, verbose=1)
        fit = rfe.fit(self._featuresMatrix, self._featuresLabels)
        numpy.set_printoptions(precision=3)

        # ranking_ : array of shape [n_features]
        # The feature ranking, such that ranking_[i] corresponds to the ranking
        # position of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1.
        print rfe.ranking_
        print fit.transform(self._featuresMatrix)


    def plotReducedFeatures(self):

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] ploting features (dimension reduction - use PCA with n_components = 2)"

        pca = PCA(2)
        fit = pca.fit(self._featuresMatrix)
        features = fit.transform(self._featuresMatrix)

        self.labelEncoding()

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
        for i in numpy.unique(self._featuresLabels):
            print "class: " + str(i)
            idx = numpy.where(self._featuresLabels == i)
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
        for i in numpy.unique(self._featuresLabels):
            print "class: " + str(i)
            idx = numpy.where(self._featuresLabels == i)
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


    def performPCAFeatureSelection(self, n_components = 3):

        self.__checkFeatureMatrixAndLables__()

        print "[LOADER] perform recursive feature selction [n_components: " + str(n_components) + "]"

        pca = PCA(n_components)
        fit = pca.fit(self._featuresMatrix)
        self._featuresMatrix = fit.transform(self._featuresMatrix)

        # summarize components
        print("[LABEL] PCA Explained Variance: %s") % fit.explained_variance_ratio_
        print(self._featuresMatrix[0:5,:])


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
