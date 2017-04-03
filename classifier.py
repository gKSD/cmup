#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re
import os
import numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from loader import Loader
from config import Config
from storage import Storage
from data import Data

# TODO: classifier comparance
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

class Classifier:
    def __init__(self, storage, directory = None):
        self._storage = storage

        self._resultFeaturesDirectory = directory if directory != None else Config.get_instance().classifierAudioFeaturesResultDirectory

        config = Config.get_instance()

        self._loader = Loader(self._storage,
                              self._resultFeaturesDirectory,
                              config.emotionClassifierLowlevelAudioFeatures,
                              config.emotionClassifierTonalAudioFeatures,
                              config.emotionClassifierRhythmAudioFeatures,
                              config.emotionClassifierAudioFeatureTypes)

        self._classifiersNames = [
                       "Nearest Neighbors",
                       "Linear SVM",
                       "RBF SVM",
                       "Gaussian Process",
                       "Decision Tree",
                       "Random Forest",
                       "Neural Net",
                       "AdaBoost",
                       "Naive Bayes",
                       "QDA"
                      ]

        self._trainData = Data()
        self._testData  = Data()
        self._cvData    = Data()


    def _loadDataProcess_impl_(self, input_data):

        print "[CLASSIFIER] loading input data"
        config = Config.get_instance()

        loaded_data = None
        try:
            loaded_data = json.loads(input_data)
        except ValueError, e:
            return False

        sections = loaded_data.keys()

        if not("train" in sections):
            raise Exception("Section \"train\" is abscent!")
        if not("test" in sections):
            raise Exception("Section \"test\" is abscent!")
        if not("cv" in sections):
            raise Exception("Section \"cv\" is abscent!")

        print json.dumps(loaded_data, separators=(',',':'))

        print loaded_data["train"]
        print loaded_data["test"]
        print loaded_data["cv"]

        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory + "/train", loaded_data["train"])
        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory + "/test", loaded_data["test"])
        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory + "/cv", loaded_data["cv"])


    def _loadDataProcess_(self, data, file_name):
        if data:
            self._loadDataProcess_impl_(data)
        if file_name:
            data = ''
            with open(file_name, 'r') as datafile:
                data = datafile.read().replace('\n', '')
            self._loadDataProcess_impl_(data)


    def loadData(self, data):
        """
        data is a JSON string with secions: train, cv, test
        """

        self._loadDataProcess_(data, None)


    def loadDataFromFile(self, file_name):

        self._loadDataProcess_(None, file_name)


    def loadAudioFeaturesToMem(self):
        print "[CLASSIFIER] loading train data"
        labels = self._loader.readFeaturesToMem(self._resultFeaturesDirectory + "/train")
        x, y = self._loader.makeFeatureMatrixAndLabels(labels)
        self._trainData.setX(x)
        self._trainData.setY(y)
        print "[CLASSIFIER] train data shape"
        print self._trainData.X().shape

        print "[CLASSIFIER] loading test data"
        labels = self._loader.readFeaturesToMem(self._resultFeaturesDirectory + "/test")
        x, y = self._loader.makeFeatureMatrixAndLabels(labels)
        self._testData.setX(x)
        self._testData.setY(y)
        print "[CLASSIFIER] test data shape"
        print self._testData.X().shape

        print "[CLASSIFIER] loading cv data"
        labels = self._loader.readFeaturesToMem(self._resultFeaturesDirectory + "/cv")
        x, y = self._loader.makeFeatureMatrixAndLabels(labels)
        self._cvData.setX(x)
        self._cvData.setY(y)
        print "[CLASSIFIER] cv data shape"
        print self._cvData.X().shape


    def plotDataset(self):

        self._loader.runFeatureStandardScaler(self._trainData)

        self._loader.plotReducedFeatures(self._trainData)


    def preprocessAudioFeatures(self):

        #self._loader.runFeatureMinMaxNormalization(self._trainData, True)
        #self._loader.runFeatureStandardization(self._trainData, True)
        self._loader.runFeatureStandardScaler(self._trainData)
        #self._loader.unscaleFeatures(self._trainData)


        #self._loader.performUnvariatefeatureSelection(self._trainData, "k_best", "f_classif", 50) # что-то выдало))
        #self._loader.performUnvariatefeatureSelection("k_best", "mutual_info_classif", 50) # тоже норм, что-то выдало
        #self._loader.performUnvariatefeatureSelection("fpr", "f_classif", 50) # тоже ок
        #self._loader.performUnvariatefeatureSelection("fdr", "f_classif", 50) # тоже ок
        #self._loader.performUnvariatefeatureSelection("fwe", "f_classif", 50) # тоже ок

        #self._loader.performRecursiveFeatureSelection(self._trainData, 50) #ok
        #self._loader.performRecursiveFeatureSelection(5) #ok

        #self._loader.performL1BasedFeatureSelection(self._trainData)
        #self._loader.performTreeBasedFeatureSelection(self._trainData)

        #self._loader.performPCAFeatureSelection(self._trainData)

        #self._loader.plotReducedFeatures()

        #self._loader.labelEncoding()


    def trainClassifier(self):
        pass


    def trainkNeighborsCLassifier(self):
        pass

    def trainSVC(kernel="linear", C=0.025, gamma = 2):
        pass

    #def trainGaussianProcessClassifier(self, 1.0 * RBF(1.0), warm_start=True):
    #    pass

    def trainDecisionTreeClassifier(self, max_depth=5):
        pass

    def trainRandomForestClassifier(self, max_depth=5, n_estimators=10, max_features=1):
        pass

    def trainMLPClassifier(self, alpha=1):
        pass

    def trainAdaBoostClassifier(self):
        pass

    def trainGaussianNB(self):
        pass

    def trainQuadraticDiscriminantAnalysis(self):
        pass


# сохранение результатов
# http://scikit-learn.org/stable/modules/model_persistence.html
# from sklearn.externals import joblib
# >>> joblib.dump(clf, 'filename.pkl') 
# Later you can load back the pickled model (possibly in another Python process) with:
# >>> clf = joblib.load('filename.pkl')

# построение кривых MSE и lerning curve
# http://scikit-learn.org/stable/modules/learning_curve.html


