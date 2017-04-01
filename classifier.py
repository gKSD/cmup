#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re
import os
import numpy


from loader import Loader
from config import Config
from storage import Storage

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


    def loadAudioFeaturesToMem(self):

        self._loader.readFeaturesToMem()

        self._loader.makeFeatureMatrixAndLabels()


    def plotDataset(self):

        self._loader.runFeatureStandardScaler()

        self._loader.plotReducedFeatures()


    def preprocessAudioFeatures(self):

        #self._loader.runFeatureMinMaxNormalization(True)
        #self._loader.runFeatureStandardization(True)
        self._loader.runFeatureStandardScaler()
        #self._loader.unscaleFeatures()

        self._loader.performUnvariatefeatureSelection("k_best", "f_classif", 50) # что-то выдало))
        #self._loader.performUnvariatefeatureSelection("k_best", "mutual_info_classif", 50) # тоже норм, что-то выдало
        #self._loader.performUnvariatefeatureSelection("fpr", "f_classif", 50) # тоже ок
        #self._loader.performUnvariatefeatureSelection("fdr", "f_classif", 50) # тоже ок
        #self._loader.performUnvariatefeatureSelection("fwe", "f_classif", 50) # тоже ок

        #self._loader.performRecursiveFeatureSelection(50) #ok

        #self._loader.performPCAFeatureSelection()

        #self._loader.plotReducedFeatures()

        #self._loader.labelEncoding()


