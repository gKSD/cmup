#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re

from loader import Loader
from config import Config
from storage import Storage

class Classifier:
    def __init__(self, storage):
        self._storage = storage
        self._loader = Loader(self._storage)

        self._resultFeaturesDirectory = Config.get_instance().classifierAudioFeaturesResultDirectory

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
        self._loadDataProcess_(self, data, None)

    def loadDataFromFile(self, file_name):
        self._loadDataProcess_(self, None, file_name)

    def _readFeaturesToMem(self):
        pass
