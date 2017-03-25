#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import re

from loader import Loader
from config import Config
from storage import Storage

class classifier:
    def __init__(self, storage):
        self.__storage = storage
        self.__loader = Loader(self.__storage)

        self.__resultFeaturesDirectory = config.classifierAudioFeaturesResultDirectory

    def __loadDataProcess_impl__(self, input_data):
        config = Config.get_instance()

        self.__loader.load(input_data)
        self.__loader.printData()
        self.__loader.extractFeaturesEssentia(self.__resultFeaturesDirectory)

    def __loadDataProcess__(self, data, file_name):
        if data:
            self.__loadDataProcess_impl__(data)
        if file_name:
            data = ''
            with open(file_name, 'r') as datafile:
                data = datafile.read().replace('\n', '')
            self.__loadDataProcess_impl__(data)

    def loadData(self, data):
        self.__loadDataProcess__(self, data, None)

    def loadDataFromFile(self, file_name):
        self.__loadDataProcess__(self, None, file_name)

    def __readFeaturesToMem(self):
        pass
