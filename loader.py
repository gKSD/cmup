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
class Loader:
    def __init__(self, storage):
        self.__storage = storage
        self.__data = []
        self._audioFeatureExtracter = AudioFeatureExtracter()
        self._features = {}

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
