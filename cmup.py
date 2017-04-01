#!/usr/bin/env python2.7
import argparse
import os
import audioop
import numpy
import glob
import scipy
import subprocess
import wave
import cPickle
import threading
import shutil
import ntpath
import matplotlib.pyplot as plt

from config import Config
from audio_feature_extracter import AudioFeatureExtracter
from loader import Loader
from storage import Storage
from emotion_classifier import EmotionClassifier 


storage = Storage()

#TODO: logging system

def loadDataProcess_impl(input_data, ids):
    loader = Loader(storage)
    loader.load(input_data, ids)
    loader.printData()
    loader.extractFeaturesEssentia("to_delete")
    #loader.fingerprint("./input_data/m1.wav")
    loader.saveData()

def loadDataProcess(data, file_name, ids):
    if data:
        loadDataProcess_impl(data, ids)
    if file_name:
        data = ''
        with open(file_name, 'r') as datafile:
            data=datafile.read().replace('\n', '')
        loadDataProcess_impl(data, ids)

def updateDataProcess(data, file, ids):
    pass
    
def extractHighLevelFeaturesProcess(ids):
    pass

def runClassifierProcess(ids):
    pass

def getResultProcess():
    pass

def trainEmotionClassifier(file, data, load):
    config = Config.get_instance()

    tmp = "calculated_essentia_features/all_dataset_for_counting/4_dataset_audio/GROUPED/train/"

    ec = EmotionClassifier(storage, tmp)

    if file:
        ec.loadDataFromFile(file)
    elif data:
        ec.loadData(data)

    ec.loadAudioFeaturesToMem()

    #ec.plotDataset()

    ec.preprocessAudioFeatures()


def convertDirMp3ToWavProcess(directory, sample_rate, _channels, need_remove_original = False, use_mp3_tags = False):
    config = Config.get_instance()
    sampleRate = sample_rate if sample_rate else config.sampleRate
    channels = _channels if _channels else config.channels
    try:
        AudioFeatureExtracter.convertDirMp3ToWav(directory, sampleRate, channels, use_mp3_tags)
        if need_remove_original:
            types = (directory + os.sep + '*.mp3',) # the tuple of file types
            filesToProcess = []

            for files in types:
                filesToProcess.extend(glob.glob(files))

            print "[CMUP] deleting files: " + ', '.join(glob.glob(files))
            #print_msg("deleting files: " + ', '.join(glob.glob(files)), "CMUP")

            for file in filesToProcess:
                print "[CMUP] deleting current file: " + file
                #print_msg("deleting current file: " + file, "CMUP")
                os.remove(file)
    except:
        raise

def dirWavChangeFs(directory, sample_rate, _channels):
    config = Config.get_instance()
    sampleRate = sample_rate if sample_rate else config.sampleRate
    channels = _channels if _channels else config.channels
    try:
        AudioFeatureExtracter.dirWavChangeFs(directory, sampleRate, channels)
    except ValueError, e:
        print "[ERROR] exception occured while converting wav file to new wav file [" + str(e) + "]"
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="A demonstration script for pyAudioAnalysis library")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    loadData = tasks.add_parser("loadData",
                 help = "loads input data, converts to wav type and extracts features from music records.\n"
                        "Input data should be defined in JSON format.\n"
                        "Input data structure: \n"
                        "   {\n"
                        "      \"ID1\": {\"files\":[<path_to_music_file1>, <path_to_music_file2>, ..., <path_to_music_fileN>]},\n"
                        "      \"ID2\": {\"dirs\" :[<path_to_dir1>, <path_to_dir2>, ..., <path_to_dirN>]},\n"
                        "      ...,\n"
                        "      \"IDN\": {\n"
                                            "\"files\":\"[<path_to_music_file1>, <path_to_music_file2>, ..., <path_to_music_fileN>],\n"
                                            "\"dirs\":[<path_to_dir1>, <path_to_dir2>, ..., <path_to_dirN>]\n"
                                        "}\n"
                        "  }\n"
                        "ID - unique user identifier, is a string value, valid symbols: alphas, numbers, '_'.\n"
                        "FeatureExtraction is performed for load data.\n"
                        "Stored: user identifiers and arrays with extracted features respectively.")
    group = loadData.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to file with input data in JSON format")
    group.add_argument("-d", "--data", help="Input data in JSON format")
    loadData.add_argument("-i", "--ids", help="Process only defined user IDs")

    updateData = tasks.add_parser("updateData", help="detect changes, updates user data, converts new records to wav type and extract features from music records")
    group = updateData.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to file with input data in JSON format")
    group.add_argument("-d", "--data", help="Input data in JSON format")
    updateData.add_argument("-i", "--ids", help="Process only defined user IDs")

    extractHighLevelFeatures = tasks.add_parser("extractHighLevelFeatures", help="extract high-level features")
    extractHighLevelFeatures.add_argument("-i", "--ids", help="Process only defined user IDs")

    runClassifier = tasks.add_parser("runClassifier", help="run classifier on extracted high-level features")
    runClassifier.add_argument("-i", "--ids", help="Process only defined user IDs")

    convertDirMp3ToWav = tasks.add_parser("convertDirMp3ToWav", help="convert all mp3 in specified directory to wav")
    convertDirMp3ToWav.add_argument("-d", "--dir", required=True, help="directory with music files")
    convertDirMp3ToWav.add_argument("-r", "--remove", action="store_true", help="specify, if you wan't to delete original mp3 records")
    convertDirMp3ToWav.add_argument("-t", "--usemp3tags", action="store_true", help="specify, if you want to use mp3 tags as new wav files names")
    convertDirMp3ToWav.add_argument("-s", "--samplerate", help = "sampling rate")
    convertDirMp3ToWav.add_argument("-c", "--channels", help = "number of channels in new wav file")

    dirWavChangeFs = tasks.add_parser("dirWavChangeFs", help="convert all .wav files in specified directory to new .wav files with specified parameters")
    dirWavChangeFs.add_argument("-d", "--dir", required=True, help="directory with music files")
    dirWavChangeFs.add_argument("-s", "--samplerate", help = "sampling rate")
    dirWavChangeFs.add_argument("-c", "--channels", help = "number of channels in new wav file")

    trainEmotionClassifier = tasks.add_parser("trainEmotionClassifier", help = "trains Music Emotion Classifier")
    group = trainEmotionClassifier.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to file with input data in JSON format")
    group.add_argument("-d", "--data", help="Input data in JSON format")
    group.add_argument("-l", "--load", action="store_true", help="Load calculated audio features")

    getResult = tasks.add_parser("getResult", help="returns classifier results")
    #TODO: print result - plots etc
    #TODO: return result in JSON (save to file or somehow)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.task == "loadData":
        loadDataProcess(args.data, args.file, args.ids)
    elif args.task == "updateData":
        updateDataProcess(arga.data, args.file, args.ids)
    elif args.task == "extractHighLevelFeatures":
        extractHighLevelFeaturesProcess(args.ids)
    elif args.task == "runClassifier":
        runClassifierProcess(args.ids)
    elif args.task == "getResult":
        getResultProcess()
    elif args.task == "convertDirMp3ToWav":
        convertDirMp3ToWavProcess(args.dir, args.samplerate, args.channels, args.remove, args.usemp3tags)
    elif args.task == "dirWavChangeFs":
        dirWavChangeFs(args.dir, args.samplerate, args.channels)
    elif args.task == "trainEmotionClassifier":
        trainEmotionClassifier(args.file, args.data, args.load)
