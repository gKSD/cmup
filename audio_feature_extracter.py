#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import os
import platform
import glob
import shutil
import subprocess
import re

from lib.audio_feature_extraction import audioFeatureExtraction as aF
from lib.audio_feature_extraction import audioTrainTest as aT
from lib.audio_feature_extraction import audioBasicIO as aB
class AudioFeatureExtracter:

    def __init(self):
        pass

    def processFile(self,
                    musicPath,
                    midTermSize,
                    midTermStep,
                    shortTermSize,
                    shortTermStep,
                    computeBeat):
        print "[CUSTOM_AUDIO_FEATURE_EXTRACTION] music_path: " + musicPath
        try:
            audioFeatures = aF.fileWavFeatureExtraction(musicPath,
                                                        midTermSize,
                                                        midTermStep,
                                                        shortTermSize,
                                                        shortTermStep,
                                                        computeBeat)
        except Exception as e:
            print str(e)
            raise

        print "[DEBUG] calculated mid features shape: " + str(audioFeatures.shape)
        return audioFeatures

    def processDirs(self,
                    resultFeatures,
                    resultFileNames,
                    dirNames,
                    midTermSize,
                    midTermStep,
                    shortTermSize,
                    shortTermStep,
                    computeBeat):
        #[features, classNames, fileNames] = aF.dirsWavFeatureExtraction(dirNames,
        #                                                             midTermSize,
        #                                                             midTermStep,
        #                                                             shortTermSize,
        #                                                             shortTermStep,
        #                                                             computeBeat)
        [resultFeatures, resultFileNames] = aF.dirsWavFeatureExtractionExtension(resultFeatures,
                                                                                 resultFileNames,
                                                                                 dirNames,
                                                                                 midTermSize,
                                                                                 midTermStep,
                                                                                 shortTermSize,
                                                                                 shortTermStep,
                                                                                 computeBeat)
        return [resultFeatures, resultFileNames]
        #return [features, classNames, fileNames]

    def processFileEssentia(self, music_file_name, destination_dir):

        print "[AUDIO_FEATURE_EXTRACTOR] processing file: " + music_file_name

        if not music_file_name.lower().endswith(('.wav', '.aif',  '.aiff', '.mp3','.au')):
            raise Exception("Wrong file extension!")

        #TMP = "tmp_essentia"
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        result_directory = destination_dir + '/'

        extractor = ''
        if platform.system() == 'Darwin':
            extractor = "./lib/essentia_osx/streaming_extractor_music"
        elif platform.system() == 'Linux':
            extractor = "./lib/essentia_linux/streaming_extractor_music"
        else:
            raise Exception("Unsupported OS!!")

        try:
            print "\033[1;36m[AUDIO_FEATURE_EXCTRACTOR] Analyzing file: {2:s}\033[0;0m".format(music_file_name.encode('utf-8', errors='ignore'))
            file_name = os.path.basename(music_file_name) # extract file name from full path
            # >>> filename = 'to_delete/qqq1/qqq2/my_file.txt'
            # >>> os.path.splitext(filename)[0]
            # >>> 'to_delete/qqq1/qqq2/my_file'
            # >>> os.path.splitext(os.path.basename(filename))[0]
            # >>> 'my_file'
            clean_file_name = os.path.splitext(file_name)[0] # gets file name without extension
            parent_folder_identifier = re.sub('\.', '', os.path.dirname(wavFile))
            parent_folder_identifier = re.sub('/', '_', os.path.dirname(parent_folder_identifier)) 
            result_file_name = result_directory + "/" + parent_folder_identifier + "_" + clean_file_name + "_features.json"
            #result_file_name = re.sub('[ -\(\)\']', '', result_file_name)

            print "[AUDIO_FEATURE_EXCTRACTOR] result file name: " + result_file_name.encode('utf-8', errors='ignore')

            command = extractor + " \"" + music_file_name + "\" \"" + result_file_name + "\""
            print "[AUDIO_FEATURE_EXCTRACTOR] command: " + command.encode('utf-8', errors='ignore')
            os.system(command.encode('utf-8', errors='ignore'))
            #exitcode = subprocess.call([command])

        except ValueError, e:
            print "[ERROR] exception occured while processing file, skip [" + str(e) + "]"
            raise


    def processDirEssentia(self, dir_name, destination_dir):

        print "[AUDIO_FEATURE_EXTRACTOR] processing dir: " + dir_name

        #TMP = "tmp_essentia"
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        result_directory = destination_dir + '/'
        #result_directory = destination_dir + '/' + re.sub('\.\./', '', dir_name)
        #if not os.path.exists(result_directory):
        #    os.makedirs(result_directory)
        try:
            extractor = ''
            if platform.system() == 'Darwin':
                extractor = "./lib/essentia_osx/streaming_extractor_music"
            elif platform.system() == 'Linux':
                extractor = "./lib/essentia_linux/streaming_extractor_music"
            else:
                raise Exception("Unsupported OS!!")

            types = ('*.wav', '*.aif',  '*.aiff', '*.mp3','*.au')

            wavFilesList = []

            for files in types:
                wavFilesList.extend(glob.glob(os.path.join(dir_name, files)))

            file_count = 0
            files_total = len(wavFilesList)
            for wavFile in wavFilesList:
                try:
                    print "\033[1;36m[AUDIO_FEATURE_EXCTRACTOR] Analyzing file {0:d} of {1:d}: {2:s}\033[0;0m".format(file_count + 1, files_total, wavFile.encode('utf-8', errors='ignore'))

                    file_count += 1

                    file_name = os.path.basename(wavFile) # extract file name from full path
                    clean_file_name = os.path.splitext(file_name)[0] # gets file name without extension
                    parent_folder_identifier = re.sub('\.', '', os.path.dirname(wavFile))
                    parent_folder_identifier = re.sub('/', '_', os.path.dirname(parent_folder_identifier))
                    result_file_name = result_directory + "/" + parent_folder_identifier + "_" + clean_file_name + "_features.json"
                    #result_file_name = re.sub('[ -\(\)\']', '', result_file_name)

                    print "[AUDIO_FEATURE_EXCTRACTOR] result file name: " + result_file_name.encode('utf-8', errors='ignore')

                    command = extractor + " \"" + wavFile + "\" \"" + result_file_name + "\""
                    print "[AUDIO_FEATURE_EXCTRACTOR] command: " + command.encode('utf-8', errors='ignore')
                    os.system(command.encode('utf-8', errors='ignore'))
                    #exitcode = subprocess.call([command])

                except ValueError, e:
                    print "[ERROR] exception occured while processing file, skip [" + str(e) + "]"

        except:
            raise

    def normalizeFeatures(self, features):
        # features -> list of feature matrices
#        (featuresNorm, MEAN, STD) = normalize()
#        return (featuresNorm, MEAN, STD)
        print "[CUSTOM_AUDIO_FEATURE_EXTRACTION] feature normalization"
        [featuresNorm, MEAN, STD] = aT.normalizeFeatures(features)        # normalize features
        MEAN = MEAN.tolist()
        STD = STD.tolist()
        return [featuresNorm, MEAN, STD]

    def normalizeFilesFeatures(self):
#
        allMtFeatures = numpy.array([])
        if len(allMtFeatures) == 0:                              # append feature vector
            allMtFeatures = MidTermFeatures
        else:
            allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
        return self.normalizeFeatures([allMtFeatures])


    @staticmethod
    def convertDirMp3ToWav(directory, sampleRate, channels, useMp3TagsAsNames):
        if not os.path.isdir(directory):
            raise Exception("Input directory path not found!")

        aB.convertDirMP3ToWav(directory, sampleRate, channels, useMp3TagsAsNames)

    @staticmethod
    def dirWavChangeFs(directory, samplerate, channels):
        if not os.path.isdir(directory):
            raise Exception("Input path not found!")

        aB.convertFsDirWavToWav(directory, samplerate, channels)

    def beatExtractionWrapper(self, wavFileName, plot):
        if not os.path.isfile(wavFileName):
            raise Exception("Input audio file not found!")

        [Fs, x] = aB.readAudioFile(wavFileName)
        F = aF.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.050 * Fs)
        BPM, ratio = aF.beatExtraction(F, 0.050, plot)
        print "Beat: {0:d} bpm ".format(int(BPM))
        print "Ratio: {0:.2f} ".format(ratio)
        return [Beat, ratio]

"""
import audioFeatureExtraction as aF
import audioVisualization as aV
import audioBasicIO
import utilities as uT
import scipy.io.wavfile as wavfile
import matplotlib.patches
import librosa

def featureExtractionFileWrapper(wavFileName, outFile, mtWin, mtStep, stWin, stStep):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")

    aF.mtFeatureExtractionToFile(wavFileName, mtWin, mtStep, stWin, stStep, outFile, True, True, True)



def beatExtractionWrapper(wavFileName, plot):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    print "KSD beatExtractionWrapper"
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName)
    F = aF.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.050 * Fs)
    BPM, ratio = aF.beatExtraction(F, 0.050, plot)
    print "Beat: {0:d} bpm ".format(int(BPM))
    print "Ratio: {0:.2f} ".format(ratio)

    tempo, beat_frames = librosa.beat.beat_track(x, Fs, start_bpm=60)
    print "KSD TEMPO => "
    print tempo
    print "KSD BEAT_FRAMES => "
    print beat_frames
    onset_env = librosa.onset.onset_strength(x, sr=Fs)
    tempo = librosa.beat.estimate_tempo(onset_env, sr=Fs, start_bpm=60)
    print "GLOBAL TEMPO => " + str (tempo)
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=Fs)
    #print "Beat: {0:d} bpm ".format(int(beat_times))


def featureExtractionDirWrapper(directory, mtWin, mtStep, stWin, stStep):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")
    aF.mtFeatureExtractionToFileDir(directory, mtWin, mtStep, stWin, stStep, True, True, True)


def featureVisualizationDirWrapper(directory):
    if not os.path.isdir(directory):
        raise Exception("Input folder not found!")
    aV.visualizeFeaturesFolder(directory, "pca", "")
    #aV.visualizeFeaturesFolder(directory, "lda", "artist")


def fileSpectrogramWrapper(wavFileName):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True)

def fileChromagramWrapper(wavFileName):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stChromagram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True)
"""

