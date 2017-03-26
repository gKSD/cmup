import configparser

class Config(object):
    obj = None

    CONFIG_PATH = './conf/cmup_conf.cfg'

    def __init__(self):
        if Config.obj is not None:
            raise Exception('[ERROR] A Config instance already exists')

        # setting default values
        self.sampleRate = 44100
        self.channels = 1

        self.midTermSize = 1.0
        self.midTermStep = 1.0
        self.shortTermSize = 0.050
        self.shortTermStep = 0.050
        self.computeBeat = True
        self.audioFeaturesResultDirectory = "user_calculated_essentia_features"

        self.classifierAudioFeaturesResultDirectory = "calculated_essentia_features"

        self.emotionClassifierLowlevelAudioFeatures = ""
        self.emotionClassifierTonalAudioFeatures = ""
        self.emotionClassifierRhythmAudioFeatures = ""
        self.emotionClassifierAudioFeatureTypes = ""

        print "[CONFIG] reading config: '" + self.CONFIG_PATH + "'"

        self.config = configparser.ConfigParser()
        self.config.read(self.CONFIG_PATH)
        for section in self.config.sections():
            if section == "AudioBasicIO":
                audioBasicIO = self.config["AudioBasicIO"]

                if audioBasicIO.get("samplerate"):
                    self.sampleRate = int(audioBasicIO["samplerate"])
                if audioBasicIO.get("channels"):
                    self.channels = int(audioBasicIO["channels"])

            elif section == "AudioFeatureExtraction":
                audiofeatureExtraction = self.config["AudioFeatureExtraction"]

                if audiofeatureExtraction.get("midTermSize"):
                    self.midTermSize = float(audiofeatureExtraction["midTermSize"])
                if audiofeatureExtraction.get("midTermStep"):
                    self.midTermStep = float(audiofeatureExtraction["midTermStep"])
                if audiofeatureExtraction.get("shortTermSize"):
                    self.shortTermSize = float(audiofeatureExtraction["shortTermSize"])
                if audiofeatureExtraction.get("shortTermStep"):
                    self.shortTermStep = float(audiofeatureExtraction["shortTermStep"])
                if audiofeatureExtraction.get("computeBeat"):
                    self.computeBeat = audiofeatureExtraction.getboolean("computeBeat")
                if audiofeatureExtraction.get("audioFeaturesResultDirectory"):
                    self.audioFeaturesResultDirectory = audiofeatureExtraction["audioFeaturesResultDirectory"]

            elif section == "Classifier":
                classifier = self.config["Classifier"]

                if classifier.get("classifierAudioFeaturesResultDirectory"):
                    self.classifierAudioFeaturesResultDirectory = classifier["classifierAudioFeaturesResultDirectory"]

            elif section == "EmotionClassifier":
                emotionClassifier = self.config["EmotionClassifier"]

                if emotionClassifier.get("emotionClassifierLowlevelAudioFeatures"):
                    self.emotionClassifierLowlevelAudioFeatures = emotionClassifier["emotionClassifierLowlevelAudioFeatures"].split('\n')
                if emotionClassifier.get("emotionClassifierTonalAudioFeatures"):
                    self.emotionClassifierTonalAudioFeatures = emotionClassifier["emotionClassifierTonalAudioFeatures"].split('\n')
                if emotionClassifier.get("emotionClassifierRhythmAudioFeatures"):
                    self.emotionClassifierRhythmAudioFeatures = emotionClassifier["emotionClassifierRhythmAudioFeatures"].split('\n')
                if emotionClassifier.get("emotionClassifierAudioFeatureTypes"):
                    self.emotionClassifierAudioFeatureTypes = emotionClassifier["emotionClassifierAudioFeatureTypes"].split('\n')


    @classmethod
    def get_instance(self):
        if self.obj is None:
            self.obj = Config()
        return self.obj
