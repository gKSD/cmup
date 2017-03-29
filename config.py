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
                    #self.emotionClassifierLowlevelAudioFeatures = self.__parseFeatures__(self.emotionClassifierLowlevelAudioFeatures)
                if emotionClassifier.get("emotionClassifierTonalAudioFeatures"):
                    self.emotionClassifierTonalAudioFeatures = emotionClassifier["emotionClassifierTonalAudioFeatures"].split('\n')
                    #self.emotionClassifierTonalAudioFeatures = self.__parseFeatures__(self.emotionClassifierTonalAudioFeatures)
                if emotionClassifier.get("emotionClassifierRhythmAudioFeatures"):
                    self.emotionClassifierRhythmAudioFeatures = emotionClassifier["emotionClassifierRhythmAudioFeatures"].split('\n')
                    #self.emotionClassifierRhythmAudioFeatures = self.__parseFeatures__(self.emotionClassifierRhythmAudioFeatures)
                if emotionClassifier.get("emotionClassifierAudioFeatureTypes"):
                    self.emotionClassifierAudioFeatureTypes = emotionClassifier["emotionClassifierAudioFeatureTypes"].split('\n')
                    #self.emotionClassifierAudioFeatureTypes = self.__parseFeatures__(self.emotionClassifierAudioFeatureTypes)


    def __parseFeatures__(self, param):
        #TODO: make it correct and nicer
        res = []

        for it in param:
            types = it.split(":")
            if len(types) == 2:
                tmp = {types[0]:[]}

                ar = types[1].split(",")
                for ar_it in ar:
                    if ar_it == "mean":
                        tmp[types[0]].append("mean")
                    if ar_it == "min":
                        tmp[types[0]].append("min")
                    if ar_it == "max":
                        tmp[types[0]].append("max")
                    if ar_it == "dvar2":
                        tmp[types[0]].append("dvar2")
                    if ar_it == "median":
                        tmp[types[0]].append("median")
                    if ar_it == "dmean2":
                        tmp[types[0]].append("dmean2")
                    if ar_it == "dmean":
                        tmp[types[0]].append("dmean")
                    if ar_it == "var":
                        tmp[types[0]].append("var")
                    if ar_it == "dvar":
                        tmp[types[0]].append("dvar")

                res.append(tmp)
            else:
                res.append(it)

        return res

    @classmethod
    def get_instance(self):
        if self.obj is None:
            self.obj = Config()
        return self.obj
