from classifier import Classifier

class EmotionClassifier(Classifier):
    def __init__(self, storage, directory = None):
        Classifier.__init__(self, storage, directory)
