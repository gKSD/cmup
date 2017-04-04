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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.externals import joblib

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

        self._hasCV = False


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

        if "cv" in sections:
            self._hasCV = True

        print json.dumps(loaded_data, separators=(',',':'))

        print loaded_data["train"]
        print loaded_data["test"]
        print loaded_data["cv"]

        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory + "/train", loaded_data["train"])
        self._loader.extractFeaturesEssentia(self._resultFeaturesDirectory + "/test", loaded_data["test"])

        if self._hasCV:
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

        if self._hasCV:
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
        self._loader.runFeatureStandardScaler(self._testData)
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

        self._loader.performPCAFeatureSelection(self._trainData, 2)
        self._loader.performPCAFeatureSelection(self._testData, 2)

        #self._loader.plotReducedFeatures()

        self._loader.labelEncoding(self._trainData)
        self._loader.labelEncoding(self._testData)


    def trainClassifier(self):
        pass


    def trainTunnedSVC(self):
        tuned_parameters = [{
                              'kernel': ['rbf'],
                              'gamma': [0.1, 0.5, 1e-2, 3e-2, 5e-2, 7e-2, 1e-3, 3e-3, 5e-3, 7e-3, 1e-4],
                              'C': [0.025, 1, 10, 100, 1000]
                            },
                            {
                              'kernel': ['linear'],
                              'C': [0.025, 1, 10, 100, 1000]}
                            ]

        scores = ['precision', 'recall', 'f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1, decision_function_shape="ovo"), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(self._trainData.X(), self._trainData.Y())

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self._testData.Y(), clf.predict(self._testData.X())
            print(classification_report(y_true, y_pred))
            print()


    def trainAndPlotSVC(self, object_file, need_load_obj = False):

        if object_file is None:
            raise Exception("Object file for storing SVC classifier object is None!")

        figure = plt.figure(figsize=(27, 9))

        x_min, x_max = self._trainData.X()[:, 0].min() - 1, self._trainData.X()[:, 0].max() + 1
        y_min, y_max = self._trainData.X()[:, 1].min() - 1, self._trainData.X()[:, 1].max() + 1

        h = .02 # step size in the mesh (сетка)

        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        print "XX.shape => "
        print xx.shape

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        ax = plt.subplot(1, 1, 1)
        ax.axis('tight')
        ax.set_title("SVC")

        #if not os.path.exists(object_file):
        #    need_load_obj = False

        #clf = None
        #if need_load_obj:
        #    print "[CLASSIFIER] loading object file: " + object_file
        #    clf = joblib(object_file)
        #else:
        print "[CLASSIFIER] creating new SVC object"
        clf = SVC(kernel="rbf", C=1, gamma=0.001)
        print "[CLASSIFIER] fit SVC object"
        clf.fit(self._trainData.X(), self._trainData.Y())
        #print "[CLASSIFIER] dumping result to object file: " + object_file
        #joblib.dump(clf, object_file)
        score = clf.score(self._testData.X(), self._testData.Y())

        print "[CLASSIFIER] predict results for points from mechgrid"
        Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
        print "[CLASSIFIER] reshaping Z"
        Z = Z.reshape(xx.shape)
        print "[CLASSIFIER] ploting countours"
        ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        for i in numpy.unique(self._trainData.Y()):
            print "class: " + str(i)
            idx = numpy.where(self._trainData.Y() == i)
            color=None
            if i == 0:
                color='y'
            elif i == 1:
                color='b'
            elif i == 2:
                color='r'
            elif i == 3:
                color='g'
            ax.scatter(self._trainData.X()[idx, 0], self._trainData.X()[idx, 1], c=color, cmap=plt.cm.Paired,
                    label=self._loader.labelsEncoder().classes_[i])
        # and testing points
        #ax.scatter(self._testData.X()[:, 0], self._testData.X()[:, 1], c=self._testData.Y(), cmap=cm_bright,alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')

        plt.tight_layout()
        plt.show()


    def trainAndPlotkNeighbors(self):
        figure = plt.figure(figsize=(27, 9))

        x_min, x_max = self._trainData.X()[:, 0].min() - 1, self._trainData.X()[:, 0].max() + 1
        y_min, y_max = self._trainData.X()[:, 1].min() - 1, self._trainData.X()[:, 1].max() + 1

        h = .02 # step size in the mesh (сетка)

        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        print "XX.shape => "
        print xx.shape

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        ax = plt.subplot(1, 1, 1)
        ax.axis('tight')
        ax.set_title("kNeighbors")

        print "[CLASSIFIER] creating new kNeighbors object"
        clf = KNeighborsClassifier(n_neighbors=45, metric='manhattan', weights='uniform')
        print "[CLASSIFIER] fit kNeighbors object"
        clf.fit(self._trainData.X(), self._trainData.Y())
        #print "[CLASSIFIER] dumping result to object file: " + object_file
        #joblib.dump(clf, object_file)
        score = clf.score(self._testData.X(), self._testData.Y())

        print "[CLASSIFIER] predict results for points from mechgrid"
        Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
        print "[CLASSIFIER] reshaping Z"
        Z = Z.reshape(xx.shape)
        print "[CLASSIFIER] ploting countours"
        ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        for i in numpy.unique(self._trainData.Y()):
            print "class: " + str(i)
            idx = numpy.where(self._trainData.Y() == i)
            color=None
            if i == 0:
                color='y'
            elif i == 1:
                color='b'
            elif i == 2:
                color='r'
            elif i == 3:
                color='g'
            ax.scatter(self._trainData.X()[idx, 0], self._trainData.X()[idx, 1], c=color, cmap=plt.cm.Paired,
                    label=self._loader.labelsEncoder().classes_[i])
        # and testing points
        #ax.scatter(self._testData.X()[:, 0], self._testData.X()[:, 1], c=self._testData.Y(), cmap=cm_bright,alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')

        plt.tight_layout()
        plt.show()


    def trainTunnedKNeighbors(self):
        tuned_parameters = [
                             {
                                 'n_neighbors': [3, 15, 25, 30, 35, 40, 45],
                                 'metric': ['minkowski', 'euclidean', 'manhattan'],
                                 'weights': ['uniform', 'distance']
                             }
                           ]

        scores = ['precision', 'recall', 'f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(KNeighborsClassifier(n_neighbors=3), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(self._trainData.X(), self._trainData.Y())

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self._testData.Y(), clf.predict(self._testData.X())
            print(classification_report(y_true, y_pred))
            print()

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


