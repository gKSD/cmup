# -*- coding: UTF-8 -*-
from loader import Loader

class Data():

    """
    just store features matrix, labels
    and preprocessing state - scaling type, selection type
    """

    def __init__(self, X = None, Y = None):

        """
        X - numpy.array matrix
        """

        self._X = [] # feature matrix
        self._Y = [] # labels
        self._X_polynomial = []

        self._scalingType   = Loader.ScalingType.ST_NONE
        self._selectionType = Loader.FeatureSelectionType.FST_NONE

        self._labelsEncoded = False


    def X(self):
        return self._X

    def Y(self):
        return self._Y

    def Xpoly(self):
        return self._X_polynomial

    def setX(self, X):
        self._X = X

    def setY(self, Y):
        self._Y = Y

    def setXpoly(self, Xpoly):
        self._X_polynomial = Xpoly

    def scalingType(self):
        return self._scalingType

    def selectionType(self):
        return self._selectionType

    def setScalingType(self, stype):
        self._scalingType = stype

    def setSelectionType(self, stype):
        self._selectionType = stype

    def labelsEncoded(self):
        return self._labelsEncoded

    def setLabelsEncoded(self, st):
        labelsEncoded = st
