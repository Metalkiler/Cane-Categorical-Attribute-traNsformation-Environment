import numpy.testing
import numpy as np
import pandas as pd
import random
import string
import unittest

from cane import scale_data, scale_single_min_max, scale_single_std

class TestPCP(unittest.TestCase):

    @staticmethod
    def create_series():
        dfNumbers = pd.DataFrame({'A': {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 1},
                                  'B': {0: 1, 1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2, 9: 0},
                                  'C': {0: 2, 1: 2, 2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 2, 9: 1}})
        return dfNumbers

    @staticmethod
    def create_expected():
        """
           A_scalled_std
            0          -0.62
            1          -0.62
            2           1.86
            3          -0.62
            4           1.86
            5           0.62
            6          -0.62
            7          -0.62
            8          -0.62
            9          -0.62
        :return:
        """
        xA_new = pd.DataFrame([-1.09, 0.47, 0.47, 0.47, -1.09, -1.09, -1.09, 0.47, 2.03, 0.47],
                              columns=["A_scalled_std"])
        return xA_new


    @staticmethod
    def create_expected_min_max():
        """
           A_scalled_min_max
        0                0.0
        1                0.5
        2                0.5
        3                0.5
        4                0.0
        5                0.0
        6                0.0
        7                0.5
        8                1.0
        9                0.5
        :return:
        """
        xA_new = pd.DataFrame([0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],
                              columns=["A_scalled_min_max"])
        return xA_new
    def teststd(self):
        df = self.create_series()
        a = scale_single_std(df["A"])

        testA = self.create_expected()
        numpy.testing.assert_array_equal(a, testA, "Falhou no Standard Deviation Scalling")

    def testminmax(self):
        df = self.create_series()
        a = scale_single_min_max(df["A"])

        testA = self.create_expected_min_max()
        numpy.testing.assert_array_equal(a, testA, "Falhou no min max Scalling")
