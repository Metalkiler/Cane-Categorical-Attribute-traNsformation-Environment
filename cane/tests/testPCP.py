import numpy.testing
import pandas
import random
import string
import unittest

from cane import __pcp_single__
from cane import __multicolumnPCP__

class TestPCP(unittest.TestCase):
    @staticmethod
    def create_series(cases):
        lst = [item.split() for item in cases.splitlines() if item.strip()]
        return pandas.Series([k for s in ([k] * int(n) for k, n in lst) for k in s])

    @staticmethod
    def create_scrambled_series(cases):
        return create_series(cases).sample(frac=1)

    def setUp(self):
        self.series = TestPCP.create_series("""
            c 89
            a 6
            d 2
            b 2
            e 1
        """)

    def test_PCP_5(self):
        """
            This tests an edge case, 
        """
        before = self.series
        filtered = __pcp_single__(before)
        expected = TestPCP.create_series("""
            c 89
            a 6
            Others 2
            Others 2
            Others 1
        """)

        numpy.testing.assert_array_equal(expected, filtered, "falhou o caso do 5%")

    def test_PCP_10(self):
        before = self.series
        filtered = __pcp_single__(before, perc_inner=0.1, mergeCategoryinner="other")
        expected = TestPCP.create_series("""
            c 89
            other 6
            other 2
            other 2
            other 1
        """)

        numpy.testing.assert_array_equal(expected, filtered, "falhou o caso do 10%")

    def test_PCP_BIG(self):
        """
            Um teste onde há 26 valores diferentes, o primeiro aparece uma vez, o segundo duas vezes, etc
            As letras são permutadas aleatóriamente para evitar qualquer bias
            A seguir, os dados são permutados aleatóriamente para que a série não tenha os valores seguidos
        """
        scramble = random.sample(string.ascii_letters, 2 * 26)
        not_used = set(scramble[:12])

        x = [k for s in ([k] * n for n, k in enumerate(scramble)) for k in s]
        y = random.sample(x, len(x))
        X = pandas.Series(y)
        expected = pandas.Series([k if k not in not_used else "other" for k in y])
        filtered = __pcp_single__(X, mergeCategoryinner="other")

        numpy.testing.assert_array_equal(expected, filtered)

    def test_PCP_multicolumn(self):
        """
            Um teste onde há 26 valores diferentes, o primeiro aparece uma vez, o segundo duas vezes, etc
            As letras são permutadas aleatóriamente para evitar qualquer bias
            A seguir, os dados são permutados aleatóriamente para que a série não tenha os valores seguidos
        """
        scramble = random.sample(string.ascii_letters, 2 * 26)
        not_used = set(scramble[:12])

        x = [k for s in ([k] * n for n, k in enumerate(scramble)) for k in s]
        y = random.sample(x, len(x))
        X = pandas.Series(y)
        expected = pandas.Series([k if k not in not_used else "other" for k in y])
        filtered = __pcp_single__(X, mergeCategoryinner="other")

        numpy.testing.assert_array_equal(expected, filtered)


if __name__ == '__main__':
    unittest.main()
