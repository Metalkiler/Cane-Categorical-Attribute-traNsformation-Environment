import numpy.testing
import pandas
import random
import string
import unittest

from cane import __pcp_single__, pcp_multicolumn, idf_multicolumn, __idf_single_dic__, __idf_single__


class TestPCP(unittest.TestCase):
    @staticmethod
    def create_series(cases):
        lst = [item.split() for item in cases.splitlines() if item.strip()]
        return pandas.Series([k for s in ([k] * int(n) for k, n in lst) for k in s])

    @staticmethod
    def create_df():
        x2 = [k for s in ([k] * n for k, n in [('a', 50),
                                               ('b', 10),
                                               ('c', 20),
                                               ('d', 15),
                                               ('e', 5)]) for k in s]

        x3 = [k for s in ([k] * n for k, n in [('a', 40),
                                               ('b', 20),
                                               ('c', 1),
                                               ('d', 1),
                                               ('e', 38)]) for k in s]
        df2 = pandas.concat(
            [pandas.DataFrame({f'x{i}': x2 for i in range(1, 3)}),
             pandas.DataFrame({f'y{i}': x3 for i in range(1, 3)})],
            axis=1)
        return df2

    @staticmethod
    def create_df_expected():
        x1_new = [k for s in ([k] * n for k, n in [('a', 50),
                                                   ('b', 10),
                                                   ('c', 20),
                                                   ('Others', 15),
                                                   ('e', 5)]) for k in s]

        x2 = [k for s in ([k] * n for k, n in [('a', 50),
                                               ('b', 10),
                                               ('c', 20),
                                               ('d', 15),
                                               ('e', 5)]) for k in s]

        x3 = [k for s in ([k] * n for k, n in [('a', 40),
                                               ('b', 20),
                                               ('c', 1),
                                               ('d', 1),
                                               ('e', 38)]) for k in s]

        y_new = [k for s in ([k] * n for k, n in [('a', 40),
                                                  ('b', 20),
                                                  ('c', 1),
                                                  ('Others', 1),
                                                  ('e', 38)]) for k in s]

        df2 = pandas.DataFrame()

        df2['x1'] = x1_new
        df2['x2'] = x2
        df2['y1'] = y_new
        df2['y2'] = x3

        return df2

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
        df = TestPCP.create_df()
        expected = TestPCP.create_df_expected()
        filtered = pcp_multicolumn(df, mergeCategory="Others", columns_use=["x1", "y1"])

        pandas.testing.assert_frame_equal(expected, filtered, "falhou o multicolumn")


if __name__ == '__main__':
    unittest.main()
