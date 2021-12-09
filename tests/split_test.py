
import unittest
from pathlib import Path

import pandas as pd

from didtool.split import split_data, split_data_random, split_data_stacking


class TestSplit(unittest.TestCase):
    def test_split_data(self):
        data = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        data = split_data(data, data.index < 500, data.index >= 750)
        self.assertEqual(data[data.group == 0].shape[0], 500)
        self.assertEqual(data[data.group == 1].shape[0], 250)
        self.assertEqual(data[data.group == -1].shape[0], 250)

    def test_split_data_random(self):
        data = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))

        # test split_data
        data = split_data_random(data, 0.6, 0.2)
        self.assertEqual(data[data.group == 0].shape[0], 600)
        self.assertEqual(data[data.group == 1].shape[0], 200)
        self.assertEqual(data[data.group == -1].shape[0], 200)

    def test_split_data_stacking(self):
        data = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        # test split_data
        data = split_data_stacking(data, data.index >= 900, 3)
        self.assertEqual(data[data.group == 0].shape[0], 300)
        self.assertEqual(data[data.group == 1].shape[0], 300)
        self.assertEqual(data[data.group == 2].shape[0], 300)
        self.assertEqual(data[data.group == -1].shape[0], 100)
