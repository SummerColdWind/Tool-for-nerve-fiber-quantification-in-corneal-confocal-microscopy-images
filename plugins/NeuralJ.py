import os
from collections import defaultdict

from process.processor import Processor
from utils.calculate import get_view_area_mm2, pixel_to_mm
from utils.common import extract_Chinese

import numpy as np
import pandas as pd

class NeuralJPlugins:
    def __init__(self):
        self.data = {
            'total': defaultdict(list),
            'primary': defaultdict(list),
            'primary_count': defaultdict(list),
            'secondary': defaultdict(list),
            'secondary_count': defaultdict(list),
            'tertiary': defaultdict(list),
        }

    def load_data(self, dir_path):
        """
        数据目录格式要求:
        root_dir
            Name
                Groups.csv
                Tracings.csv
                Vertices.csv
                Name.ndf
                Name.tif
        """
        for name in os.listdir(dir_path):
            true_name = extract_Chinese(name)
            try:
                tracings = pd.read_csv(os.path.join(dir_path, name, 'Tracings.csv'), encoding='utf-8')
            except UnicodeDecodeError:
                tracings = pd.read_csv(os.path.join(dir_path, name, 'Tracings.csv'), encoding='GBK')


            lengths = tracings.groupby('Type')['Length [pix]'].sum()
            try:
                self.data['primary'][true_name].append(lengths['Primary'])
                self.data['primary_count'][true_name].append(np.sum(tracings['Type'] == 'Primary'))
            except KeyError:
                self.data['primary'][true_name].append(0)
                self.data['primary_count'][true_name].append(0)

            try:
                self.data['secondary'][true_name].append(lengths['Secondary'])
                self.data['secondary_count'][true_name].append(np.sum(tracings['Type'] == 'Secondary'))
            except KeyError:
                self.data['secondary'][true_name].append(0)
                self.data['secondary_count'][true_name].append(0)

            try:
                self.data['tertiary'][true_name].append(lengths['Tertiary'])
            except KeyError:
                self.data['tertiary'][true_name].append(0)

            self.data['total'][true_name].append(sum(lengths[k] for k in lengths.index))

    def process_data(self):
        data = {
            'total': {},
            'primary': {},
            'primary_count': {},
            'secondary': {},
            'secondary_count': {},
            'tertiary': {},
        }
        for col, raw in self.data.items():
            for name, item in raw.items():
                data[col][name] = np.mean(item)

        data = pd.DataFrame.from_dict(data)
        data['CNFL'] = data['total'].map(lambda x: pixel_to_mm(x / get_view_area_mm2()))
        data['CNFD'] = data['primary_count'].map(lambda x: x / get_view_area_mm2())
        data['CNBD'] = data['secondary_count'].map(lambda x: x / get_view_area_mm2())
        self.data = data

