import os
import itertools
from collections import defaultdict
import logging
from typing import Literal

from utils.common import show_image, load_image, save_image, get_canvas, split, extract_Chinese
from utils.calculate import *
from utils.onnx.detector import NerveSegmenter
from process.skeleton import get_skeleton
from process.point import get_points
from process.instance import get_instance
from process.graph import get_trunk
from process.draw import draw_digit_image, draw_result_image
from process.concat import get_nerve

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CALCULATE_TYPE = Literal['dispatch', 'all']

class Processor:
    def __init__(self):
        self.segmenter = None
        self.image = None
        self.images = {
            'binary': None,
            'result': None,
        }

        self.model_path = None

        self.segments = []
        self.nodes = []
        self.nerves = []


    def load_model(self, path=None):
        path = self.model_path if path is None else path
        self.segmenter = NerveSegmenter(path)


    def load_image(self, path):
        image = load_image(path)
        image = image[:384, :384]
        self.image = image
        self.segments = []
        self.nodes = []
        self.nerves = []

    def process(self):
        binary = self.segmenter(self.image)
        self.images['binary'] = binary
        skeleton = get_skeleton(binary)
        blocks, _ = split(skeleton, split_skeleton=True)
        for block in blocks:
            points, end_points = get_points(block)
            instance = get_instance(binary, block, points, end_points)
            segments, nodes = instance
            self.segments.extend(segments)
            self.nodes.extend(nodes)

            trunks = get_trunk(*instance)
            if trunks:
                for sis, nis in trunks:
                    for si in sis:
                        segments[si].class_segment = 'main'
                    self.nerves.append(get_nerve([segments[si] for si in sis], [nodes[ni] for ni in nis]))

        # 重新编号
        for i, segment in enumerate(self.segments):
            segment.index = i

        for i, node in enumerate(self.nodes):
            node.index = i

        self.images['result'] = draw_result_image(self.segments, self.nodes, self.image)


    def set_model_path(self, path):
        self.model_path = path

    def calculate(self, dir_path, image_suffix='tif', mode: CALCULATE_TYPE = 'dispatch'):
        """
        参数计算:
            (1)角膜神经纤维密度(CNFD)，神经总数/ mm2;
            (2)角膜神经分支密度(CNBD)，主要神经干分支数/mm2;
            (3)角膜神经纤维长度(CNFL)，所有神经纤维及分支的总长度(mm/mm2);
            (4)角膜神经总分支密度(CTBD)，所有分支总数/mm2;
            (5)角膜神经纤维面积(CNFA)，总神经纤维面积(mm2 / mm2);
            (6)角膜神经纤维宽度(CNFW)，平均神经纤维宽度(mm/mm2);
            (7)角膜神经纤维弯曲度(CNFT)(弯曲系数[TC])。它代表了从连接各主要神经纤维末端的直线开始的扭曲程度。
        mode:
            dispatch: 假设每个子文件夹内存放了该受试者的所有图片
            all: 假设父文件夹下存放了所有图片，并且文件名中含有中文名字
        """
        params = defaultdict(lambda: defaultdict(list))
        params_mean = defaultdict(dict)

        if mode == 'dispatch':
            for person in tqdm(os.listdir(dir_path)):
                try:
                    name = extract_Chinese(person)
                    for image in os.listdir(os.path.join(dir_path, person)):
                        if image.endswith(f'.{image_suffix}'):
                            image_path = os.path.join(dir_path, person, image)
                            self.load_image(image_path)
                            self.process()

                            params[name]['CNFL'].append(get_CNFL(self))
                            params[name]['CNFD'].append(get_CNFD(self))
                            params[name]['CNBD'].append(get_CNBD(self))
                except Exception as e:
                    logging.error(f'When process `{person}`: ')
                    logging.error(e)
        elif mode == 'all':
            for image in tqdm(os.listdir(dir_path)):
                if image.endswith(f'.{image_suffix}'):
                    try:
                        name = extract_Chinese(image)
                        image_path = os.path.join(dir_path, image)
                        self.load_image(image_path)
                        self.process()

                        params[name]['CNFL'].append(get_CNFL(self))
                        params[name]['CNFD'].append(get_CNFD(self))
                        params[name]['CNBD'].append(get_CNBD(self))
                    except Exception as e:
                        logging.error(f'When process `{image}`: ')
                        logging.error(e)


        for name, pm in params.items():
            params_mean[name]['CNFL'] = np.mean(params[name]['CNFL'])
            params_mean[name]['CNFD'] = np.mean(params[name]['CNFD'])
            params_mean[name]['CNBD'] = np.mean(params[name]['CNBD'])

        df = pd.DataFrame.from_dict(params_mean, orient='index')
        return df

