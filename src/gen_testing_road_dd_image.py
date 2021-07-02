import os
from PIL import Image
import numpy as np
import cv2 as cv
import torch


class GenTestingRoadDDImage:
    def __init__(self, dem_map_path, road_perpendicular_line_path):
        self.dem_map_path = dem_map_path
        self.road_perpendicular_line_path = road_perpendicular_line_path

        self.img = None
        return

    def work(self):
        dem_name_set = os.listdir(self.dem_map_path)
        for dem_name in dem_name_set:
            # rpl image path to write
            ori_split_name = dem_name.split('_')
            if len(ori_split_name) > 1:
                ori_name = ori_split_name[0] + '.png'
            else:
                ori_name = ori_split_name[0]
            img_path = os.path.join(self.road_perpendicular_line_path, ori_name)

            img = self.gen_testing_dd_img()

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.imwrite(img_path, img)
        return

    def gen_testing_dd_img(self):
        if self.img is None:
            dem_name_set = os.listdir(self.dem_map_path)
            dem_name = dem_name_set[0]
            img_path = os.path.join(self.dem_map_path, dem_name)
            pil_img = Image.open(img_path).convert('RGB')
            cv_img = np.array(pil_img)
            img = cv_img * 0
            self.img = self.gen_horizontal_dd_img(img)
        return self.img

    def gen_horizontal_dd_img(self, img):
        step = 1
        height = img.shape[0]
        width = img.shape[1]
        for h in range(0, height, step):
            for w in range(0, width, step):
                if h % 3 == 0:
                    val = (255 * 0.7071067811865476, 255 * 0.7071067811865476, 0)
                    img[h, w] = val
        return img
