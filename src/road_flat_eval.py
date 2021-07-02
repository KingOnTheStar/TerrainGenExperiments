import os
from PIL import Image
import numpy as np
import cv2 as cv
import torch


class RoadFlatEval:
    def __init__(self, dem_map_path, road_perpendicular_line_path):
        self.dem_map_path = dem_map_path
        self.road_perpendicular_line_path = road_perpendicular_line_path
        return

    def eval(self, show_img=False):
        dem_name_set = os.listdir(self.dem_map_path)
        rpl_name_set = os.listdir(self.road_perpendicular_line_path)
        if len(dem_name_set) != len(rpl_name_set):
            print(f'len(dem_name_set) != len(rpl_name_set) abort')

        for dem_name in dem_name_set:
            # Read DEM image
            img_path = os.path.join(self.dem_map_path, dem_name)
            pil_img = Image.open(img_path).convert('RGB')
            cv_img = np.array(pil_img)

            tor_img = torch.tensor(cv_img).float()
            tor_img = torch.mean(tor_img, dim=-1, keepdim=True)
            pixel_difh = tor_img[1:, :-1, :] - tor_img[:-1, :-1, :]
            pixel_difw = tor_img[:-1, 1:, :] - tor_img[:-1, :-1, :]

            pixel_diff = torch.cat((pixel_difw, pixel_difh), -1)

            # Read rpl image
            ori_split_name = dem_name.split('_')
            if len(ori_split_name) > 1:
                ori_name = ori_split_name[0] + '.png'
            else:
                ori_name = ori_split_name[0]
            img_path = os.path.join(self.road_perpendicular_line_path, ori_name)
            pil_img = Image.open(img_path).convert('RGB')
            cv_img = np.array(pil_img)

            sign_mtx = cv_img[:-1, :-1, -1]
            tor_img = torch.tensor(cv_img).float()
            tor_img[:-1, :-1, 0][sign_mtx & 1 == 1] *= -1
            tor_img[:-1, :-1, 1][sign_mtx & (1 << 1) == (1 << 1)] *= -1

            directional_deri_img = tor_img[:-1, :-1, :-1].div(255)
            dd_loss_mat = pixel_diff * directional_deri_img

            # directional_deri_img test
            rdx = directional_deri_img[:, :, :1]
            rdy = directional_deri_img[:, :, 1:]
            tor_img = torch.sqrt(torch.pow(rdx, 2) + torch.pow(rdy, 2))
            #tor_img[tor_img != 0] += -1
            sum_num = torch.sum(directional_deri_img)
            max_num = torch.max(tor_img)
            min_num = torch.min(tor_img)

            # Get final image
            dd_img = dd_loss_mat
            ddx_img = dd_img[:, :, :1]
            ddy_img = dd_img[:, :, 1:]
            tor_img = torch.sqrt(torch.pow(ddx_img, 2) + torch.pow(ddy_img, 2))

            # Calculate score
            total_grad = torch.sum(tor_img)
            tmp = tor_img.clone()
            tmp[tor_img > 0] = 1
            total_num = torch.sum(tmp)
            if total_num == 0:
                mean_grad = torch.tensor(0.0)
            else:
                mean_grad = total_grad / total_num
            print(mean_grad.item())


            # Show image
            if show_img:
                tor_img = 100 * tor_img
                cv_img = tor_img.type(torch.uint8).numpy()
                cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
                cv.imshow('cv_img', cv_img)
                cv.waitKey()
        return
