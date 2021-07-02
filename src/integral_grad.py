import os
from PIL import Image
import numpy as np
import cv2 as cv
import torch


class IntegralGrad:
    def __init__(self, dem_map_name):
        self.dem_map_name = dem_map_name

        self.img = None
        return

    def work(self):
        pil_img = Image.open(self.dem_map_name).convert('RGB')
        ori_img = np.array(pil_img)
        ori_img = ori_img[:, :, 0:1]

        pad_ori_img = np.pad(ori_img, ((0, 1), (0, 1), (0, 0)), 'edge')
        grad_mtx = self.get_gradien(pad_ori_img)
        grad_img = self.to_grad_img(grad_mtx)

        integrated_mtx = self.integral_grad_mtx_path_diagonal(ori_img, grad_mtx)
        integrated_img = self.to_integrated_img(integrated_mtx)

        diff = np.sum(np.abs(ori_img.astype('float') - integrated_img.astype('float')))
        print(diff)

        cv.imshow('ori_img', ori_img)
        cv.imshow('grad_img', grad_img)
        cv.imshow('integrated_img', integrated_img)
        cv.waitKey()
        return

    def grad_merge(self, grad_x_mtx, grad_y_mtx):
        grad_mtx = torch.cat((grad_x_mtx, grad_y_mtx), dim=2)
        return grad_mtx

    def grad_split(self, grad_mtx):
        grad_x_mtx = grad_mtx[:, :, 0:1]
        grad_y_mtx = grad_mtx[:, :, 1:2]
        return grad_x_mtx, grad_y_mtx

    def get_gradien(self, ori_img):
        ori_mtx = torch.tensor(ori_img).float()
        grad_x_mtx = ori_mtx[1:, :-1, :] - ori_mtx[:-1, :-1, :]
        grad_y_mtx = ori_mtx[:-1, 1:, :] - ori_mtx[:-1, :-1, :]
        return self.grad_merge(grad_x_mtx, grad_y_mtx)

    def to_grad_img(self, grad_mtx):
        grad_x_mtx, grad_y_mtx = self.grad_split(grad_mtx)
        grad_img_mtx = torch.sqrt(torch.pow(grad_x_mtx, 2) + torch.pow(grad_y_mtx, 2))
        grad_img = 50 * grad_img_mtx.type(torch.uint8).numpy()
        return grad_img

    def integral_grad_mtx_path_x2y(self, ori_img, grad_mtx):
        integrated_mtx = 0 * torch.tensor(ori_img).float()
        grad_x_mtx, grad_y_mtx = self.grad_split(grad_mtx)

        width = integrated_mtx.shape[0]
        height = integrated_mtx.shape[1]
        for x in range(0, width):
            for y in range(0, height):
                if x == 0 and y == 0:
                    integrated_mtx[x, y, 0] = 0
                elif y == 0:
                    integrated_mtx[x, y, 0] = integrated_mtx[x - 1, y, 0] + grad_x_mtx[x - 1, y, 0]
                else:
                    integrated_mtx[x, y, 0] = integrated_mtx[x, y - 1, 0] + grad_y_mtx[x, y - 1, 0]
        return integrated_mtx

    def integral_grad_mtx_path_y2x(self, ori_img, grad_mtx):
        integrated_mtx = 0 * torch.tensor(ori_img).float()
        grad_x_mtx, grad_y_mtx = self.grad_split(grad_mtx)

        width = integrated_mtx.shape[0]
        height = integrated_mtx.shape[1]
        for y in range(0, height):
            for x in range(0, width):
                if x == 0 and y == 0:
                    integrated_mtx[x, y, 0] = 0
                elif x == 0:
                    integrated_mtx[x, y, 0] = integrated_mtx[x, y - 1, 0] + grad_y_mtx[x, y - 1, 0]
                else:
                    integrated_mtx[x, y, 0] = integrated_mtx[x - 1, y, 0] + grad_x_mtx[x - 1, y, 0]
        return integrated_mtx

    def integral_grad_mtx_path_diagonal(self, ori_img, grad_mtx):
        integrated_mtx = 0 * torch.tensor(ori_img).float()
        grad_x_mtx, grad_y_mtx = self.grad_split(grad_mtx)

        x_step = 'x'
        y_step = 'y'
        width = integrated_mtx.shape[0]
        height = integrated_mtx.shape[1]
        for y in range(0, height):
            for x in range(0, width):
                if y % 8 == 0 and x == 0:
                    print(x, y)
                if x == 0 and y == 0:
                    integrated_mtx[x, y, 0] = 0
                else:
                    step_array = self.get_step_array(x, y, x_step, y_step)
                    sub_x = 0
                    sub_y = 0
                    val = integrated_mtx[sub_x, sub_y, 0].item()
                    assert len(step_array) == x + y
                    for step in step_array:
                        if step == x_step:
                            val += grad_x_mtx[sub_x, sub_y, 0]
                            sub_x += 1
                        else:
                            val += grad_y_mtx[sub_x, sub_y, 0]
                            sub_y += 1
                    integrated_mtx[x, y, 0] = val
        return integrated_mtx

    def get_step_array(self, x, y, x_step='x', y_step='y'):
        step_array = []
        x_idx = 0
        if x > 0:
            step_array.append(x_step)
        for i in range(0, y):
            y_idx = i * (x / float(y))
            y_idx = int(y_idx)
            while y_idx > x_idx:
                x_idx += 1
                step_array.append(x_step)
            if y_idx == x_idx:
                step_array.append(y_step)
        for i in range(0, x - (x_idx + 1)):
            step_array.append(x_step)
        return step_array

    def to_integrated_img(self, integrated_mtx):
        min_val = torch.min(integrated_mtx)
        if min_val < 0:
            integrated_mtx -= min_val
        return integrated_mtx.type(torch.uint8).numpy()
