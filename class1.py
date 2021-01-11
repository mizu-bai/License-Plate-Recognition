import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from Preprocess import Preprocess
from SplitLicensePlate import SplitLicensePlate
from ListFile import get_file_list


def split_lp(input_img):
    """
    拆分字符
    """
    splitLicensePlate = SplitLicensePlate(input_img)
    # 获取 row_sum 和 col_sum
    row_sum, col_sum = splitLicensePlate.col_and_row_sum()
    bin_img = splitLicensePlate.bin_img
    bgr_bin_img = splitLicensePlate.gray2bgr(bin_img)
    # 分割字符
    color_chars, bin_chars = splitLicensePlate.split_chars(row_sum, col_sum)
    bgr_bin_chars = []
    for bin_char in bin_chars:
        bgr_bin_chars.append(splitLicensePlate.gray2bgr(bin_char))

    return bin_img, bgr_bin_img, row_sum, col_sum, color_chars, bin_chars, bgr_bin_chars


if __name__ == "__main__":
    imgs = []
    get_file_list("I类车牌", imgs)
    tmp_dir = "I类车牌预处理"
    res_dir = "I类车牌结果"
    # 创建临时目录
    if os.path.isdir(tmp_dir):
        print(f"\"{tmp_dir}\"目录已存在")
    else:
        os.makedirs(tmp_dir)
        print(f"创建\"{tmp_dir}\"目录保存图片")
    # 创建结果目录
    if os.path.isdir(res_dir):
        print(f"\"{res_dir}\"目录已存在")
    else:
        os.makedirs(res_dir)
        print(f"\"{res_dir}\"目录已存在")

    for img in imgs:
        preprocess = Preprocess(img)
        std_img = preprocess.auto_invert_colors(preprocess.rsz_img)
        img_name = img.split("/")[1].split(".")[0]
        # 保存结果
        cv.imwrite(f"{tmp_dir}/{img_name}.jpg", std_img)

        # 分割车牌
        bin_img, bgr_bin_img, row_sum, col_sum, color_chars, bin_chars, bgr_bin_chars = split_lp(
            f"{tmp_dir}/{img_name}.jpg")
        # cv.imshow("bin_img", bin_img)
        plt.figure(1)
        # row_sum
        plt.subplot(2, 2, 1)
        plt.barh(range(len(row_sum)), row_sum)
        plt.ylim(len(row_sum) - 1, 0)
        plt.title("row_sum")
        # col_sum
        plt.subplot(2, 2, 4)
        plt.bar(range(len(col_sum)), col_sum)
        plt.xlim(0, len(col_sum) - 1)
        plt.title("col_sum")
        # bin_img
        plt.subplot(2, 2, 2)
        plt.imshow(bgr_bin_img)
        plt.title("bin_img")

        # 分割字符
        plt.figure(2)
        # 绘图
        n = len(color_chars)
        if n != 7:
            print(f"分割{img}可能有误")
        else:
            print(f"分割{img}成功")
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(cv.cvtColor(color_chars[i], cv.COLOR_BGR2RGB))
            plt.subplot(2, n, i + n + 1)
            plt.imshow(bgr_bin_chars[i])

        # 显示
        plt.show()
