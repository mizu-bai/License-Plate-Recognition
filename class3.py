import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from CutLicensePlate import CutLicensePlate
from SplitLicensePlate import SplitLicensePlate
from ListFile import get_file_list


def cut_lp(input_img):
    """
    从图中找出车牌
    """
    cutLicensePlate = CutLicensePlate(input_img)
    # 轮廓 list
    contours = cutLicensePlate.find_rectangle_contour()
    # 最合理的车牌轮廓
    rsn_rectangle = cutLicensePlate.screen_contours(contours)
    # 在输入图片上框出车牌, 并切割出车牌, 提取字符轮廓
    draw_img, cut_img = cutLicensePlate.cut_license_plate(rsn_rectangle)

    return draw_img, cut_img


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
    # 获取 III 类车牌列表
    imgs = []
    get_file_list("III类车牌", imgs)
    tmp_dir = "III类车牌预处理"
    res_dir = "III类车牌结果"
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
        try:
            draw_img, cut_img = cut_lp(img)
            print(f"在图片{img}中找到了车牌！")
            img_name = img.split("/")[1].split(".")[0]
            cut_img_name = img_name + ".jpg"
            draw_img_name = img_name + "范围.jpg"
            # 保存结果
            cv.imwrite(f"{tmp_dir}/{cut_img_name}", cut_img)
            cv.imwrite(f"{res_dir}/{cut_img_name}", cut_img)
            cv.imwrite(f"{res_dir}/{draw_img_name}", draw_img)
            cv.imshow("draw_img", draw_img)
            cv.imshow("cut_img", cut_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        except IndexError:
            print(f"在图片{img}中没找到车牌!")

    cut_imgs = []
    get_file_list(tmp_dir, cut_imgs)
    for cut_img in cut_imgs:
        bin_img, bgr_bin_img, row_sum, col_sum, color_chars, bin_chars, bgr_bin_chars = split_lp(
            cut_img)
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
        # 保存结果
        # plt.savefig(f"{res_dir}/{img_name}累加直方图.jpg")
        # 分割字符
        plt.figure(2)
        # 绘图
        n = len(color_chars)
        if n != 7:
            print("分割的图像可能有误")
        else:
            print(f"分割{img}成功")

        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(cv.cvtColor(color_chars[i], cv.COLOR_BGR2RGB))
            plt.subplot(2, n, i + n + 1)
            plt.imshow(bgr_bin_chars[i])

        # 保存结果
        # plt.savefig(f"{res_dir}/{img_name}分割结果.jpg")
        # 显示
        plt.show()
