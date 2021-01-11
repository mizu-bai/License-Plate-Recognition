import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Preprocess(object):
    """
    这个类用于对于图像进行预处理
    输入 input_img: string 类型, 车牌图片的路径
    输出 output_img: 三维 numpy 数组, 处理后的车牌图片
    类变量 self.src_img: 读入的车牌图片, 三维 numpy 数组
    类变量 self.rsz_img: 缩放后的车牌图片, 三维 numpy 数组
    """
    def __init__(self, input_img):
        self.src_img = cv.imread(input_img)
        self.rsz_img = cv.resize(self.src_img, (0, 0),
                                 fx=220 / self.src_img.shape[1],
                                 fy=70 / self.src_img.shape[0],
                                 interpolation=cv.INTER_NEAREST)

    def auto_invert_colors(self, input_img):
        """
        这个函数用来自动反色车牌, 使二值化后的字符为白色
        输入 input_img: 车牌图片, 三维 numpy 数组
        输出 output_img: 自动反色后的车牌图片, 三维 numpy 数组
        """
        std_img = input_img.copy()
        gray_img = cv.cvtColor(std_img, cv.COLOR_BGR2GRAY)
        _, bin_img = cv.threshold(gray_img, 0, 255,
                                  cv.THRESH_BINARY | cv.THRESH_OTSU)
        black_pixel = 0
        white_pixel = 0
        for y in range(bin_img.shape[0]):
            for x in range(bin_img.shape[1]):
                if bin_img[y, x] == 0:
                    black_pixel += 1
                else:
                    white_pixel += 1

        if black_pixel > white_pixel:
            return std_img
        else:
            ivt_img = input_img.copy()
            for y in range(ivt_img.shape[0]):
                for x in range(ivt_img.shape[1]):
                    ivt_img[y, x] = 255 - ivt_img[y, x]

            return ivt_img

    def process(self, input_img):
        """
        这个函数把车牌图像处理成二值图, 用于后续检测
        输入 input_img: 车牌图片, 三维 numpy 数组
        输出 output_img: 处理后的车牌图片, 三维 numpy 数组
        """
        std_img = input_img.copy()
        # Gaussian 模糊去噪
        blur_img = cv.GaussianBlur(std_img, (5, 5), 0)

        # Gamma 变换
        def adjust_gamma(image, gamma=1.0):
            inv_gamma = 1.0 / gamma
            table = []
            for i in range(256):
                table.append(((i / 255.0)**inv_gamma) * 255)
            table = np.array(table).astype("uint8")
            return cv.LUT(image, table)

        ehc_img = adjust_gamma(blur_img, gamma=2.8)

        return ehc_img


# from ListFile import get_file_list
# img_list = []
# get_file_list("Class I/", img_list)
# print(img_list)
# for img in img_list:
#     demo = Preprocess(img)
#     cv.imshow("rsz_img", demo.rsz_img)
#     ivt_img = demo.auto_invert_colors(demo.rsz_img)
#     # print(f"The Color of {img} is {color}")
#     cv.imshow("inv_img", ivt_img)

#     ehc_img = demo.process(ivt_img)
#     cv.imshow("ehc_img", ehc_img)
#     gray_img = cv.cvtColor(ehc_img, cv.COLOR_BGR2GRAY)
#     _, bin_img = cv.threshold(gray_img, 0, 255,
#                               cv.THRESH_BINARY | cv.THRESH_OTSU)
#     cv.imshow("bin_img", bin_img)

#     cv.waitKey(0)
#     cv.destroyAllWindows()