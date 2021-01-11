import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class SplitLicensePlate(object):
    """
    这个类实现了车牌中的字符的分割
    读入车牌区域图片, 将其中字符分割为单个图片
    输入 input_img: string 类型, 车牌图片的路径
    类变量 self.src_img: 读入的车牌图片, 三维 numpy 数组
    类变量 self.rsz_img: 缩放后的车牌图片, 三维 numpy 数组
    类变量 self.bin_img: 二值化的车牌图片, 三维 numpy 数组
        调用类方法 col_and_row_sum() 后才被赋值
    """
    def __init__(self, input_img):
        self.src_img = cv.imread(input_img)
        self.rsz_img = cv.resize(self.src_img, (0, 0),
                                 fx=220 / self.src_img.shape[1],
                                 fy=70 / self.src_img.shape[0],
                                 interpolation=cv.INTER_NEAREST)

    def col_and_row_sum(self):
        """
        这个函数把二值图片从 row 和 col 两个方向上累加, 用来寻找字符边界
        输入 无
        输出 row_sum, col_sum: 一维 list, 列表中元素为 int, 存储从两个方向上累加的值
        """
        # 先 copy 一份车牌图像
        std_img = self.rsz_img.copy()
        # 灰度化
        gray_img = cv.cvtColor(std_img, cv.COLOR_BGR2GRAY)
        # 二值化
        _, bin_img = cv.threshold(gray_img, 0, 255,
                                  cv.THRESH_BINARY + cv.THRESH_OTSU)
        self.bin_img = bin_img
        # 获取车牌宽高
        height, width = bin_img.shape
        # 累计直方图
        # 按 row 方向累加
        row_sum = []
        for y in range(height):
            tmp = 0
            for x in range(width):
                if bin_img[y, x] != 0:
                    tmp += 1

            row_sum.append(tmp)

        # 按 col 方向累加
        col_sum = []
        for x in range(width):
            tmp = 0
            for y in range(height):
                if bin_img[y, x] != 0:
                    tmp += 1

            col_sum.append(tmp)

        return row_sum, col_sum

    def split_chars(self, row_sum, col_sum):
        """
        这个函数按这个函数按 row_sum 和 col_sum 找出各字符的界限, 再切割出各个字符
        输入 row_sum 和 col_sum, 均为一维列表, 列表中元素为 int, 为两个方向上的累加直方图
        输出 color_chars 和 bin_chars, 均为一维列表, 列表中元素分别为三维和二维 numpy 数组
        """
        # 划一个下界
        row_sum = [
            0 if val < (min(row_sum) + np.mean(row_sum)) / 3 else val
            for val in row_sum
        ]
        col_sum = [
            0 if val < (min(col_sum) + np.mean(col_sum)) / 3.8 else val
            for val in col_sum
        ]
        # 判断边界
        # row
        row_limits = []
        in_char = False
        start = 0
        end = 0
        for i in range(len(row_sum)):
            if not in_char and row_sum[i] != 0:
                in_char = True
                start = i
            elif in_char and row_sum[i] == 0:
                end = i
                in_char = False
                row_limits.append([start, end])

        if start > end and start != len(row_sum) - 1:
            row_limits.append([start, len(row_sum) - 1])

        row_limit = []
        for i in range(len(row_limits)):
            if i == 0:
                row_limit = row_limits[0]
            else:
                cur_limit = row_limits[i]
                if cur_limit[1] - cur_limit[0] > row_limit[1] - row_limit[0]:
                    row_limit = cur_limit

        # col
        col_limits = []
        in_char = False
        start = 0
        end = 0
        for i in range(len(col_sum)):
            if not in_char and col_sum[i] != 0:
                in_char = True
                start = i
            elif in_char and col_sum[i] == 0:
                end = i
                in_char = False
                col_limits.append([start, end])

        if start > end and start != len(col_sum) - 1:
            col_limits.append([start, len(col_sum) - 1])

        # 按宽度筛选
        new_col_limits = []
        for col_limit in col_limits:
            if 15 < col_limit[1] - col_limit[0]:
                new_col_limits.append(col_limit)
            else:
                curr_area = self.bin_img[row_limit[0]:row_limit[1],
                                         col_limit[0]:col_limit[1]]
                white_pixel = 0
                pixel_count = 0
                for y in range(curr_area.shape[0]):
                    for x in range(curr_area.shape[1]):
                        pixel_count += 1
                        if curr_area[y, x] != 0:
                            white_pixel += 1
                if float(white_pixel) / pixel_count > 0.45:
                    # 最后一个数字是 1 的时候和右边界的距离应大于半个字符宽度
                    if len(col_sum) - 1 - col_limit[1] < 7:
                        continue

                    # 扩展 1 的区域
                    col_limit[0] = 2 * col_limit[0] - col_limit[1]
                    col_limit[1] = 2 * col_limit[1] - col_limit[0]
                    
                    # 考虑左边界噪音
                    if col_limit[0] < 0:
                        continue
                    else:
                        new_col_limits.append(col_limit)

        col_limits = new_col_limits

        # 截取图像
        color_chars = []
        bin_chars = []
        for col_limit in col_limits:
            color_char = self.rsz_img[row_limit[0]:row_limit[1],
                                      col_limit[0]:col_limit[1]]
            color_char = cv.resize(color_char, (0, 0),
                                   fx=22 / color_char.shape[1],
                                   fy=45 / color_char.shape[0],
                                   interpolation=cv.INTER_NEAREST)
            color_chars.append(color_char)

            bin_char = self.bin_img[row_limit[0]:row_limit[1],
                                    col_limit[0]:col_limit[1]]
            bin_char = cv.resize(bin_char, (0, 0),
                                 fx=22 / bin_char.shape[1],
                                 fy=45 / bin_char.shape[0],
                                 interpolation=cv.INTER_NEAREST)
            bin_chars.append(bin_char)

        return color_chars, bin_chars

    def gray2bgr(self, gray_img):
        """
        这个函数把灰度图转 bgr 色彩空间
        输入 gray_img: 二维 numpy 数组
        输出 rgb_img: 三维 numpy 数组
        """
        bgr_img = np.ones([gray_img.shape[0], gray_img.shape[1], 3],
                          dtype=np.uint8)
        for y in range(gray_img.shape[0]):
            for x in range(gray_img.shape[1]):
                bgr_img[y, x] = np.asarray(gray_img[y, x] * 3, dtype=np.uint8)

        return bgr_img


# def test(input_img):
#     demo = SplitLicensePlate(input_img)
#     # 获取 row_sum 和 col_sum
#     row_sum, col_sum = demo.col_and_row_sum()
#     # 绘图
#     plt.figure(1)
#     # row_sum
#     plt.subplot(2, 2, 1)
#     plt.barh(range(len(row_sum)), row_sum)
#     plt.ylim(len(row_sum) - 1, 0)
#     plt.title("row_sum")
#     # col_sum
#     plt.subplot(2, 2, 4)
#     plt.bar(range(len(col_sum)), col_sum)
#     plt.xlim(0, len(col_sum) - 1)
#     plt.title("col_sum")
#     # bin_img
#     plt.subplot(2, 2, 2)
#     plt.imshow(demo.gray2bgr(demo.bin_img))
#     plt.title("bin_img")
#     # # 分割字符
#     color_chars, bin_chars = demo.split_chars(row_sum, col_sum)
#     plt.figure(2)
#     # 绘图
#     n = len(color_chars)
#     if n != 7:
#         print("分割的图像可能有误")

#     for i in range(n):
#         plt.subplot(2, n, i + 1)
#         plt.imshow(cv.cvtColor(color_chars[i], cv.COLOR_BGR2RGB))
#         plt.subplot(2, n, i + n + 1)
#         plt.imshow(demo.gray2bgr(bin_chars[i]))

#     # 图片保持显示
#     plt.show()
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     # 读入图片
#     imgs = [
#         "./test_imgs/晋A00002.png", "./test_imgs/晋AAS999.png",
#         "./test_imgs/粤A3Y347.png", "./test_imgs/贵OA2803.png",
#         "./II类车牌/晋F21696.jpg"
#     ]
#     for i in imgs:
#         test(i)
