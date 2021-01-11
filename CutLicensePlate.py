import cv2 as cv
import numpy as np


class CutLicensePlate(object):
    """
    这个类实现了车牌区域的切割
    读入车牌图片, 并对其大小缩放以标准化
    输入 input_img: string 类型, 车牌图片的路径
    类变量 self.src_img: 读入的车牌图片, 三维 numpy 数组
    类变量 self.rsz_img: 缩放后的车牌片, 三维 numpy 数组
    """
    def __init__(self, input_img):
        self.src_img = cv.imread(input_img)
        self.rsz_img = cv.resize(self.src_img, (0, 0),
                                 fx=720 / self.src_img.shape[1],
                                 fy=720 / self.src_img.shape[1],
                                 interpolation=cv.INTER_NEAREST)

    def find_rectangle_contour(self):
        """
        在图像中寻找可能包含车牌的矩形区域
        输出 contours: 一维 list, 其中元素为三维 numpy 数组
        """
        # 复制一份转换为标准大小的图片, 不改变输入的图片
        std_img = self.rsz_img.copy()

        # BGR 转 HSV 色彩空间
        hsv_img = cv.cvtColor(std_img, cv.COLOR_BGR2HSV)

        # 车牌蓝色范围
        lower_blue = np.array([100, 76, 55])
        upper_blue = np.array([125, 153, 221])
        blue_mask = cv.inRange(hsv_img, lower_blue, upper_blue)
        blue_img = cv.bitwise_and(std_img, std_img, mask=blue_mask)

        # 灰度化
        gray_img = cv.cvtColor(blue_img, cv.COLOR_BGR2GRAY)

        # Gaussian 模糊
        blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # Sobel 滤波
        sbl_img = cv.Sobel(blur_img, cv.CV_16S, 1, 0)
        sbl_img = cv.convertScaleAbs(sbl_img)

        # 二值化
        _, bin_img = cv.threshold(sbl_img, 0, 255,
                                  cv.THRESH_BINARY | cv.THRESH_OTSU)

        # 闭操作
        knl = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        cls_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, knl, iterations=1)

        # 中值滤波去噪
        fnl_img = cv.medianBlur(cls_img, 3)

        # 发现轮廓
        contours, _ = cv.findContours(fnl_img, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

        return contours

    def screen_contours(self, contours):
        """
        通过车牌长宽比例和面积从包含车牌的矩形区域中筛选车牌
        输入轮廓列表 contours: list, 其中元素为三维 numpy 数组
        返回最合理的车牌轮廓 rsn_contours: 三维 numpy 数组
        """
        # 复制一份转换为标准大小的图片, 不改变输入的图像
        std_img = self.rsz_img.copy()
        # 车牌规格
        # 我国车牌长 440 mm, 宽 140 mm, 长宽比为 3.14
        length = 440
        width = 140
        ratio = length / width
        area_min = 45**2 * ratio
        area_max = 250**2 * ratio
        error = 0.3
        ratio_min = ratio * (1 - error)
        ratio_max = ratio * (1 + error)

        # 矩形轮廓筛选
        # rectangles: 二维 list, 存储矩形范围
        rectangles = []
        for contour in contours:
            x = []
            y = []
            for point in contour:
                x.append(point[0][1])
                y.append(point[0][0])

            # rectangle: 一维 list, 其中元素为 int 类型, 存储了矩形轮廓的 x 和 y 坐标范围
            rectangle = [min(y), min(x), max(y), max(x)]

            # 通过长宽比例和面积筛选
            area = (rectangle[2] - rectangle[0]) * (rectangle[3] -
                                                    rectangle[1])
            ratio = (rectangle[2] - rectangle[0]) / (rectangle[3] -
                                                     rectangle[1])
            if area_min < area < area_max and ratio_min < ratio < ratio_max:
                rectangles.append(rectangle)

        # 通过颜色筛选
        # BGR 转 HSV 色彩空间
        hsv_img = cv.cvtColor(std_img, cv.COLOR_BGR2HSV)
        # 车牌蓝色范围
        lower_blue = np.array([100, 76, 55])
        upper_blue = np.array([125, 153, 221])
        # 遍历矩形轮廓
        rsn_rectangle = rectangles[0]
        blue_ratio = []
        for rectangle in rectangles:
            # 切割图片
            cut_img = hsv_img[rectangle[1]:rectangle[3],
                              rectangle[0]:rectangle[2]]
            height = cut_img.shape[0]
            width = cut_img.shape[1]
            blue_pixel = 0
            pixel_count = 0
            for row in range(height):
                for col in range(width):
                    pixel_count += 1
                    if lower_blue[0] < cut_img[row, col][0] < upper_blue[0] \
                        and lower_blue[1] < cut_img[row, col][1] < upper_blue[1]\
                            and lower_blue[2] < cut_img[row, col][2] < upper_blue[2]:
                        blue_pixel += 1

            blue_ratio.append(float(blue_pixel) / pixel_count)

        rsn_rectangle = rectangles[blue_ratio.index(max(blue_ratio))]

        return rsn_rectangle

    def cut_license_plate(self, rsn_rectangle):
        """
        在输入图片上框出车牌, 并切割出车牌, 提取字符轮廓
        输入 rsn_rectangle: 一维 list, 其中元素为 int 类型, 存储了矩形轮廓的 x 和 y 坐标范围
        输出 2 个图像, 即三维 numpy 数组, 分别为框出车牌的图像, 和车牌图像
        """
        # 复制一份转换为标准大小的图片
        draw_img = self.rsz_img.copy()
        # 框出车牌范围
        cv.rectangle(draw_img, (rsn_rectangle[0], rsn_rectangle[1]),
                     (rsn_rectangle[2], rsn_rectangle[3]), (0, 0, 255))

        # 切割出车牌
        cut_img = self.rsz_img[rsn_rectangle[1] + 5:rsn_rectangle[3] - 5,
                               rsn_rectangle[0] + 5:rsn_rectangle[2] - 5]
        cut_img = cv.resize(cut_img, (0, 0),
                            fx=220 / cut_img.shape[1],
                            fy=70 / cut_img.shape[0],
                            interpolation=cv.INTER_NEAREST)
        return draw_img, cut_img


# demo
# 读入图片
# def test(input_img):
#     demo = CutLicensePlate(input_img)
#     # 轮廓 list
#     contours = demo.find_rectangle_contour()
#     # 最合理的车牌轮廓
#     rsn_rectangle = demo.screen_contours(contours)
#     # 在输入图片上框出车牌, 并切割出车牌, 提取字符轮廓
#     draw_img, cut_img = demo.cut_license_plate(rsn_rectangle)
#     # 显示图片
#     cv.imshow("draw_img", draw_img)
#     cv.imshow("cut_img", cut_img)
#     # 图片保持显示
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     imgs = [
#         "./org_imgs/晋A00002.jpg",
#         "./org_imgs/粤A3Y347.jpg",
#         "./org_imgs/晋AAS999.jpg",
#         "./org_imgs/云AU7526.jpg",
#         "./org_imgs/晋C88888.jpg",
#         "./org_imgs/京K89779.jpg",
#         "./org_imgs/贵OA2803.jpg",
#     ]
#     for i in imgs:
#         try:
#             test(i)
#         except IndexError:
#             print("找不到车票呜呜呜")
