import cv2 as cv
import numpy as np


def cut_license_plate(input_img):
    """
    input_img: 图片文件名, string 类型
    本函数可以将从图片中分割出车牌轮廓, 并返回车牌轮廓的列表

    """
    # 读入图片
    src_img = cv.imread(input_img)
    # cv.imshow("src_img", src_img)

    # 转 hsv 寻找蓝色来二值化图像
    hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    # 蓝色范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    blue_img = cv.inRange(hsv_img, lower_blue, upper_blue)
    # cv.imshow("blue_img", blue_img)

    # 高斯模糊
    blur_img = cv.GaussianBlur(blue_img, (5, 5), 0)
    # cv.imshow("blur_img", blur_img)

    # Sobel 滤波
    sbl_img = cv.Sobel(blur_img, cv.CV_16S, 1, 0)
    sbl_img = cv.convertScaleAbs(sbl_img)
    # cv.imshow("sbl_img", sbl_img)

    # 二值化
    _, bin_img = cv.threshold(sbl_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("bin_img", bin_img)

    # 闭操作
    knl = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    cls_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, knl, iterations=1)
    # cv.imshow("cls_img", cls_img)

    # 发现轮廓
    contours, hierarchy = cv.findContours(
        cls_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # print(contours)

    # 我国车牌长 440 mm, 宽 140 mm, 比例为 3.14
    length = 440
    width = 140
    ratio = length / width
    area_min = 50 ** 2 * ratio
    area_max = 125 ** 2 * ratio
    error = 0.5
    ratio_min = ratio * (1 - error)
    ratio_max = ratio * (1 + error)

    # 矩形轮廓筛选
    rectangles = []
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x)]
        area = (max(y) - min(y)) * (max(x) - min(x))
        if area_min < area < area_max:
            ratio = (max(y) - min(y)) / (max(x) - min(x))
            if ratio_min < ratio < ratio_max:
                rectangles.append(r)

    # print(rectangles)

    # 在原图上框出车牌
    draw_img = src_img.copy()
    for rectangle in rectangles:
        cv.rectangle(
            draw_img,
            (rectangle[0], rectangle[1]),
            (rectangle[2], rectangle[3]),
            (0, 0, 255),
        )

    cv.imshow("draw_img", draw_img)

    # 根据矩形轮廓切割出车牌
    cut_img_list = []
    for rectangle in rectangles:
        cut_img = src_img[
            rectangle[1] + 15 : rectangle[3] - 15, rectangle[0] + 5 : rectangle[2] - 5
        ]
        # cv.imshow("cut_img", cut_img)
        cut_img_list.append(cut_img)

    # 提取车牌字符轮廓
    char_img_list = []
    for cut_img in cut_img_list:
        # 灰度化
        gray_img = cv.cvtColor(cut_img, cv.COLOR_BGR2GRAY)
        # Canny 边缘检测
        edge_img = cv.Canny(gray_img, 200, 200)
        cv.imshow("edge_img", edge_img)
        char_img = []
        char_img_list.append(edge_img)

    # 图片保持显示
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 返回字符轮廓列表
    return char_img_list
