# License Plate Recognition

是笔者本学期数据科学课程大作业，也是笔者 partner 数字图像处理课程的大作业。

## 干嘛用的

这份代码可以从车牌图像中分割出各个字符，之后拿去做识别之类的（模板匹配，CNN 什么的都可以）。

测试数据分为三类，I 类和 II 类是裁剪过的车牌，但 II 类有较多干扰因素，如噪点较多，清晰度差，光照不均匀等，III 类是包含车牌的图片，需要寻找车牌，再进行字符分割。

## 看看效果

### I 类（II 类和这个差不多）

原始图像

![鲁JD9309](https://github.com/mizu-bai/License-Plate-Recognition/raw/main/Demo/鲁JD9309.jpg)

分割结果

![鲁JD9309分割结果](https://github.com/mizu-bai/License-Plate-Recognition/raw/main/Demo/鲁JD9309分割结果.png)

### III 类

车牌范围

![粤A3Y347车牌范围](https://github.com/mizu-bai/License-Plate-Recognition/raw/main/Demo/粤A3Y347车牌范围.png)

分割结果

![粤A3Y347分割结果](https://github.com/mizu-bai/License-Plate-Recognition/raw/main/Demo/粤A3Y347分割结果.png)

## 运行环境

笔者的运行环境如下：

* OS: macOS Catalina 10.15.7 (19H114)
* Python: 3.7.9
* OpenCV: 4.4.0.46
* NumPy: 1.18.5 

## 运行方式

```shell
$ python -u class*.py # 输入数据类别对应的数字
```

## 备注

这个东西做得比较粗糙，有挺多地方可以再改进的，大概列一下。

1. 没有对车牌图像做仿射变换，很多字符识别出来是斜的；
2. 对 I 类和 II 类懒得再切割一下边缘了，对后续的投影和分割有一定影响；
3. III 类车牌需要对蓝色部分进行筛选，使用的蓝色 HSV 范围可以再调整一下，提高适用性。
