from cut_license_plate import *
import os


def get_file_list(dir, file_list, ext=None):
    """
    获取目录 dir 下所有文件
    """
    cur_dir = dir
    if os.path.isfile(dir):
        if ext is None:
            file_list.append(dir)
        else:
            if ext in dir[-3:]:
                file_list.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            cur_dir = os.path.join(dir, s)
            get_file_list(cur_dir, file_list, ext)

    return file_list


if __name__ == "__main__":

    img_folder = "./demo/"
    img_list = get_file_list(img_folder, [], "jpg")
    print(f"本次执行检索到 {str(len(img_list))} 张图像")

    if "result" in os.listdir("./"):
        print("result 目录已存在")
    else:
        os.makedirs("./result")
        print("创建 result 目录保存图片")

    license_list = []

    i = 1
    for img in img_list:
        print(f"正在识别第 {str(i)} 张图片")
        cut_img_list = cut_license_plate(img)
        for cut_img in cut_img_list:
            cv.imwrite(f"./result/{i - 1}.jpg", cut_img)
        license_list.append(cut_img)
        i += 1
