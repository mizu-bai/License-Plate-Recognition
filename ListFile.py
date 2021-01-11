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
