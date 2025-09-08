import numpy as np
import os
import cv2

target_folder = ""
save_folder = f"ppSeg/{target_folder}"

# 將檔案夾中想要的檔案格式的檔案列出
def list_files_and_folders(directory:str, t:str):
    # t: search type ex: mp4
    path_list = []

    files = os.listdir(directory)
    for file in files:
        # 利用 os.path.join 建立完整的路徑
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path) and full_path.split(".")[-1]==t:
            path_list.append(full_path)
        
        elif os.path.isdir(full_path):
            path_list = path_list + list_files_and_folders(full_path, t)
    return path_list

def crop_and_save(img_path:str, img_label:str, save_folder:str, class_num:str):
    img = cv2.imread(img_path)
    file_name = img_path.split("/")[-1][:-4] # [:-4]是找最後面四個字，如img123.png就會提出.png
