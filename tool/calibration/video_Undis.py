import cv2
import numpy as np
import os
import argparse


def list_files_and_folders(directory, t:str):
    #t:search type. ex:mp4
    path_list=[]
    # 使用 os.listdir 获取目录下所有文件和文件夹的列表
    items = os.listdir(directory)
    for item in items:
        # 利用 os.path.join 构建完整路径
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and full_path.split(".")[-1]==t:
            path_list.append(full_path)
            
        elif os.path.isdir(full_path):
            path_list = path_list+list_files_and_folders(full_path, t)
    return path_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Input_dir', default="F:\\data_0802-0803\\pollen_time\\0802", type=str,
                        help='input video path')
    global ARGV
    ARGV = parser.parse_args()
parse_args()




camera_matrix = np.load("camera_matrix_beeMain-down.npy")
dist_coeffs = np.load("dist_coeffs_beeMain-down.npy")

#Input_dir = "F:\pollne_video_0415"
Input_dir = ARGV.Input_dir
Output_dir= os.path.join(Input_dir,"undis")

target_paths = list_files_and_folders(Input_dir, t="mp4")

Output_dir=os.path.join(ARGV.Input_dir, "undist")
os.makedirs(Output_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

for video_path in target_paths:
    print(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(Output_dir, video_name+".mp4")
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (1440,  1080))
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
            out.write(undistorted_image)
        else:
            break

    out.release()




