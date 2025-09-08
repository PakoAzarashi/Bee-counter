import cv2
import os 
import numpy
from moviepy.editor import *

def list_files_and_folders(directory:str, t:str):
    #t:search type. ex:.mp4
    if t == "mp4":
        t = [".mp4"]
    path_list=[]
    # 使用 os.listdir 获取目录下所有文件和文件夹的列表
    items = os.listdir(directory)
    for item in items:
        # 利用 os.path.join 构建完整路径
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and os.path.splitext(full_path)[-1] in t:
            path_list.append(full_path)
            
        elif os.path.isdir(full_path):
            path_list = path_list+list_files_and_folders(full_path, t)
    return path_list

def get_duration(video_path):

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = length / fps
    print(f"{duration} second")
    cap.release()

    return duration

video_dir = "F:\\pako_file\\tracking_datasets\\validation_data\\新增資料夾"
target_dir = "F:\\pako_file\\tracking_datasets\\validation_data\\validation_undist_crop"

path_list = list_files_and_folders(video_dir, "mp4")

for path in path_list:
    
    filename = os.path.basename(path).split(".")[0]

    video = VideoFileClip(path)  # 讀取影片

    output = video.subclip(t_start=0,t_end=(1,0))                # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_1.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(1,0),t_end=(2,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_2.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(2,0),t_end=(3,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_3.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(3,0),t_end=(4,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_4.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(4,0),t_end=(5,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_5.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(5,0),t_end=(6,0))                # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_6.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(6,0),t_end=(7,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_7.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(7,0),t_end=(8,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_8.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(8,0),t_end=(9,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_9.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(9,0),t_end=(10,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_10.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(10,0),t_end=(11,0))                # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_11.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(11,0),t_end=(12,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_12.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(12,0),t_end=(13,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_13.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(13,0),t_end=(14,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_14.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(14,0),t_end=(15,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_15.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(15,0),t_end=(16,0))                # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_16.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(16,0),t_end=(17,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_17.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(17,0),t_end=(18,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_18.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(18,0),t_end=(19,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_19.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(19,0),t_end=(20,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_20.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(20,0),t_end=(21,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_21.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(21,0),t_end=(22,0))                # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_22.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(22,0),t_end=(23,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_23.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(23,0),t_end=(24,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_24.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(24,0),t_end=(25,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_25.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(25,0),t_end=(26,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_26.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(26,0),t_end=(27,0))            # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_27.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(27,0),t_end=(28,0))            # 剪輯後10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_28.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(28,0),t_end=(29,0))            # 剪輯前10分鐘影片 ( 單位秒 ) 
    output.write_videofile(os.path.join(target_dir, f"{filename}_29.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    output = video.subclip(t_start=(29,0),t_end=(30,0))                        # 剪輯中10分鐘影片 ( 單位秒 )
    output.write_videofile(os.path.join(target_dir, f"{filename}_30.mp4"),temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    
    # 輸出影片，注意後方需要加上參數，不然會沒有聲音
    print('ok')