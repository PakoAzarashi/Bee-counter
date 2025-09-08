#不畫圖，只技術，寫excel
import time
import cv2
import numpy as np
import torch
import os
from torchvision.transforms import ToPILImage, Resize
import time
from ultralytics import YOLO
import pandas as pd

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients import init_trainer
from super_gradients.training.utils.distributed_training_utils import setup_device


from collections import defaultdict
from superG.proccess import pre_proccess2, post_proccess, post_proccess_mutiSize

from utils.bee_tracking import bee_tracking

import argparse

ARGV: None #all arg
VIDEO_PATH: str
RESOULT_PATH: str
MODELs_PATH: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default="demo.mp4", type=str,
                        help='input video path')
    parser.add_argument('--result_path', default="./", type=str,
                        help='')
    parser.add_argument('--yolo11_path', default='D:\\bee_projet\\bee_Model\\yolov8\\bee_pollen-2024_04-v2-1_final\\0530_runs_HB\\detect\\mid_768\\weights\\best.pt', type=str,
                    help='')
    parser.add_argument('--pplite_path', default='F:\\pako_file\\segmentation_result\\bee_pollen-2024_04-08-v4\\pp_lite_t_seg50_224\\segmentation_transfer_learning\\RUN_20241211_034236_569410\\ckpt_best.pth', type=str,
                    help='')
    parser.add_argument('--save_txt', default=True, type=bool,
                    help='')
                    
    parser.add_argument('--show', default=False, type=bool,
                help='')
    parser.add_argument('--frame_imgsz', default=768, type=int,
                    help='')
    
    parser.add_argument('--pollen_imgsz', default=224, type=int,
                help='')

    parser.add_argument('--projet_name', default="unknow", type=str,)
    global ARGV
    ARGV = parser.parse_args()
    return ARGV

def draw_now_have (image, count):
    # 设置字体、字号和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # 白色

    # 在四个角落添加文字
    text_top_left = f"ALL: {count}"
    cv2.putText(image, text_top_left, (1200, 1000), font, font_size, font_color, font_thickness)

def draw_count (image, count):
    # 设置字体、字号和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # 白色

    # 获取图像的高度和宽度

    # 在四个角落添加文字
    text_top_left = f"L_B: {count[0]}"
    cv2.putText(image, text_top_left, (10, 30), font, font_size, font_color, font_thickness)

    text_bottom_left = f"L_P: {count[2]}"
    cv2.putText(image, text_bottom_left, (10, 70), font, font_size, font_color, font_thickness)

    text_top_right = f"R_B: {count[1]}"
    cv2.putText(image, text_top_right, (1100, 30), font, font_size, font_color, font_thickness)

    text_bottom_right = f"R_P: {count[3]}"
    cv2.putText(image, text_bottom_right, (1100, 70), font, font_size, font_color, font_thickness)

def draw_Pcount (image, track_objs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_thickness = 2

    for id, object_class in track_objs.items():
        if(object_class.i_pollen/object_class.i)>0.3:
            font_color = (0, 100, 255)  # 白色
        else:
            font_color = (255, 255, 255)  # 白色
        t = str(id)+str((int(object_class.i_pollen), int(object_class.i)))
        center = object_class.end.tolist()
        cv2.putText(image, t, center, font, font_size, font_color, font_thickness)
def get_total_frames(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 释放视频捕获对象
    cap.release()
    
    return total_frames


def main():
    #------------------init------------------#

    # Load the YOLOv8 model
    yolo_model = YOLO(ARGV.yolo11_path)
    # pplite_model = models.get(model_name=Models.PP_LITE_T_SEG75,
    #                 arch_params={"use_aux_heads": False},
    #                 num_classes=1,
    #                 checkpoint_path=ARGV.pplite_path).cuda().eval()
    pplite_model = models.get(model_name="pp_lite_t_seg50",
                    arch_params={"use_aux_heads": False},
                    num_classes=1,
                    checkpoint_path=ARGV.pplite_path).cuda().eval()
    # yolov8 stream
    results = yolo_model.track(ARGV.video_path
                            ,iou = 0.6
                            ,conf = 0.1
                            ,verbose=False
                            ,persist=True
                            ,tracker="bytetrack.yaml"
                            ,stream=True
                            ,agnostic_nms=True
                            ,imgsz=ARGV.frame_imgsz
                            )
    # useful setup
    init_trainer()
    #setup_device("cpu")
    setup_device(num_gpus=-1)

    #------------------Arg setting------------------#
    Re_ID = 11 #超過則移除紀錄
    wtih_pollen_num = 0 #帶花粉的蜜蜂label_num
    dis_threshold=1000
    pollen_bearing_rate=0.3

    #------------------some init------------------#
    tracking_ids = torch.tensor([], device="cuda:0")
    track_objs = {}

    count_total = np.array([0, 0, 0, 0, 0])  #[left, right, left_pollen, right_pollen, area]
    segArea_batch=[]
    total_Pretime=0

    #------------------loger------------------#
    os.makedirs(ARGV.result_path, exist_ok=True)

    #------------------Looping------------------#
    # step 1. Tracking after detection by yolo11 API
    # step 2. Crop pollen-bearing bee' BBox
    # step 3. Segment pollen
    # step 4. Update track_history & Counting (In-and-Out and Pollen-area)


    print("START")
    # Loop through the video frames
    t0=time.time()
    for result in results:
        result = result.cuda()
        total_Pretime+=result.speed['preprocess']
        if result.boxes.is_track:
            #-------- step 1.--------#
            all_result = result.boxes

            #-------- step 2.--------#
            track_info = torch.cat([all_result.id.unsqueeze(1), all_result.xyxy, all_result.cls.unsqueeze(1)], dim=1).int() # track_info: [id, x0, y0, x1, y1, cls]
            detect_ids = track_info[:,0]

            # 生成所有索引
            all_inx = torch.arange(len(track_info), device="cuda")

            # 找到與 wtih_pollen_num 相符的索引
            withPollen_inx = torch.where(track_info[:,-1] == wtih_pollen_num)[0]

            # 找到與 wtih_pollen_num 不相符的索引
            withoutPollen_inx = all_inx[~torch.isin(all_inx, withPollen_inx)]
            #withPollen_inx = torch.where(track_info[:,-1]==wtih_pollen_num)[0]

            #-------- step 3.--------#
            if withPollen_inx.nelement()>0:
                crop_xyxys = track_info[withPollen_inx, 1:5]

                y_sizes = crop_xyxys[:, 3] - crop_xyxys[:, 1]  # y_max - y_min
                x_sizes = crop_xyxys[:, 2] - crop_xyxys[:, 0]  # x_max - x_min
                # 紀錄在resize之前的image大小(原始大小)
                size_batch = torch.stack((y_sizes, x_sizes), dim=1).tolist()

                crop_ids = track_info[withPollen_inx, 0]
                tensor_img = torch.from_numpy(np.transpose(result.orig_img, (2, 0, 1))).cuda()
                #wiht_crops_np = [result.orig_img[y_min:y_max, x_min:x_max, :] for x_min, y_min, x_max, y_max in crop_xyxys]
                #with_crops_PIL = [ToPILImage()(wiht_crop[:,:,::-1]) for wiht_crop in wiht_crops_np] 

                """
                    進入模型前需將影像轉換成batch的形式才能夠運作
                """

                # 先crop下帶花粉蜜蜂的bounding box再resize，轉成tensor格式，最後cat將影像(tensor)以batch的方式疊加起來(4 dimention)
                with_crops_tensor = torch.cat([Resize(size=(ARGV.pollen_imgsz, ARGV.pollen_imgsz))(tensor_img[:, y_min:y_max, x_min:x_max]).unsqueeze(0) for x_min, y_min, x_max, y_max in crop_xyxys], dim=0).float() #input_batch.shape be like:[n, 3, w, h]

                #pre-process
                #input_batch = torch.cat([pre_proccess(with_crop_PIL ,128) for with_crop_PIL in with_crops_PIL]) #input_batch.shape be like:[n, 3, w, h]
                input_batch = pre_proccess2(with_crops_tensor) #input_batch.shape be like:[n, 3, w, h]
                output_batch = pplite_model(input_batch) #output_batch be like:[n, 3, w, h]
                # 真正後面會使用到的batch，變回原始size
                segMap_batch = post_proccess_mutiSize(output_batch,
                                                conf=0.5, 
                                                size_batch=size_batch) #segMap_batch.shape be like:torch.size=[n, 3, w, h]
                segArea_batch = [segMap.sum((-1,-2)) for segMap in segMap_batch] #sum of last 2 dim (seg Map). Area of target seg. Shape be like:list.size=[n,1]
            else:
                segArea_batch=[]

            #-------- step 4.--------#
            
            miss_ids = tracking_ids[~torch.isin(tracking_ids, detect_ids)] # tracking_ids difference detect_ids
                
            for miss_id in miss_ids: # miss_id
                miss_obj = track_objs[miss_id.item()]
                if miss_obj.miss_time < Re_ID:
                    miss_obj.miss_time += 1
                else:
                    count = miss_obj.over(dis_threshold=dis_threshold, pollen_bearing_rate=pollen_bearing_rate)
                    count_total += count
                    track_objs.pop(miss_id.item())
            
        
            for Info in track_info[withoutPollen_inx]:  #track_update
                id = Info[0].item()
                if id in track_objs:
                    track_objs[id].update(Info[1:])
                else:
                    track_objs[id] = bee_tracking(Info[1:])
            
            for Info, pollen_area in zip(track_info[withPollen_inx], segArea_batch):  #track_update 
                id = Info[0].item()
                if id in track_objs:
                    track_objs[id].update(Info[1:])
                    track_objs[id].update_pollen(pollen_area.int().squeeze(0)[0])
                else:
                    track_objs[id] = bee_tracking(Info[1:])

            tracking_ids = torch.tensor(list(track_objs.keys()), device="cuda:0") # update ids_list
        else:
            #tracking_ids = set()
            tracking_ids = torch.tensor([], device="cuda:0")
            track_objs = {}
    t1=time.time()
    print("ending....")


    # 获取并打印总帧数
    total_frame = get_total_frames(ARGV.video_path)
    total_time=t1-t0
    #count_total: ["leff_bee", "right_bee", "leff_pollen", "right_pollen", "pollen_area"]
    result_dict = {
            'Model': ARGV.projet_name,
            'video_name':os.path.basename(ARGV.video_path),
            'In_Without': count_total[1],
            'Out_Without': count_total[0],
            'In_With': count_total[3],
            'out_With': count_total[2],
            'Area':count_total[4],
            'total_frame':total_frame,
            'total_Pretime':total_Pretime/1000,
            'total_time':total_time,
            'FPS':total_frame/(total_time-(total_Pretime/1000)),
        }
    print(total_frame/(total_time-(total_Pretime/1000)))
    return result_dict
def list_files_and_folders(directory: str, t: str, search_subdirectories: bool):
    #t:search type. ex:mp4
    if t == "img":
        t = [".jpg", ".png"]
    path_list=[]
    # 使用 os.listdir 获取目录下所有文件和文件夹的列表
    items = os.listdir(directory)
    for item in items:
        # 利用 os.path.join 构建完整路径
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and os.path.splitext(full_path)[-1] in t:
            path_list.append(full_path)
            
        elif os.path.isdir(full_path) and search_subdirectories:
            path_list = path_list+list_files_and_folders(full_path, t, search_subdirectories)
    return path_list

if __name__ == '__main__':
    def scrip(target_path):

        version = "white_way_mutiple_04-11_v2"
        pp_lite_path='F:\\pako_file\\segmentation_result\\bee_pollen-2024_04-08-v4\\pp_lite_t_seg50_224\\segmentation_transfer_learning\\RUN_20241211_034236_569410\\ckpt_best.pth'
        pollen_imgsz=224
        yolo11_scale, imgsz = ["l_768", 768]
        yolo11_type = "best.pt"

        video_paths = list_files_and_folders(target_path, ".mp4", False)
        result_dict_list=[]
        exl_path = os.path.join(target_path, 'pollenTracktest_allnagetive.xlsx')
        print(f"{yolo11_type}, {yolo11_scale}")
        print(exl_path)
        for video_path in video_paths:
            print(video_path)
            # yolov8_path= f'D:\\bee_projet\\bee_Model\\yolov8\\bee_pollen-2024_04-v2-1_final\\0530_runs_HB\\detect\\{yolov8_scale}\\weights\\{yolov8_type}'
            yolo11_path= f'F:\\pako_file\\obj_detection_results\\runs\\detect_v2\\{version}_{yolo11_scale}\\weights\\{yolo11_type}'

            parse_args()
            ARGV.video_path = video_path
            ARGV.projet_name = f"{yolo11_scale}_{yolo11_type.split('.')[0]}"
            ARGV.yolo11_path = yolo11_path
            ARGV.pp_lite_path = pp_lite_path
            ARGV.result_path = f"./{yolo11_type}_{yolo11_scale}"
            ARGV.pollen_imgsz = pollen_imgsz
            ARGV.frame_imgsz = imgsz
            result_dict_list.append(main())


        with open (exl_path.replace("xlsx","txt"), "w+") as f:
            f.write(str(result_dict_list))
            
        try:
            result_pd = pd.DataFrame(result_dict_list)
            with pd.ExcelWriter(exl_path) as writer:
                result_pd.to_excel(writer, sheet_name='1', index=False)
        except:
            pass

    target_paths = [
                "F:/pako_file/model/tracking/video"
                ]


    for target_path in target_paths:
        scrip(target_path)

    os.system(f"python F:/pako_file/model/tracking/val_loop.py")
    print("Finish!")
        
