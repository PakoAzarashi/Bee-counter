import time
import cv2
import numpy as np
import torch
import os
from torchvision.transforms import ToPILImage, Resize
import time
from ultralytics import YOLO

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients import init_trainer
from super_gradients.training.utils.distributed_training_utils import setup_device


from collections import defaultdict
from utils.bee_tracking import bee_tracking
from superG.proccess import pre_proccess2, post_proccess_mutiSize

import argparse

from PIL import Image

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
    parser.add_argument('--pplite_path', default='model/pplite-seg/RUN_20240312_185027_576124_best.pth', type=str,
                    help='')
    parser.add_argument('--save_txt', default=True, type=bool,
                    help='')
    
    parser.add_argument('--frame_imgsz', default=0, type=int,
                    help='')
    
    parser.add_argument('--pollen_imgsz', default=224, type=int,
                help='')
                    
    parser.add_argument('--show', default=False, type=bool,
                help='')
    parser.add_argument('--save_show', default=True, type=bool,
            help='')
    global ARGV
    ARGV = parser.parse_args()

def draw_now_have (image, count):
    # 设置字体、字号和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # 白色

    # 获取图像的高度和宽度
    height, width = 1080,1440

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
    height, width = 1080,1440

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

#------------------init------------------#
parse_args()

# Load the YOLO11 model
yolo_model = YOLO(ARGV.yolo11_path)
pplite_model = models.get(model_name="pp_lite_t_seg50",
                   arch_params={"use_aux_heads": False},
                   num_classes=1,
                   checkpoint_path=ARGV.pplite_path).cuda().eval()
# yolo11 stream
results = yolo_model.track(ARGV.video_path
                        ,iou = 0.6
                        ,conf = 0.1
                        ,imgsz=ARGV.frame_imgsz
                        ,verbose=False
                        ,persist=True
                        ,tracker="bytetrack.yaml"
                        ,stream=True
                        ,agnostic_nms=True
                        )
# useful setup
init_trainer()
#setup_device("cpu")
setup_device(num_gpus=-1)

# Open the video file
video_name = os.path.splitext(os.path.basename(ARGV.video_path))[0]
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

#------------------Arg setting------------------#
i=1
Re_ID = 10 #超過則移除紀錄
wtih_pollen_num = 0 #帶花粉的蜜蜂label_num
dis_threshold=1000
pollen_bearing_rate=0.3

#------------------some init------------------#
tracking_ids = torch.tensor([], device="cuda:0")
track_objs = {}
area_list = []
ids_list = []

count_keys = ["leff_bee", "right_bee", "leff_pollen", "right_pollen", "pollen_area"]
count_total = np.array([0, 0, 0, 0, 0])  #[left, right, left_pollen, right_pollen, area]
segArea_batch=[]
detect_ids=None
track_history = defaultdict(lambda: [])

#------------------loger------------------#
save_path = f"{video_name}_log.txt"
os.makedirs(ARGV.result_path, exist_ok=True)
loger = open(os.path.join(ARGV.result_path, save_path), "w+")


ts = time.localtime(time.time())
t_now = time.strftime('%Y/%m/%d %H:%M:%S',ts)

loger.write(f"{video_name}, writed at {t_now}\n")
loger.write(f"{count_keys}\n")

#------------------Looping------------------#
# step 1. Tracking after detection by yolo11 API
# step 2. Crop pollen-bearing bee' BBox
# step 3. Segment pollen
# step 4. Update track_history & Counting (In-and-Out and Pollen-area)


if ARGV.save_show:
    os.makedirs(ARGV.result_path, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(ARGV.result_path, video_name+"_draw.mp4"), fourcc, 30.0, (1440,  1080))

print("START")
# Loop through the video frames
for result in results:
    result = result.cuda()
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
        withPollen_inx = torch.where(track_info[:,-1]==wtih_pollen_num)[0]

        #-------- step 3.--------#
        if withPollen_inx.nelement()>0:
            crop_xyxys = track_info[withPollen_inx, 1:5]

            y_sizes = crop_xyxys[:, 3] - crop_xyxys[:, 1]  # y_max - y_min
            x_sizes = crop_xyxys[:, 2] - crop_xyxys[:, 0]  # x_max - x_min
            size_batch = torch.stack((y_sizes, x_sizes), dim=1).tolist()

            crop_ids = track_info[withPollen_inx, 0]
            
            tensor_img = torch.from_numpy(np.transpose(result.orig_img, (2, 0, 1))).cuda()
            # wiht_crops_np = [result.orig_img[y_min:y_max, x_min:x_max, :] for x_min, y_min, x_max, y_max in crop_xyxys]
            # with_crops_PIL = [ToPILImage()(wiht_crop[:,:,::-1]) for wiht_crop in wiht_crops_np]

            with_crops_list = []
            for x_min, y_min, x_max, y_max in crop_xyxys:
                # 裁剪原始影像
                cropped_tensor = tensor_img[:, y_min:y_max, x_min:x_max]                
                # 調整大小至固定尺寸
                resized_tensor = Resize(size=(ARGV.pollen_imgsz, ARGV.pollen_imgsz))(cropped_tensor)                
                # 增加一個維度
                resized_tensor = resized_tensor.unsqueeze(0)
                # 將結果添加到列表中
                with_crops_list.append(resized_tensor)
            # 合併所有裁剪後的張量
            with_crops_tensor = torch.cat(with_crops_list, dim=0).float()
            # with_crops_tensor = torch.cat([Resize(size=(ARGV.pollen_imgsz, ARGV.pollen_imgsz))(tensor_img[:, y_min:y_max, x_min:x_max]).unsqueeze(0) for x_min, y_min, x_max, y_max in crop_xyxys], dim=0).float()
            
            #pre-process
            # input_batch = torch.cat([pre_proccess(with_crop_PIL ,128) for with_crop_PIL in with_crops_PIL]) #input_batch.shape be like:[n, 3, w, h]
            input_batch = pre_proccess2(with_crops_tensor) #input_batch.shape be like:[n, 3, w, h]
            output_batch = pplite_model(input_batch) #output_batch be like:[n, 3, w, h]
            segMap_batch = post_proccess_mutiSize(output_batch,
                                            conf=0.5, # 改conf根本沒用qwq
                                            size_batch=size_batch) #segMap_batch.shape be like:[n, 3, w, h]
            segArea_batch = [segMap.sum((-1,-2)) for segMap in segMap_batch] #sum of last 2 dim (seg Map). Area of target seg. Shape be like:[n,1]


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
                # 把outlier去除
                count_total += count
                loger.write(str({i:list(count)})+"\n")
                track_objs.pop(miss_id.item())
        
       
        for Info in track_info[withoutPollen_inx]:  #track_update
            id = Info[0].item()
            if id in track_objs:
                track_objs[id].update(Info[1:])
            else:
                track_objs[id] = bee_tracking(Info[1:])
        #print(track_objs.keys())
        #print(withPollen_inx)
        for Info, pollen_area in zip(track_info[withPollen_inx], segArea_batch):  #track_update
            id = Info[0].item()
            if id in track_objs:
                track_objs[id].update(Info[1:])
                track_objs[id].update_pollen(pollen_area.int().squeeze(0)[0])
            else:
                track_objs[id] = bee_tracking(Info[1:])
        
        tracking_ids = torch.tensor(list(track_objs.keys()), device="cuda:0") # update ids_list

    else:
        pass
        #tracking_ids = set()
        tracking_ids = torch.tensor([], device="cuda:0")
        track_objs = {}
        detect_ids=None
    #--------draw---------#

    #annotated_frame = result.plot(line_width=2,
    #                              font_size=10)
    
    annotated_frame = result.orig_img
    
    # Plot the tracks
    for track_id in track_objs.keys():
        """
        track = track_history[track_id]
        color = track_objs[track_id].color

        track.append(track_objs[track_id].end.tolist())  # x, y center point
        if len(track) > 100:  # retain 30 tracks for 30 frames
            track.pop(0)
        """
        track = [track_objs[track_id].start.tolist(), track_objs[track_id].end.tolist()]
        color = track_objs[track_id].color
        
        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=3)

    draw_count(annotated_frame, count_total)
    # cv2.polylines(annotated_frame, [np.array([[lim_l,0], [lim_l,1080]])], isClosed=False, color=(255,255,255), thickness=5)
    # cv2.polylines(annotated_frame, [np.array([[lim_r,0], [lim_r,1080]])], isClosed=False, color=(255,255,255), thickness=5)

    draw_Pcount(annotated_frame, track_objs)
    if detect_ids is not None:
        draw_now_have(annotated_frame, len(detect_ids))

    if ARGV.show:
    # Display the annotated frame
        display_frame = cv2.resize(annotated_frame, (1200, 900), interpolation=cv2.INTER_AREA)
        cv2.imshow("YOLO11 Tracking", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if ARGV.save_show:
        out.write(annotated_frame)

    i+=1

print("before ending....")

count_keys = ["leff_bee", "right_bee", "leff_pollen", "right_pollen", "pollen_area"]
count_dict= dict(zip(count_keys, count_total))
print(count_dict)
loger.write(str({"all":list(count_total)})+"\n")

ts = time.localtime(time.time())
t_now = time.strftime('%Y/%m/%d %H:%M:%S',ts)
loger.write(f"time: {t_now}")
loger.close()

loger2 = open(os.path.join(ARGV.result_path, "total_tracking.txt"), "a+")
loger2.write(str({video_name:list(count_total)})+"\n")

cv2.destroyAllWindows()
out.release()