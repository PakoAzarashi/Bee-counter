import timeit
import os
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
from thop import profile
import itertools
import time

from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients import Trainer, init_trainer
from super_gradients.training.utils.distributed_training_utils import setup_device


# 計算iou
def calculate_iou(label_map1, label_map2, class_id=1):
    """
    计算两个label_map之间指定类别的IoU。

    参数:
    - label_map1: 第一个label_map的numpy数组。
    - label_map2: 第二个label_map的numpy数组。
    - class_id: 要计算IoU的类别ID。

    返回:
    - IoU值。
    """
    # 將label_map計算交集和並集mask，只關注特定類別
    mask1 = label_map1 == class_id
    mask2 = label_map2 == class_id
    
    # 計算交集和並集
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # 計算IoU
    iou = intersection / union if union != 0 else 0
    return iou

# 列出檔案夾中之檔案
def list_files_and_folders(directory:str, t:str):
    #t:search type. ex:.mp4
    if t == "img":
        t = [".jpg", ".png"]
    path_list=[]
    # 使用 os.listdir 取得目錄下所有檔案和資料夾的列表
    items = os.listdir(directory)
    for item in items:
        # 利用 os.path.join 建構完整路徑
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and os.path.splitext(full_path)[-1] in t:
            path_list.append(full_path)
            
        elif os.path.isdir(full_path):
            path_list = path_list+list_files_and_folders(full_path, t)
    return path_list

pre_proccess_base = Compose([
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
])

def pre_proccess(img, re_size):
    # Resize the image and display
    img = Resize(size=(re_size, re_size))(img)
    # Run pre-proccess - transforms to tensor and apply normalizations.
    img_inp = pre_proccess_base(img).unsqueeze(0).cuda()
    return img_inp

def post_proccess(mask_prot, conf, x_dim, y_dim):
    # threshold of 0.5 for binary mask prediction. 
    mask_tensor = torch.sigmoid(mask_prot).gt(conf).squeeze()
    mask_tensor_ori = Resize(size = ([x_dim, y_dim]), interpolation=Image.NEAREST)(mask_tensor.unsqueeze(0))
    return mask_tensor_ori

def post_proccess2(mask_prot, conf, x_dim, y_dim):
    # threshold of 0.5 for binary mask prediction. 
    mask_tensor = mask_prot.gt(conf).squeeze()
    mask_tensor_ori = Resize(size = ([x_dim, y_dim]), interpolation=Image.NEAREST)(mask_tensor.unsqueeze(0))
    return mask_tensor_ori

def find_deepest_folders(root_dir):
    deepest_folders = []

    for root, dirs, files in os.walk(root_dir):
        if not dirs:  # If there are no subdirectories, it's a bottom-level folder
            deepest_folders.append(root)
    
    return deepest_folders

# 預熱，並計算FPS與inference time
def test_FPS(model, img_inp):
    num_iterations = 2000  # 迭代次数

    #  正式測試階段（1）使用Event來測試
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 預熱階段
    for _ in range(1000):
        model(img_inp)
    torch.cuda.synchronize()
    start_event.record()  # 紀錄開始時間
    for _ in range(num_iterations):
        model(img_inp)
    end_event.record()  # 紀錄結束時間
    # 同步等待GPU操作完成
    torch.cuda.synchronize()

    inference_time = start_event.elapsed_time(end_event) / 1000.0  # 轉換為秒
    fps = num_iterations / inference_time

    return fps, inference_time

def main_val(scales, input_sizes):
    for scale, input_size in itertools.product(scales, input_sizes):
        # model位置
        model_dir = f"F:\\pako_file\\segmentation_result\\bee_pollen-2024_04-08-v4_allhavenagetive\\{scale}_{input_size}"
        # useful setup
        init_trainer()
        #setup_device("cpu")
        setup_device(num_gpus=-1)
        #result_df = val_pplite(model_dir, dataset_path)
        checkpoints_dir_path = find_deepest_folders(model_dir)[0]

        model = models.get(model_name=scale,
                    arch_params={"use_aux_heads": False},
                    num_classes=1,
                    checkpoint_path=os.path.join(checkpoints_dir_path, "ckpt_best.pth")).cuda().eval()
        
        image_root = os.path.join(dataset_path, "images")
        images_paths = list_files_and_folders(image_root, "img")
        conf=0.9

        i=0
        IoU_list = []
        F1_list= []
        P_list= []
        R_list= []
        gts_area=[]
        dataset_path
        for image_path in images_paths:
            img = Image.open(image_path)
            x_dim, y_dim = img.size
            img_inp = pre_proccess(img=img, re_size=input_size)
            
            # Run inference
            mask_prot = model(img_inp)
            mask_tensor_ori = post_proccess2(mask_prot, conf=conf, x_dim = y_dim, y_dim = x_dim)
            predict_map = mask_tensor_ori
            predict_map = predict_map.squeeze(0).cpu().numpy().astype(bool)

            #load label
            label_path = image_path.replace("images", "labels")

            #calculate
            label_map = cv2.imread(label_path).transpose(2, 0, 1)[0, :, :].astype(bool)
            intersection = np.logical_and(predict_map, label_map).sum()
            if predict_map.sum() == 0:
                P=0
            else:
                P = intersection/predict_map.sum() 
            R = intersection/label_map.sum()
            if P==0 or R==0:
                F1=0
            else:    
                F1 = 2*P*R/(P+R)

            F1_list.append(F1)
            P_list.append(P)
            R_list.append(R)
            IoU_list.append(calculate_iou(label_map, predict_map))
            gts_area.append(label_map.sum())
            i+=1

        # speed_test
        fps, inference_time = test_FPS(model, img_inp)
        # 將數據轉換為DataFrame
        data = pd.DataFrame({'IoU': IoU_list, 'Area': gts_area})

        # 根據真實面積將資料分成四個部分
        bins = [0, 500, 1000, 1500, float('inf')]
        labels = ['0-500', '500-1000', '1000-1500', '>1500']
        data['Area_bins'] = pd.cut(data['Area'], bins=bins, labels=labels, right=False)

        # 計算每個區間的平均IoU值

        IoU_0, IoU_500, IoU_1000, IoU_1500 = data.groupby('Area_bins')['IoU'].mean().values
        
        #model complexity info
        FLOPs, Parameters = profile(model, inputs=(img_inp,))

        result_dict_list.append({
                'Model': scale,
                'Size': input_size,
                "P":sum(P_list)/i, 
                "R":sum(R_list)/i, 
                "IoU":sum(IoU_list)/i,
                "IoU_0-500": IoU_0,
                "IoU_500-1000": IoU_500,
                "IoU_1000-1500": IoU_1000,
                "IoU_1500-": IoU_1500,
                'Inference Time (ms)': inference_time,
                'FPS':fps,
                "Parameters (M) ": Parameters/1e6,
                "FLOPs (G)": FLOPs/1e9,
            })
    result_table = pd.DataFrame(result_dict_list)

    return result_table

# 測試資料位置
dataset_path = "F:\\pako_file\\segmentation_datasets\\dataset_mix_v4\\test"
scales = ["pp_lite_t_seg50", "pp_lite_b_seg50", "pp_lite_t_seg75", "pp_lite_b_seg75"]
input_sizes = [96,128,160,192,224]
result_dict_list = []
conf = 0.5

# 記得要回到main_val裡改model
result_table = main_val(scales, input_sizes)
data_name = dataset_path.split("\\")[-2]

# 輸出檔案位置與名稱
with open(f'F:/pako_file/segmentation_result/bee_pollen-2024_04-08-v4_allhavenagetive/{data_name}_test_performance4.txt', 'w') as f:
    f.write(result_table.to_string())

print(result_table)