import pandas as pd
import openpyxl
import os
from ultralytics.utils import torch_utils
from ultralytics import YOLO
import itertools
import numpy as np
import time

# 德出現在年月份，作為檔名使用
t = time.time()
date = time.strftime('%Y%m%d')

# 確認檔案是否存在
def check_file(output_file):
    if os.path.isfile(output_file):
        os.remove(output_file)

def get_vals(model_path, imgsz, save_name, yaml_path):
    # Load a model
    model = YOLO(model_path)
    # Customize validation settings
    validation_results = model.val(data=yaml_path,
                                split='val', 
                                imgsz=imgsz, 
                                batch=1, 
                                conf=0.5, 
                                iou=0.5, 
                                device="0",
                                name=save_name,
                                )
    
    mAP50 = validation_results.box.map50
    mAP5095 = round(validation_results.box.map, 5)

    AP50_beewith,  AP50_beewithout = validation_results.box.ap50
    AP5095_beewith,  AP5095_beewithout = np.round(validation_results.box.ap, 5)

    Reacll50_bee = round(validation_results.confusion_matrix.matrix[[0,1], [0,1]].sum()/validation_results.confusion_matrix.matrix.sum(0)[0:2].sum(), 4)
    Precision50_bee = round(validation_results.confusion_matrix.matrix[[0,1], [0,1]].sum()/validation_results.confusion_matrix.matrix.sum(), 4)

    Reacll50_beewith, Reacll50_beewithout = np.round(validation_results.box.r, 4)
    Precision50_beewith, Precision50_beewithout = np.round(validation_results.box.p, 4)

    inference_time = round(validation_results.speed['inference'],2)
    all_time = round(sum(list(validation_results.speed.values())),2)

    Parameters_M = torch_utils.get_num_params(model)/1000**2
    FLOPS_G = torch_utils.get_flops_with_torch_profiler(model=model, imgsz=imgsz)
    FPS = round(1000 / inference_time, 2)
    result_df = pd.DataFrame([{
            'Model': save_name,
            'mAP50': mAP50,
            'mAP5095': mAP5095,
            'AP50_beewith': AP50_beewith,
            'AP5095_beewith': AP5095_beewith,
            'AP50_beewithout': AP50_beewithout,
            'AP5095_beewithout': AP5095_beewithout,
            'Recall50_bee': Reacll50_bee,
            'Precision50_bee': Precision50_bee,
            'Recall50_beewith': Reacll50_beewith,
            'Recall50_beewithout': Reacll50_beewithout,
            'Precision50_beewith': Precision50_beewith,
            'Precision50_beewithout': Precision50_beewithout,
            'Inference Time (ms)': inference_time,
            'All Time (ms)': all_time,
            "Parameters (M) ": Parameters_M,
            "FLOPS (G)":FLOPS_G,
            "FPS": FPS
        }])
    return result_df

def write_excel(output_file, name, model_path, model_scales, imgsz_list):
    # check_file(output_file)
    # try:
    # 初始化ExcelWriter
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 创建一个空的DataFrame来初始化Excel文件
        df_init = pd.DataFrame()
        df_init.to_excel(writer, index=False)
        header_written = True

        # 逐行写入每个模型的结果
        for model_scale, imgsz in itertools.product(model_scales, imgsz_list):
            model_path = f'F:/pako_file/obj_detection_results/runs/detect_v2/{name}_{model_scale}_{imgsz}/weights/best.pt'
            result_df = get_vals(model_path, imgsz, f"val-{model_scale}_{imgsz}", yaml_path)

            if header_written:
                # 追加到Excel文件
                result_df.to_excel(writer, index=False, startrow=writer.sheets['Sheet1'].max_row)
                header_written=False
            else:
                result_df.to_excel(writer, index=False, header=not writer.sheets, startrow=writer.sheets['Sheet1'].max_row)
        writer.save()
        print(f"Results saved to {output_file}")
    # except:
    #     print("error2")

        
if __name__ == "__main__":

    # 模型設定與資料初始化
    yaml_path = "F:/pako_file/obj_detection_datasets/white_way_mutiple_04-11_v2/white_way_mutiple_04-11_v2.yaml"
    model_path = "F:/pako_file/obj_detection_results/runs/detect_v2"
    result_path= "F:/pako_file/obj_detection_results"
    name = (yaml_path.split("/")[-1]).split(".")[0]
    output_file = os.path.join(result_path, f"{name}_model_results_{date}_train(FPS).xlsx")

    model_scales=["n", "s", "m", "l"]
    imgsz_list=[512, 576, 640, 704, 768]

    for i in range(3):
        write_excel(output_file, name, model_path, model_scales, imgsz_list)

