import time
import numpy as np
from ultralytics import YOLO

def get_val(model_path, model_scale, imgsz):
    model = YOLO(f"{model_path}/{name}_{model_scale}_{imgsz}/weights/best.pt")  # Load the best model
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)  # Create blank image
    sum_time = 0
    for i in range(1000):
        start = time.time()
        model.predict(img, verbose=False)
        # print(f"Scale {model_scale}_{imgsz}: {(time.time() - start) * 1000:.2f}ms")
        if i>0:
            sum_time += (time.time() - start) * 1000

    print(f"Average time for {model_scale}_{imgsz}: {sum_time / 999:.2f}ms")
    

if __name__ == "__main__":

    # 模型設定與資料初始化
    for i in range(3):
        yaml_path = "F:/pako_file/obj_detection_datasets/white_way_mutiple_04-11_v2/white_way_mutiple_04-11_v2.yaml"
        model_path = "F:/pako_file/obj_detection_results/runs/detect_v2"
        model_scales = ["n", "s", "m", "l"]
        imgsz_list=[512, 576, 640, 704, 768]
        name = (yaml_path.split("/")[-1]).split(".")[0]
        for model_scale in model_scales:
            for imgsz in imgsz_list:
                
                get_val(model_path, model_scale, imgsz)