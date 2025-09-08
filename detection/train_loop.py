from ultralytics import YOLO
import os
import itertools

# model = YOLO("yolov8s.pt")

# if __name__ == '__main__':
#     results = model.train(data="F:/pako_file/model/ultralytics-main/ultralytics/dataset/white_way_0429-0430.yaml", 
#                       epochs=700, 
#                       imgsz=640,
#                       batch=32,
#                       name="v8_s_640",
#                       lrf=0.0005,
#                       pretrained=True,
#                       patience=100)

scales = ["m"]
input_sizes=["512","576","640","704","768"]
# input_sizes=["512","576","640","704","768"]
yaml_path = "C:/dataset/white_way_mutiple_04-11_v2/white_way_mutiple_04-11_v2.yaml"
name = (yaml_path.split("/")[-1]).split(".")[0]

for scale, input_size in itertools.product(scales, input_sizes):
    CMD = f"yolo detect train data={yaml_path} name={name}_{scale}_{input_size} model=yolo11{scale[0]}.pt epochs=700 imgsz={input_size} batch=32 lrf=0.0005 pretrained=True patience=100"
    print(CMD)
    os.system(CMD)

    