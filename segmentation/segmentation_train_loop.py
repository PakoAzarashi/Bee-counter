import os
import itertools
# model scales and input sizes
scales = ['pp_lite_t_seg50', 'pp_lite_b_seg50', 'pp_lite_t_seg75', 'pp_lite_b_seg75']
input_sizes=["96","128","160","192","224"]
# dataset path
dataset = 'F:\\pako_file\\segmentation_datasets\\dataset_mix_v4'

# exp:python 欲執行之檔案絕對位置 --model {scale} --img_size {input_size} --dataset_path {dataset} --project_name {scale}_{input_size}
for scale, input_size in itertools.product(scales, input_sizes):
    CMD = f"python F:/pako_file/model/pp_liteSeg/train_HBv2.py --model {scale} --img_size {input_size} --dataset_path {dataset} --project_name {scale}_{input_size}"
    print(CMD)
    os.system(CMD)
