#------------------------------#
"""
    輸入單張蜜蜂通道frame與label，先建立整張的label_map (mask_map) 
    再從BBox定位pollen-bearing bee, crop & save 單隻蜜蜂的花粉標註 (label_map)
    之後可在透過labelMap，抽取polygon Points
"""
import numpy as np
import cv2
import os
import cv2
import numpy as np
from skimage.measure import find_contours
split_set = ""

#old dataset
# img_root = f"D:\\bee projet\\bee_dataset\pollenSeg_dataset\V3_0406\{split_set}\\images"
# label_root = f"D:\\bee projet\\bee_dataset\pollenSeg_dataset\V3_0406\{split_set}\\labels"
img_root = f"F:\\pako_file\\segmentation_dataset\\rename2\\images"
label_root = f"F:\\pako_file\\segmentation_dataset\\rename2\\labels"
#new dataset
# new_root = f"D:\\bee projet\\bee_dataset\pollenSeg_dataset\\V3_test"
new_root = f"F:\\pako_file\\segmentation_dataset\\rename2\\crop2"
obj_num = "1" #cls_num
poly_num = "2" #cls_num
save_mode="label_map"

def list_files_and_folders(directory:str, t:str):
    #t:search type. ex:.mp4
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
            
        elif os.path.isdir(full_path):
            path_list = path_list+list_files_and_folders(full_path, t)
    return path_list
def extract_polygons(label_map, class_id:int =1):
    """
    提取指定类别的所有对象的多边形（polygon）坐标。

    参数:
    - labelmap: 一个二维numpy数组，表示语义分割的labelmap。
    - class_id: 需要提取多边形的类别ID。

    返回:
    - polygons: 一个列表，包含每个对象的多边形坐标。每个多边形坐标是一个形状为(N, 2)的numpy数组，N是多边形顶点的数量。
    """
    # 为指定的类别创建一个二值mask
    mask = label_map == class_id
    
    # 使用skimage.find_contours寻找mask中所有对象的边缘
    contours = find_contours(mask, level=0.1)
    
    # 将每个轮廓的坐标转换为多边形坐标列表
    polygons = [contour[:, [1, 0]] for contour in contours]  # 调整坐标顺序以匹配(x, y)格式
    
    return polygons


def main_process(obj_num:str ,poly_num:str, img_path:str, new_root:str, save_mode:str="label_map"):
    assert save_mode in ["label_txt", "label_map"]
    img = cv2.imread(img_path)
    # 中文路徑時的用法
    # img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)

    file_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_root, file_name+'.txt')

    new_images_root = os.path.join(new_root, "images")
    new_labels_root = os.path.join(new_root, "labels")

    os.makedirs(new_images_root, exist_ok=True)
    os.makedirs(new_labels_root, exist_ok=True)
    try:
        y_dim, x_dim, _ = img.shape
    except:
        print(f"{img_path} cannot found!!")


    with open(label_path, 'r+') as f:
        lines = f.readlines()

    #first loop for bulid
    label_map = np.zeros([y_dim, x_dim], dtype="uint8")
    for line in lines:
        if line[0] == poly_num:
            line_str =  line.split('\n')[0].split(' ')[1:]
            obj_label_math= (np.reshape(np.array(list(map(float, line_str))), (-1,2)) * [x_dim, y_dim]).astype('int')
            cv2.fillPoly(label_map, [obj_label_math], color=1)

    #secend loop for mask and crop
    i=0
    for line in lines:
        line_str =  line.split('\n')[0].split(' ')
        line_num = list(map(float, line_str))
        if line[0] == obj_num:
            write_IMAGE_path = os.path.join(new_images_root, file_name+f'_{i}.png')
            
            
            #[cls, x, y, w, h]
            xy = np.array(line_num)[[1,2]]
            wh = np.array(line_num)[[3,4]]
            x0, y0 = xy-(wh)/2
            x2, y2 = xy+(wh)/2
            
            x0, x2 = x0*x_dim, x2*x_dim
            y0, y2 = y0*y_dim, y2*y_dim

            # 抓出帶花粉蜜蜂的bounding box位置
            crop_img = img[int(y0):int(y2),int(x0):int(x2)]
            
            cv2.imwrite(write_IMAGE_path, crop_img)
            # 中文路徑時的用法
            # cv2.imencode(".png", crop_img)[1].tofile(write_IMAGE_path)
            crop_label = label_map[int(y0):int(y2),int(x0):int(x2)]

            if save_mode=="label_map":
                write_LABEL_path = os.path.join(new_labels_root, file_name+f'_{i}.png')
                cv2.imwrite(write_LABEL_path, crop_label, [cv2.IMWRITE_PNG_BILEVEL, 1])

            elif save_mode=="label_txt":
                write_LABEL_path = os.path.join(new_labels_root, file_name+f'_{i}.txt')
                obj_points = extract_polygons(crop_label, class_id = 1) # class_id in label_map
                with open (write_LABEL_path, "w") as t:
                    if len(obj_points)>0:
                        for obj_point in obj_points:
                            crop_dim = crop_label.shape[::-1]
                            line = ' '.join(map(str, (obj_point/crop_dim).ravel()))
                            t.write("0 " + line + "\n")

            print("Save: ", write_LABEL_path)
        i+=1
    return img

os.makedirs(new_root, exist_ok=True)
img_paths = list_files_and_folders(directory=img_root, t='img')

for img_path in img_paths:
    print(img_path)
    main_process(obj_num ,poly_num, img_path, new_root, save_mode=save_mode)
