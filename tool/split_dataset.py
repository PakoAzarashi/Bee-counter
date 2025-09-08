import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, train_ratio, valid_ratio, test_ratio):
    #assert train_ratio + valid_ratio + test_ratio == 1, "比例总和必须为1"

    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    # 获取所有图片的文件名（假设图片和标签文件名相同，仅扩展名不同）
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    # 按照比例划分数据集
    train_files, test_files = train_test_split(images, test_size=test_ratio, random_state=42)
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)  # 调整验证集比例
    train_files, valid_files = train_test_split(train_files, test_size=valid_ratio_adjusted, random_state=42)

    # 创建新的目录结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(data_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, 'labels'), exist_ok=True)

    # 定义一个辅助函数来复制文件
    def copy_files(files, split, label_type):
        for f in files:
            # 复制图片
            shutil.copy(os.path.join(images_dir, f), os.path.join(data_dir, split, 'images', f))
            # 复制标签（假设标签文件的扩展名为.txt）
            label_file = os.path.splitext(f)[0] + f'.{label_type}'
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(data_dir, split, 'labels', label_file))

    # 复制文件到相应的目录
    copy_files(train_files, 'train', label_type="png")
    copy_files(valid_files, 'valid', label_type="png")
    copy_files(test_files, 'test', label_type="png")

    print(f"数据集切分完成：训练集{len(train_files)}，验证集{len(valid_files)}，测试集{len(test_files)}。")

# 使用示例
data_directory = 'F:\\pako_file\\segmentation_result\\bee_pollen-2024_04-08-v2\\crop'
split_dataset(data_directory, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)
