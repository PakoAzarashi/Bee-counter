
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as torch_transforms

from super_gradients.training.transforms.transforms import SegResize
from super_gradients.training import models
from super_gradients.training.utils.callbacks import BinarySegmentationVisualizationCallback, Phase
from super_gradients.training.utils.early_stopping import EarlyStop

from super_gradients.training.losses.bce_dice_loss import BCEDiceLoss

import time

#LossFuc = OhemBCELoss(threshold=0.5) #ok
#LossFuc = DetailLoss() #ok
#LossFuc = nn.BCEWithLogitsLoss() #ok
LossFuc = BCEDiceLoss() #ok

from super_gradients.training.metrics.segmentation_metrics import IoU, BinaryIOU, BinaryDice, PixelAccuracy

from super_gradients import Trainer
import argparse
import time
ARGV: None #all arg
VIDEO_PATH: str
RESOULT_PATH: str
MODELs_PATH: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lossfunction', default='BCEDiceLoss', type=str,
                        help='')
    parser.add_argument('--model', default='pp_lite_t_seg50', type=str,
                        help='"B" is a basic scale, "T" is mean tiny. ex: pp_lite_t_seg50 or pp_lite_b_seg50' )
    parser.add_argument('--project_name', default=str(time.time()), type=str,
                        help='')
    # 資料位置
    parser.add_argument('--dataset_path', default="F:\\pako_file\\segmentation_datasets\\dataset_mix_v4", type=str,
                        help='D:\\bee_projet\\bee_dataset\\bee_pollen-2024_04-v2-1_final_Pollen\\fusion_1b2')
    parser.add_argument('--img_size', default=96, type=int,
                        help='')
    global ARGV
    ARGV = parser.parse_args()


def create_unique_folder(path):
    # 檢查文件夾是否存在
    if not os.path.exists(path):
        os.makedirs(path)  # 如果不存在，創建該文件夾
        print(f"Folder created: {path}")
        return path
    else:
        i = 1
        # 當文件夾存在，嘗試添加數字後綴（folder1, folder2, ...)
        while True:
            new_path = f"{path}{i}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print(f"Folder created: {new_path}")
                return new_path
            i += 1


class CustomDataset(Dataset):
    NORMALIZATION_MEANS = [0.485, 0.456, 0.406]
    NORMALIZATION_STDS = [0.229, 0.224, 0.225]

    image_transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(NORMALIZATION_MEANS, NORMALIZATION_STDS)
    ])

    label_transform = torch_transforms.ToTensor()

    def __init__(self, root_dir, input_height, input_width):
        self.root_dir = root_dir
        self.input_height = input_height
        self.input_width = input_width

        # Assuming the structure is root_dir -> images_folder, labels_folder
        self.images_folder = os.path.join(root_dir, "images")
        self.labels_folder = os.path.join(root_dir, "labels")

        # Assuming corresponding image and label files have the same names
        self.image_files = os.listdir(self.images_folder)
        self.label_files = os.listdir(self.labels_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        pre_process = SegResize(h=self.input_height, w=self.input_width)

        image_filename = self.image_files[idx]
        mask_filename = self.label_files[idx]

        image_path = os.path.join(self.images_folder, image_filename)
        mask_path = os.path.join(self.labels_folder, mask_filename)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).split()[-1]

        transform_result = pre_process({"image": image, "mask": mask})
        image, mask = transform_result['image'], transform_result["mask"]

        image_tensor = self.image_transform(image)
        mask_tensor = self.label_transform(mask)[0]

        return image_tensor, mask_tensor


parse_args()

SUPERVISELY_DATASET_DOWNLOAD_PATH=os.path.join(os.getcwd(),"data")


CHECKPOINT_DIR=create_unique_folder(ARGV.project_name)
dataset_path = ARGV.dataset_path

# 定义一些参数
batch_size = 64
shuffle = True
input_size = ARGV.img_size
num_classes = 1
epochs = 1500
model_dict={}
model_name=str

loss_function = BCEDiceLoss()

# get預訓練模型
model = models.get(model_name=ARGV.model,
                   arch_params={"use_aux_heads": False},
                   num_classes=1,
                   pretrained_weights="cityscapes").cuda()

trainer = Trainer(experiment_name="segmentation_transfer_learning", ckpt_root_dir=CHECKPOINT_DIR)
# patience設定
early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="target_IOU", mode="max", patience=100, verbose=True)

# 訓練參數
train_params = {"max_epochs": epochs,
                "lr_mode": "PolyLRScheduler",
                "initial_lr": 0.5,
                "lr_warmup_epochs": 10,
                "multiply_head_lr": 10,
                "optimizer": "Lamb",
                "loss": "BCEDiceLoss",
                "mixed_precision": False,
                #"loss_weights":[0.1, 100],
                #"num_classes":1,
                "ema": True,
                "ema_params":
                    {
                    "decay": 0.9999,
                    "decay_type": "exp",
                    "beta": 15,
                    },
                "zero_weight_decay_on_bias_and_bn": True,
                "average_best_models": True,
                "metric_to_watch": "target_IOU",
                #"greater_metric_to_watch_is_better": True,
                #"train_metrics_list": [BinaryIOU(), BinaryDice()],
                "valid_metrics_list": [BinaryIOU(), BinaryDice()],
                "loss_logging_items_names": ["loss"],
                "phase_callbacks": [BinarySegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                                                            freq=1,
                                                                            last_img_idx_in_batch=4)],
                "phase_callbacks": [early_stop_acc],
                }


# 创建CustomDataset实例
train_dataset = CustomDataset(root_dir = os.path.join(dataset_path , "train"), input_height=input_size, input_width=input_size)
valid_dataset = CustomDataset(root_dir = os.path.join(dataset_path , "valid"), input_height=input_size, input_width=input_size)

# 创建DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)



trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)
print("Best Checkpoint mIoU is: "+ str(trainer.best_metric))