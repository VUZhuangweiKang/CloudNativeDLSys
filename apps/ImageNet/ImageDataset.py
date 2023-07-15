import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensorV2
from lib.DLCJob import DLCJobDataset


class ImageDataset(DLCJobDataset):
    def __init__(self, name: str = 'train'):
        super().__init__(name)
        self.cls_idx = {}
        self.transform = Compose([
            Resize(224, 224),
            HorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _process_item(self, item_cloud_path: str, contents: np.array):
        label = item_cloud_path.split('/')[-1]
        if label not in self.cls_idx:
            self.cls_idx[label] = len(self.cls_idx)
        cls = self.cls_idx[label]

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # IMREAD_COLOR 表示读取为彩色图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读取的图像默认为 BGR，这里将其转换为 RGB
        img = self.transform(image=img)['image']
        
        return img, cls