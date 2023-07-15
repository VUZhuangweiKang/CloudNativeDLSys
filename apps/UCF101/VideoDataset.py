from typing import Any
import torch
import os
import numpy as np
import io
import av
from lib.DLCJob import DLCJobDataset


class VideoDataset(DLCJobDataset):
    def __init__(self, trainTest_dir, name='train', num_frames = 50, transform = None, target_transform = None):
        super().__init__(name)

        self.trainTest_dir = trainTest_dir
        self.mode = name.strip().lower()
        
        assert self.mode in ['train', 'test']
        
        self.transform = transform
        self.target_transform = target_transform

        self.num_frames = num_frames
        
        self.class_dict = {}
        with open(f"{trainTest_dir}/classInd.txt", 'r') as f:
            for line in f.readlines():
                classIdx, classLabel = line.split()
                self.class_dict[classLabel] = int(classIdx)
        
        path_txt = os.path.join(self.trainTest_dir, f"{self.mode}.txt")
        self.video_filename_list, self.classesIdx_list = self.read_traintest_txt(path_txt)
        self.video_filename_list = {fname: i for i, fname in enumerate(self.video_filename_list)}
        
    def __len__(self):
        return len(self.video_filename_list)
        
    def read_traintest_txt(self, path):
        video_filename = []
        classIdx = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if self.mode == 'train':
                    filename, Idx = line.split()
                    video_filename.append(filename.lower())
                    classIdx.append(int(Idx))
                elif self.mode == 'test':
                    filename = line
                    classLabel = filename.split('/')[0]
                    Idx = self.class_dict[classLabel]
                    video_filename.append(filename.lower())
                    classIdx.append(int(Idx))
                    
        return video_filename, classIdx

    def read_video(self, video_bytes):
        frames = []
        count_frames = 0
        
        # Create a BytesIO object and open the video
        container = av.open(io.BytesIO(video_bytes))

        for frame in container.decode(video=0):
            # Convert PyAV video frame to PIL image then to tensor
            pil_image = frame.to_image()
            np_image = np.array(pil_image)

            if self.transform:
                transformed = self.transform(image = np_image)
                frame = transformed['image']

            frames.append(frame)
            count_frames += 1
        
        stride = count_frames // self.num_frames

        new_frames = []
        count = 0
        for i in range(0, count_frames, stride):
            if count >= self.num_frames:
                break
            new_frames.append(frames[i])
            count += 1
        
        return torch.stack(new_frames, dim = 0)
            
    def _process_item(self, item_cloud_path: str, contents: Any) -> Any:
        video_filename = item_cloud_path.split('/')[-1]
        video_filename = f"{video_filename.split('_')[1]}/{video_filename}".lower()
        idx = self.video_filename_list[video_filename]
        classIdx = self.classesIdx_list[idx]
        frames = self.read_video(contents)
        return frames, classIdx - 1