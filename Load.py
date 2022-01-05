import glob, os, json
import pandas as pd
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
import cv2
import torch

STAGE1_DATA_LENGTH = -1 # -1 means the all data should be computed
CLASSIFIER_DATA_LENGTH = -1
FRAC_DATA_LENGTH = -1


def data_load(floder_path):
    fpath = floder_path
    # print(fpath)
    frac_imgs_path = glob.glob(fpath+'/Images/Fracture/*')
    norm_imgs_path = glob.glob(fpath+'/Images/Normal/*')
    frac_cord_path = glob.glob(fpath+'/Annotations/Fracture_Coordinate/*')
    slice_path = glob.glob(fpath+'/Annotations/Scaphoid_Slice/*') # json 檔
    
    print("[MING] Num of each floder: ", len(frac_imgs_path), len(norm_imgs_path), len(frac_cord_path), len(slice_path))

    return frac_imgs_path, norm_imgs_path, frac_cord_path, slice_path

def stage_one_df_create(slice_path, floder_path):
    df = pd.DataFrame(columns = ['img_id' , 'x1', 'y1', 'x2', 'y2'])
    for i in slice_path:
        basename = os.path.basename(i)
        img_id = os.path.splitext(basename)[0]
        with open(floder_path + '/Annotations/Scaphoid_Slice/' + img_id + '.json') as f:
            data_bbox = json.load(f)[0]['bbox']

        df = df.append({'img_id':img_id ,
                        'x1':float(data_bbox[0]), 
                        'y1':float(data_bbox[1]), 
                        'x2':float(data_bbox[2]), 
                        'y2':float(data_bbox[3])} , ignore_index=True)
    print("Length of stage 1: ", len(df))
    # print(df.head())
    return df

def stage_one_data_loader(df, floder_path):
    val_data = HandDataset(df[:], get_valid_transform(), floder_path)
    
    def collate_fn(batch): # batching
        return tuple(zip(*batch))

    valid_data_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        # num_workers=1,
        collate_fn=collate_fn
    )
    return val_data, valid_data_loader

def stage_classifier_data_loader(df, floder_path):
    val_data = Clip_classification_HandDataset(df[:], get_torch_transform(), floder_path)

    def collate_fn(batch): # batching
        return tuple(zip(*batch))

    valid_data_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        # num_workers=1,
        collate_fn=collate_fn
    )
    return val_data, valid_data_loader

def stage_fracture_detect_data_loader(df, floder_path):
    val_data = Clip_HandDataset(df[:], get_valid_transform(), floder_path)
    
    def collate_fn(batch): # batching
        return tuple(zip(*batch))

    valid_data_loader = DataLoader(
        val_data,
        batch_size=4,
        shuffle=False,
        # num_workers=1,
        collate_fn=collate_fn
    )
    return val_data, valid_data_loader

# Albumentations
def get_valid_transform():
    return Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_torch_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class HandDataset(Dataset):
    def __init__(self, data_frame, transforms, floder_path) -> None:
        super().__init__()
        self.df = data_frame
        self.images = data_frame['img_id']
        # print(len(self.images))
        self.transforms = transforms
        self.fpath = floder_path
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(idx, self.images[idx])
        
        image_filename = str(self.images[idx]) + '.bmp'
        # print(image_filename)
        if os.path.exists(self.fpath + '/Images/Fracture/' + image_filename): # 在 Fracture 的資料夾中
            image_arr = cv2.imread(self.fpath + '/Images/Fracture/' + image_filename, cv2.IMREAD_COLOR)
        elif os.path.exists(self.fpath + '/Images/Normal/' + image_filename): # 在 Normal 的資料夾中
            image_arr = cv2.imread(self.fpath + '/Images/Normal/' + image_filename, cv2.IMREAD_COLOR)
        else:
            print("Error Loading img")
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32) 
        image_arr /= 255.0

        image_id = self.images[idx]
        point = self.df[self.df['img_id'] == image_id]
        boxes = point[['x1', 'y1', 'x2', 'y2']].values        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # 有改這個
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((point.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['img_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms:
            sample = {
                'image': image_arr,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
        target['boxes'] = torch.stack(tuple(map(torch.tensor, 
                                                zip(*sample['bboxes'])))).permute(1, 0)
        
        return image, target, image_id


def classifier_df_create(slice_path, floder_path):
    df = pd.DataFrame(columns = ['img_id' , 'frac', 'x1', 'y1', 'x2', 'y2'])
    for i in slice_path:
        basename = os.path.basename(i)
        img_id = os.path.splitext(basename)[0]
        # print(img_id)
        with open(floder_path+'/Annotations/Scaphoid_Slice/' + img_id + '.json') as f:
                data_bbox = json.load(f)[0]['bbox']
        if os.path.exists(floder_path+'/Annotations/Fracture_Coordinate/' + img_id + '.csv'): # 在 Fracture 的資料夾中
            df = df.append({'img_id':img_id ,
                            'frac':float(1), # '1' stands for fracture scaphoid
                            'x1':int(data_bbox[0]), 
                            'y1':int(data_bbox[2]), 
                            'x2':int(data_bbox[1]), 
                            'y2':int(data_bbox[3])} , ignore_index=True)
        else:
            df = df.append({'img_id':img_id ,
                            'frac':float(0),  # '0' stands for fracture scaphoid
                            'x1':int(data_bbox[0]), 
                            'y1':int(data_bbox[2]), 
                            'x2':int(data_bbox[1]), 
                            'y2':int(data_bbox[3])} , ignore_index=True)
    return df

class Clip_classification_HandDataset(Dataset):
    def __init__(self, data_frame, transforms, floder_path) -> None:
        super().__init__()
        self.df = data_frame
        self.images = data_frame['img_id']
        # print(len(self.images))
        self.transforms = transforms
        self.fpath = floder_path
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(idx, self.images[idx])
        # print(self.images[idx])
        image_filename = str(self.images[idx]) + '.bmp'
        # print(image_filename)
        if os.path.exists(self.fpath+'/Images/Fracture/' + image_filename): # 在 Fracture 的資料夾中
            image_arr = cv2.imread(self.fpath+'/Images/Fracture/' + image_filename, cv2.IMREAD_COLOR)
        elif os.path.exists(self.fpath+'/Images/Normal/' + image_filename): # 在 Normal 的資料夾中
            image_arr = cv2.imread(self.fpath+'/Images/Normal/' + image_filename, cv2.IMREAD_COLOR)
        else:
            print("Error Loading img")
        # print(image_arr)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32) 
        image_arr /= 255.0
        
        image_id = self.images[idx]
        point = self.df[self.df['img_id'] == image_id]
        clip_bbox = point[['x1', 'x2', 'y1', 'y2']].values   
        image_arr = image_arr[clip_bbox[:, 1][0]:clip_bbox[:, 3][0], clip_bbox[:, 0][0]:clip_bbox[:, 2][0]]
        image_arr = cv2.resize(image_arr, (500, 500), interpolation=cv2.INTER_LINEAR)
        image_arr = image_arr.transpose((2,0,1))

        if point[['frac']].values == 1:
            label = 1
        if point[['frac']].values == 0:
            label = 0
            
        return image_arr, label, image_id


def fracture_df_create(slice_path, floder_path):
    df = pd.DataFrame(columns = ['img_id' , 'x1', 'x2', 'y1', 'y2','cx1', 'cy1', 'cx2','cy2', 'angle'])
    for i in slice_path:
        basename = os.path.basename(i)
        img_id = os.path.splitext(basename)[0]

        if os.path.exists(floder_path + '/Annotations/Fracture_Coordinate/' + img_id + '.csv'): # 在 Fracture 的資料夾中
            with open(floder_path + '/Annotations/Scaphoid_Slice/' + img_id + '.json') as f:
                clip_bbox = json.load(f)[0]['bbox']

            data_bbox = pd.read_csv(floder_path + '/Annotations/Fracture_Coordinate/' + img_id + '.csv')
            df = df.append({'img_id':img_id ,
                            'x1':int(clip_bbox[0]), 
                            'x2':int(clip_bbox[1]), 
                            'y1':int(clip_bbox[2]), 
                            'y2':int(clip_bbox[3]),
                            'cx1':float(data_bbox['ctrx'][0])-float(data_bbox['width'][0])/2, 
                            'cy1':float(data_bbox['ctry'][0])-float(data_bbox['height'][0])/2, 
                            'cx2':float(data_bbox['ctrx'][0])+float(data_bbox['width'][0])/2,
                            'cy2':float(data_bbox['ctry'][0])+float(data_bbox['height'][0])/2, 
                            'angle':float(data_bbox['angle'][0])} , ignore_index=True)
    print(len(df))
    df.head()
    return df

class Clip_HandDataset(Dataset):
    def __init__(self, data_frame, transforms, floder_path) -> None:
        super().__init__()
        self.df = data_frame
        self.images = data_frame['img_id']
        self.transforms = transforms
        self.fpath = floder_path
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = str(self.images[idx]) + '.bmp'
        if os.path.exists(self.fpath + '/Images/Fracture/' + image_filename): # 在 Fracture 的資料夾中
            image_arr = cv2.imread(self.fpath + '/Images/Fracture/' + image_filename, cv2.IMREAD_COLOR)
        else:
            print("Error Loading img")
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32) 
        image_arr /= 255.0

        image_id = self.images[idx]
        point = self.df[self.df['img_id'] == image_id]
        clip_bbox = point[['x1', 'x2', 'y1', 'y2']].values     
        image_arr = image_arr[clip_bbox[:, 1][0]:clip_bbox[:, 3][0], clip_bbox[:, 0][0]:clip_bbox[:, 2][0]]

        boxes = point[['cx1', 'cy1', 'cx2', 'cy2']].values
        if boxes[0,0] < 0:
            boxes[0,0] = 0
        if boxes[0,1] < 0:
            boxes[0,1] = 0
        if boxes[0,2] > image_arr.shape[0]:
            boxes[0,2] = image_arr.shape[0]
        if boxes[0,3] > image_arr.shape[1]:
            boxes[0,3] = image_arr.shape[1]

        boxes_angle = point[['angle']].values

        labels = torch.ones((point.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['img_id'] = torch.tensor(idx)
        # target['area'] = area
        target['iscrowd'] = iscrowd
        target['angle'] = torch.from_numpy(boxes_angle)
        if self.transforms:
            sample = {
                'image': image_arr,
                'bboxes': target['boxes'],
                'labels': target['labels'],
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
        target['boxes'] = torch.stack(tuple(map(torch.tensor, 
                                                zip(*sample['bboxes'])))).permute(1, 0)
        
        return image, target, image_id