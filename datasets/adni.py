import os
import glob
import csv
import json
import random
import itertools
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot as plt
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = REPO_ROOT / "assets" / "splits_adni.json"
LEGACY_SPLIT_FILE = REPO_ROOT / "splits_adni.json"

FIXED_CLASS_MAP = {
    'AD': torch.tensor([0,0, 0, 1], dtype=torch.int32),
    'CN': torch.tensor([0,0, 1, 0], dtype=torch.int32),
    'MCI': torch.tensor([0,1, 0, 0], dtype=torch.int32)
}


def get_class_name_from_one_hot(one_hot_tensor):
    for class_name, class_tensor in FIXED_CLASS_MAP.items():
        if torch.equal(one_hot_tensor, class_tensor):
            return class_name
    return None  # Return None if no match is found

def normalize_to_minus1_to_1(image):
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val < 1e-5:
        return torch.zeros_like(image)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized * 2 - 1


def build_full_sequence(frame_infos):
    """Return the complete sequence of timepoints in order"""
    # Already sorted by time, so just return the full sequence
    if len(frame_infos) > 0:
        return [frame_infos]
    return []


def build_contiguous_sequences(frame_infos, traj_length, stride=1):
    """Sliding window sequences with specified stride"""
    sequences = []
    n = len(frame_infos)
    for i in range(0, n - traj_length + 1, stride):
        window = frame_infos[i:i+traj_length]
        sequences.append(window)
    return sequences

def build_all_combinations(frame_infos, traj_length):
    """All possible combinations of increasing time points"""
    sequences = []
    for comb in itertools.combinations(frame_infos, traj_length):
        if all(comb[i]['time'] < comb[i+1]['time'] for i in range(traj_length-1)):
            sequences.append(list(comb))
    return sequences

class TrajectoryDataset(Dataset):
    def __init__(self, root_dir='/ImageFlowNet/data/ADNI/Processed_ADNI/', traj_length=4, 
                 transform=None, mode="sequence", 
                 split="train", split_ratios=(0.7, 0.2, 0.1), seed=42,
                 sequence_mode="contiguous", stride=1,
                 include_classes=None,
                 view='axial'):
        self.view = view
        self.root_dir = ROOT+root_dir
        self.traj_length = traj_length
        self.transform = transform
        self.mode = mode.lower()
        self.split = split.lower()
        self.seed = seed
        self.sequence_mode = sequence_mode
        self.stride = stride
        self.include_classes = include_classes
        self.df = pd.read_csv(ROOT+'/ImageFlowNet/data/ADNI/ADNI1_Complete.csv')
        assert self.split in ["train", "val", "test"], "Invalid split"
        assert sequence_mode in ["contiguous", "all_combinations", "full_sequence"], "Invalid sequence mode"
        self.samples = []
        self.patient_classes = self._load_patient_classes()
        self.split_data = self._load_or_create_split(split_ratios)
      
        for case_folder in self.split_data[self.split]:
            case_path = os.path.join(self.root_dir, case_folder)
            if not os.path.isdir(case_path):
                continue
            
            patient_class = self.patient_classes.get(case_folder, None)
            if self.include_classes and patient_class not in self.include_classes:
                continue
            

            subject_df = self.df.loc[self.df['Subject'] == case_folder].copy()
            subject_df['Acq Date'] = pd.to_datetime(subject_df['Acq Date'])
            subject_df = subject_df.sort_values('Acq Date')
            subject_df['Time'] = subject_df['Acq Date'] - subject_df['Acq Date'].iloc[0]
            subject_df['Time'] = subject_df['Time'].dt.days
            match_str ="axial_slices/axial_slice_*" if self.view=='axial' else "coronal_slice_*"
            slice_folders = glob.glob(os.path.join(case_path, match_str))
            for slice_folder in slice_folders:
                slice_path = os.path.join(case_path, slice_folder)
                img_files = glob.glob(os.path.join(slice_path, "*.png"))
                frame_infos = []
                for img_file in img_files:
                    try:
                        time = float(os.path.splitext(os.path.basename(img_file))[0])
                        frame_info = {
                            'time': time,
                            'img_path': img_file,
                            'class': FIXED_CLASS_MAP[patient_class],
                            'age': subject_df.loc[subject_df['Time']== time, 'Age'].values[0]
                        }
                        frame_infos.append(frame_info)
                    except:
                        continue
                frame_infos.sort(key=lambda x: x['time'])

                if self.mode == "sequence":
                    if self.sequence_mode == "full_sequence":
                        # For full sequence mode, we need at least 1 frame
                        if len(frame_infos) < 1:
                            continue
                        sequences = build_full_sequence(frame_infos)
                    elif self.sequence_mode == "contiguous":
                        # For contiguous mode, we need at least traj_length frames
                        if len(frame_infos) < self.traj_length:
                            continue
                        sequences = build_contiguous_sequences(
                            frame_infos, self.traj_length, self.stride
                        )
                    else:  # all_combinations
                        if len(frame_infos) < self.traj_length:
                            continue
                        sequences = build_all_combinations(
                            frame_infos, self.traj_length
                        )
                    
                    self.samples.extend(sequences)
                else:
                    self.samples.extend(frame_infos)
        print('Sequence Mode:', self.sequence_mode)
        print(f"Total {self.split} samples ({self.mode} mode): {len(self.samples)}")
    
    def _load_patient_classes(self):
        patient_classes = {}
        csv_path = os.path.join(ROOT+'/ImageFlowNet/data/ADNI/ADNI1_Complete.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    patient_classes[row['Subject']] = row['Group']
        return patient_classes    

    def _load_or_create_split(self, split_ratios):
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        split_file = SPLIT_FILE if SPLIT_FILE.exists() else LEGACY_SPLIT_FILE
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            if split_file != SPLIT_FILE:
                with open(SPLIT_FILE, 'w') as f:
                    json.dump(split_data, f, indent=2)
        else:
            random.seed(self.seed)
            print(self.root_dir)
            all_cases = [d for d in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir, d))]
            if self.include_classes is not None:
                all_cases = [case for case in all_cases 
                            if self.patient_classes.get(case, None) in self.include_classes]
            random.shuffle(all_cases)
            num_train = int(len(all_cases) * split_ratios[0])
            num_val = int(len(all_cases) * split_ratios[1])
            split_data = {
                "train": all_cases[:num_train],
                "val": all_cases[num_train:num_train + num_val],
                "test": all_cases[num_train + num_val:]
            }
            with open(SPLIT_FILE, 'w') as f:
                json.dump(split_data, f, indent=2)
        return split_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "sequence":
            subseq = self.samples[idx]
            times = torch.tensor([x['time'] for x in subseq], dtype=torch.float32)
            ages = torch.tensor([x['age'] for x in subseq], dtype=torch.float32)
            images = []
            for x in subseq:
                img = Image.open(x['img_path']).convert("L")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            images = torch.stack(images)
            
            # Add sequence_length for full_sequence mode
            return {
                'times': times,
                'images': images,
                'class': subseq[0]['class'],
                'age': ages,
                'sequence_length': len(subseq),  # Include sequence length
                '_dataset_sequence_mode': self.sequence_mode  # Include sequence mode
            }
        else:
            frame = self.samples[idx]
            img = Image.open(frame['img_path']).convert("L")
            if self.transform:
                img = self.transform(img)
            return {
                'image': img,
                'class': frame['class']
            }

def collate_fn(batch):
    if 'images' in batch[0]:
        # Get sequence mode from dataset instance associated with this batch
        dataset = batch[0].get('_dataset_sequence_mode', None)
        
        # For full_sequence mode always use list format, otherwise use tensor format
        if dataset == "full_sequence":
            return {
                'times': [item['times'] for item in batch],
                'images': [item['images'] for item in batch],
                'class': torch.stack([item['class'] for item in batch]),
                'age': [item['age'] for item in batch],
                'sequence_length': torch.tensor([item['sequence_length'] for item in batch])
            }
        else:  # contiguous or all_combinations modes - use tensor format
            return {
                'times': torch.stack([item['times'] for item in batch]),
                'images': torch.stack([item['images'] for item in batch]).unsqueeze(-3),
                'class': torch.stack([item['class'] for item in batch]),
                'age': torch.stack([item['age'] for item in batch]),
                'sequence_length': torch.tensor([item['sequence_length'] for item in batch])
            }
    else:
        return {
            'images': torch.stack([item['image'] for item in batch]).unsqueeze(1),
            'class': [item['class'] for item in batch]
        }

def create_dataloaders_full_sequence(image_size=(256, 256), batch_size=16,
                      num_workers=0, split_ratios=(0.8, 0.1, 0.1), modes="sequence",
                      train_sequence_mode="full_sequence",include_classes=['AD', 'CN']):
    if modes == "autoencoder":
        train_transform = T.Compose([
            T.Resize(image_size),
            T.RandomApply([T.RandomRotation(45)], p=0.2),
            T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.2),
            T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
            T.Lambda(normalize_to_minus1_to_1)
        ])
    else:
        print('Sequence training Transform')
        train_transform = T.Compose([
            T.Resize(image_size),
            T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
            T.Lambda(normalize_to_minus1_to_1)
        ])
    
    val_transform = T.Compose([
        T.Resize(image_size),
        T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
        T.Lambda(normalize_to_minus1_to_1)
    ])

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = TrajectoryDataset(
        traj_length=-1,# not used in full_sequence mode
        transform=train_transform if split == "train" else val_transform,
        mode=modes,
        split=split,
        split_ratios=split_ratios,
        sequence_mode=train_sequence_mode,
        stride=1, # not used in full_sequence mode
        include_classes=include_classes
    )
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=1,
                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader



def create_dataloaders(image_size=(256, 256), batch_size=16, n_sequences=3, 
                      num_workers=0, split_ratios=(0.8, 0.1, 0.1), modes="autoencoder",
                      train_sequence_mode="contiguous",stride=1, include_classes=['AD', 'CN'],view='axial'):
    if modes == "autoencoder":
        train_transform = T.Compose([
            T.Resize(image_size),
            T.RandomApply([T.RandomRotation(45)], p=0.2),
            T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.2),
            T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
            T.Lambda(normalize_to_minus1_to_1)
        ])
    else:
        print('Sequence training Transform')
        train_transform = T.Compose([
            T.Resize(image_size),
            T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
            T.Lambda(normalize_to_minus1_to_1)
        ])
    
    val_transform = T.Compose([
        T.Resize(image_size),
        T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32)),
        T.Lambda(normalize_to_minus1_to_1)
    ])

    datasets = {}
    dir= '/ImageFlowNet/data/ADNI/Processed_ADNI_axial/' if view== 'axial' else '/ImageFlowNet/data/ADNI/Processed_ADNI/' 
    
    for split in ["train", "val", "test"]:
        datasets[split] = TrajectoryDataset(
        root_dir=dir,
        traj_length=n_sequences if split in ["train","val"] else 4,
        transform=train_transform if split == "train" else val_transform,
        mode=modes,
        split=split,
        split_ratios=split_ratios,
        sequence_mode=train_sequence_mode if split == "train" else "contiguous",
        stride=stride,
        include_classes=include_classes,
        view=view
    )
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=1,
                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader













if __name__ == "__main__":
    mode_='autoencoder'
    train_loader, val_loader, test_loader = create_dataloaders(
        modes=mode_,
        batch_size=32,view='axial'
    )
    for batch in train_loader:
        if mode_ == "autoencoder":
            print("Images shape:", batch['images'].shape)
            plt.imshow(batch['images'][5][0].numpy(), cmap='gray')
            plt.savefig('test.png')
        # else:
        #     # print("Times shape:", type(batch['times']))
        #     for time in batch['times']:
        #         print("Times shape:", time.shape, time)
        #     print('New Batch')
        #     # print("Images shape:", batch['images'][0].shape)
        #     # print("Classes:", batch['class'][0].shape, batch['class'][0])
        #     # print('Age:', batch['age'][0].shape, batch['age'][0])
