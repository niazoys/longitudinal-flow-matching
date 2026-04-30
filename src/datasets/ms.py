import os
import glob
import csv
import json
import random
import itertools
from PIL import Image
from PIL import ImageFile
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from pathlib import Path


ROOT = str(Path(__file__).resolve().parent.parent.parent)


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




class SequenceLengthBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains sequences of uniform length.
    Batches are created by grouping samples with the same sequence length.
    """
    def __init__(self, dataset, batch_size, shuffle=True, min_batch_size=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_batch_size = min_batch_size  # Minimum batch size
        self.length_groups = list(dataset.samples_by_length.keys())
        self.batch_indices = self._generate_batches()
        
    def _generate_batches(self):
        batches = []
        
        # Process each length group
        for length in self.length_groups:
            # Get all indices for this length
            indices = list(range(len(self.dataset.samples_by_length[length])))
            
            # Skip if we don't have enough samples to meet minimum batch size
            if len(indices) < self.min_batch_size:
                continue
                
            if self.shuffle:
                random.shuffle(indices)
            
            # Create standard batches
            i = 0
            length_batches = []
            while i + self.batch_size <= len(indices):
                batch = indices[i:i + self.batch_size]
                length_batches.append(batch)
                i += self.batch_size
            
            # Handle remaining samples
            remaining = indices[i:]
            if len(remaining) >= self.min_batch_size:
                # If enough samples remain to meet minimum, create another batch
                length_batches.append(remaining)
            elif len(remaining) > 0 and length_batches:
                # Add remaining samples to the last batch
                length_batches[-1] = length_batches[-1] + remaining
            
            # Add all batches for this length to our result
            for batch in length_batches:
                batches.append((length, batch))
        
        if self.shuffle:
            random.shuffle(batches)
        return batches
    
    def __iter__(self):
        if self.shuffle:  # Regenerate batch indices if shuffling
            self.batch_indices = self._generate_batches()
            
        for length, batch_indices in self.batch_indices:
            # Convert to global dataset indices
            global_indices = []
            for idx in batch_indices:
                # Convert to the index in the main samples list
                global_idx = self.dataset.length_to_global_index[length][idx]
                global_indices.append(global_idx)
            
            yield global_indices
    
    def __len__(self):
        return len(self.batch_indices)

class UniformLengthTrajectoryDataset(Dataset):
    def __init__(self, root_dir='ImageFlowNet/data/ms/brain_MS_images_256x256/', traj_length=4, 
                 transform=None, mode="sequence", 
                 split="train", split_ratios=(0.7, 0.2, 0.1), seed=42,
                 sequence_mode="full_sequence", stride=1):
        self.root_dir = ROOT+root_dir
        self.traj_length = traj_length
        self.transform = transform
        self.mode = mode.lower()
        self.split = split.lower()
        self.seed = seed
        self.sequence_mode = sequence_mode
        self.stride = stride
        self.include_classes = False
        assert self.split in ["train", "val", "test"], "Invalid split"
        assert sequence_mode in ["contiguous", "all_combinations", "full_sequence"], "Invalid sequence mode"
        self.samples = []
        self.split_data = self._load_or_create_split(split_ratios)
      
        for case_folder in self.split_data[self.split]:
            case_path = os.path.join(self.root_dir, case_folder)
            if not os.path.isdir(case_path):
                continue
            
            match_str ="slice*" 
            slice_folders = glob.glob(os.path.join(case_path, match_str))
            for slice_folder in slice_folders:
                slice_path = os.path.join(case_path, slice_folder)
                img_files = glob.glob(os.path.join(slice_path,'preprocessed', "*.png"))
                frame_infos = []
                for img_file in img_files:
                    try:
                        time = float(os.path.splitext(os.path.basename(img_file))[0].split('_')[-1])-1.0
                        frame_info = {
                            'time': time,
                            'img_path': img_file,
                            'class': torch.tensor([0, 0,0,0], dtype=torch.float32),
                            'age': torch.tensor(0.0, dtype=torch.float32),
                            'sex': torch.tensor([0,0], dtype=torch.float32)
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
        
        # Group samples by sequence length when using full_sequence mode
        if self.sequence_mode == "full_sequence":
            self.samples_by_length = {}
            self.length_to_global_index = {}
            
            for idx, sample in enumerate(self.samples):
                seq_len = len(sample)
                if seq_len not in self.samples_by_length:
                    self.samples_by_length[seq_len] = []
                    self.length_to_global_index[seq_len] = []
                self.samples_by_length[seq_len].append(sample)
                self.length_to_global_index[seq_len].append(idx)
            
            # Print sequence length distribution
            print(f"Sequence length distribution in {self.split} set:")
            for length, samples in sorted(self.samples_by_length.items()):
                print(f"  Length {length}: {len(samples)} samples")
    

    def _load_or_create_split(self, split_ratios=(0.75, 0.1, 0.15)):
        all_cases=['training01', 'training02', 'training03', 'training04', 
                   'test01', 'test02', 'test03', 'test04','test05',
                   'test06', 'test07', 'test08', 'test09', 'test10',
                   'test11', 'test12', 'test13', 'test14', 
                   'training05']
        split_data = {
                "train": all_cases[:int(len(all_cases) * split_ratios[0])],
                "val": all_cases[int(len(all_cases) * split_ratios[0]):int(len(all_cases) * (split_ratios[0] + split_ratios[1]))],
                "test": all_cases[int(len(all_cases) * (split_ratios[0] + split_ratios[1])):]
            }
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
            
            return {
                'times': times,
                'images': images,
                'class': subseq[0]['class'],
                'age': ages,
                'sex': subseq[0]['sex'],
                'sequence_length': len(subseq),
                '_dataset_sequence_mode': self.sequence_mode
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

def uniform_length_collate_fn(batch):
    if 'images' in batch[0]:
        # For full_sequence mode, now we can use tensor format because lengths are uniform within a batch
        if batch[0]['_dataset_sequence_mode'] == "full_sequence":
            # All sequences in this batch will have the same length
            seq_length = batch[0]['sequence_length']
            
            # Convert to tensor format
            return {
                'times': torch.stack([item['times'] for item in batch]),
                'images': torch.stack([item['images'] for item in batch]).unsqueeze(-3),
                'class': torch.stack([item['class'] for item in batch]),
                'age': torch.stack([item['age'] for item in batch]),
                'sex': torch.stack([item['sex'] for item in batch]),
                'sequence_length': torch.tensor([item['sequence_length'] for item in batch])
            }
        else:  # contiguous or all_combinations modes
            return {
                'times': torch.stack([item['times'] for item in batch]),
                'images': torch.stack([item['images'] for item in batch]).unsqueeze(-3),
                'class': torch.stack([item['class'] for item in batch]),
                'age': torch.stack([item['age'] for item in batch]),
                'sex': torch.stack([item['sex'] for item in batch]),
                'sequence_length': torch.tensor([item['sequence_length'] for item in batch])
            }
    else:
        return {
            'images': torch.stack([item['image'] for item in batch]).unsqueeze(1),
            'class': [item['class'] for item in batch]
        }

def create_uniform_length_dataloaders(image_size=(256, 256), batch_size=16,
                      num_workers=0, split_ratios=(0.8, 0.1, 0.1), modes="sequence",
                      train_sequence_mode="full_sequence"):
    """
    Creates dataloaders with uniform sequence length batches for full_sequence mode.
    Each batch will contain sequences of the same length.
    """
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
    dir= '/ImageFlowNet/data/ms/brain_MS_images_256x256/'
    for split in ["train", "val", "test"]:
        datasets[split] = UniformLengthTrajectoryDataset(
            root_dir=dir,
            traj_length=-1,  # not used in full_sequence mode
            transform=train_transform if split == "train" else val_transform,
            mode=modes,
            split=split,
            split_ratios=split_ratios,
            sequence_mode=train_sequence_mode,
            stride=1,  # not used in full_sequence mode
        )
    
    # Use custom batch sampler for full_sequence mode
    if train_sequence_mode == "full_sequence":
        train_sampler = SequenceLengthBatchSampler(datasets['train'], batch_size, shuffle=True)
        val_sampler = SequenceLengthBatchSampler(datasets['val'], batch_size, shuffle=False)
        test_sampler = SequenceLengthBatchSampler(datasets['test'], 1, shuffle=False)
        
        train_loader = DataLoader(datasets['train'], batch_sampler=train_sampler, 
                                num_workers=num_workers, collate_fn=uniform_length_collate_fn)
        val_loader = DataLoader(datasets['val'], batch_sampler=val_sampler,
                                num_workers=num_workers, collate_fn=uniform_length_collate_fn)
        test_loader = DataLoader(datasets['test'], batch_sampler=test_sampler,
                                num_workers=num_workers, collate_fn=uniform_length_collate_fn)
    else:
        # Original code for other sequence modes
        train_loader = DataLoader(datasets['train'], batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers, collate_fn=uniform_length_collate_fn)
        val_loader = DataLoader(datasets['val'], batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, collate_fn=uniform_length_collate_fn)
        test_loader = DataLoader(datasets['test'], batch_size=1,
                                shuffle=False, num_workers=num_workers, collate_fn=uniform_length_collate_fn)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    mode_ = 'autoencoder'  # or 'sequence'
    train_loader, val_loader, test_loader = create_uniform_length_dataloaders(
        modes=mode_,
        batch_size=8
    )
    
    # Test that batches contain uniform length sequences
    for i, batch in enumerate(train_loader):
        if i >= 13:  # Just check a few batches
            break
            
        print(f"Batch {i}:")
        sequence_lengths = batch['sequence_length']
        print(f"  All sequences have length: {sequence_lengths[0].item()}")
        assert all(sl == sequence_lengths[0] for sl in sequence_lengths), "Batch contains mixed sequence lengths!"
        
        # Check shape of tensors
        print(f"  Times shape: {batch['times'].shape}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Class shape: {batch['class'].shape}")
        print(f"  sex shape: {batch['sex']}")