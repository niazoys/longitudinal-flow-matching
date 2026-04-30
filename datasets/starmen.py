import os
import glob
import csv
import json
import random
import itertools
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
# (Other imports as needed)

class StarmenDataset(Dataset):
    def __init__(self, num_visits=10, memory=2, num_patients=500,
                 trainvaltest=['train','val'], ae_training=False,
                 n_sequence=4, sequence_mode="contiguous", stride=1,
                 root_dir='/gpfs/work4/0/prjs1289/ImageFlowNet/data/starmen/',
                 inference_mode=False):
        """
        Parameters:
            num_visits: (int) number of visits per patient to consider (if fixed).
            memory: (int) (legacy) number of memory visits used (if applicable).
            num_patients: (int) maximum number of patients to use.
            trainvaltest: (list) which splits to include, as specified in the CSV.
            ae_training: (bool) if True use autoencoder mode (one image per sample).
            n_sequence: (int) length of the sequence (trajectory) to generate.
            sequence_mode: (str) "contiguous" for sliding window sequences,
                                  "all_combinations" for every possible increasing sequence.
            stride: (int) stride to use in contiguous mode.
            root_dir: (str) root directory of the dataset.
            inference_mode: (bool) if True, may modify sample retrieval.
        """
        self.num_visits = num_visits
        self.memory = memory  # (This may be legacy if you now use full sequences)
        self.num_patients = num_patients
        self.trainvaltest = trainvaltest
        self.ae_training = ae_training
        self.n_sequence = n_sequence
        self.sequence_mode = sequence_mode
        self.stride = stride
        self.root_dir = root_dir
        self.inference_mode = inference_mode

        # Load CSV containing the train/val/test split info.
        self.df = pd.read_csv(os.path.join(root_dir, 'df_normalized.csv'))
        self.df = self.df[self.df['trainvaltest'].isin(trainvaltest)]
        
        # Process patients and sequences
        self._process_patients()
        if not self.ae_training:
            self._precompute_sequences()

    def _process_patients(self):
        """Extract and filter patients based on the number of visits."""
        # Assume the 'id' column contains strings like "subject_s<number>"
        # We extract the number for sorting and filtering.
        self.patients = np.unique(
            np.array(self.df['id']
                     .apply(ast.literal_eval)
                     .str[0]
                     .str.extract(r'subject_s(\d+)')[0]
                     .astype(int))
        )
        # Take all available patients if self.num_patients is 
        # greater than the length of self.patients
        
        if self.num_patients > len(self.patients):
            self.patients = self.patients
        else:
            self.patients = self.patients[:self.num_patients]


        # Keep only those patients with at least n_sequence visits.
        self.valid_patients = []
        for pid in self.patients:
            patient_mask = self.df['id'].str.contains(f'subject_s{pid}')
            if patient_mask.sum() >= self.n_sequence:
                self.valid_patients.append(pid)
        
        print(f"Found {len(self.valid_patients)} patients with at least {self.n_sequence} visits.")

    def _get_patient_data(self, pid):
        """Get sorted patient data for the given patient based on normalized time."""
        patient_df = self.df[self.df['id'].str.contains(f'subject_s{pid}')][:self.num_visits]
        # print('patient_df',[s[-15:] for s in patient_df['path']])
        return patient_df.sort_values('n_time')

    def _precompute_sequences(self):
        """Generate sequences from patient visits based on n_sequence and sequence_mode."""
        self.sequences = []
        
        for pid in self.valid_patients:
            patient_df = self._get_patient_data(pid)
            visits = patient_df  # Use all available visits (sorted by time)
            times = visits['n_time'].to_numpy()

            if len(visits) < self.n_sequence:
                continue

            if self.sequence_mode == "contiguous":
                # Create sequences using a sliding window with the specified stride.
                for i in range(0, len(visits) - self.n_sequence + 1, self.stride):
                    seq_visits = visits.iloc[i:i+self.n_sequence].reset_index(drop=True)
                    self.sequences.append({
                        'pid': pid,
                        'visits': seq_visits,
                        'times': times[i:i+self.n_sequence],
                        'type': 'forward'  # Mark as forward sequence
                    })
            elif self.sequence_mode == "all_combinations":
                # Create all combinations of visits with increasing order.
                indices = list(range(len(visits)))
                for comb in itertools.combinations(indices, self.n_sequence):
                    comb = sorted(comb) # Ensure increasing order
                    seq_visits = visits.iloc[list(comb)].reset_index(drop=True)
                    t = np.array([times[i] for i in comb])
                    self.sequences.append({
                        'pid': pid,
                        'visits': seq_visits,
                        'times': t,
                        'type': 'forward'  # Mark as forward sequence
                    })
            else:
                raise ValueError("Invalid sequence_mode. Use 'contiguous' or 'all_combinations'.")
                
        # Create augmented sequences
        self._create_augmented_sequences()
        
    def _create_augmented_sequences(self):
        """Create backward and static sequences for data augmentation."""
        # Start with all original sequences
        original_sequences = self.sequences.copy()
        random.shuffle(original_sequences)  # Shuffle to randomize the partitioning
        
        # Calculate partition sizes (45% forward, 45% backward, 20% static)
        total_size = len(original_sequences)
        forward_size = int(total_size * 0.45)
        backward_size = int(total_size * 0.45)
        static_size = total_size - forward_size - backward_size  # Calculate remainder to ensure we use all sequences
        
        # Partition the original sequences
        forward_sequences = original_sequences[:forward_size]  # First 45% stay as forward
        backward_candidates = original_sequences[forward_size:forward_size+backward_size]  # Next 45% become backward
        static_candidates = original_sequences[int(total_size * 0.70):]  # Remaining 10% become static
        
        # Process backward sequences
        backward_sequences = []
        for seq in backward_candidates:
            reversed_visits = seq['visits'].iloc[::-1].reset_index(drop=True)
            backward_sequences.append({
                'pid': seq['pid'],
                'visits': reversed_visits,
                'times': seq['times'],  # Keep original times
                'type': 'backward'  # Mark as backward sequence
            })
        
        # Process static sequences
        static_sequences = []
        for seq in static_candidates:
            random_pos = random.randint(0, self.n_sequence - 1)
            static_visit = pd.DataFrame([seq['visits'].iloc[random_pos]] * self.n_sequence)
            static_sequences.append({
                'pid': seq['pid'],
                'visits': static_visit,
                'times': seq['times'],  # Keep original times
                'type': 'static'  # Mark as static sequence
            })
        
        # Mark forward sequences explicitly as forward
        for seq in forward_sequences:
            seq['type'] = 'forward'
        
        # Combine all sequences
        self.sequences = forward_sequences + backward_sequences + static_sequences
        
        # Shuffle the final sequence list for randomness
        random.shuffle(self.sequences)
        
        print(f"Dataset composition: {len(forward_sequences)} forward, {len(backward_sequences)} backward, {len(static_sequences)} static sequences")

    def __len__(self):
        if self.ae_training:
            return len(self.df)
        else:
            return len(self.sequences)

    def _load_image(self, path):
        """Load image stored as a numpy array file and return as a torch tensor.
           Adjust this function as needed if your image format is different."""
        img = np.load(os.path.join(self.root_dir, path))
        return torch.tensor(img).unsqueeze(0).float()

    def __getitem__(self, idx):
        if self.ae_training:
            return self._ae_item(idx)
        else:
            return self._clinical_item(idx)

    def _ae_item(self, idx):
        """Return a single image sample for autoencoder training."""
        path = self.df.iloc[idx]['path']
        return {'images': self._load_image(path)}

    def _clinical_item(self, idx):
        """Return a clinical sequence sample.
           The sample contains the full sequence of images and their normalized times."""
        seq = self.sequences[idx]
        visits = seq['visits']
        # Load all images in the sequence.
        images = torch.stack([self._load_image(p) for p in visits['path']])
        times = torch.tensor(seq['times'], dtype=torch.float32)
        return {
            'pid': seq['pid'],
            'images': images,   # shape: (n_sequence, channels, H, W)
            'times': times,     # shape: (n_sequence,)
            'type': seq['type'] # sequence type: 'forward', 'backward', or 'static'
        }
    
    @property
    def dims(self):
        # Update this property as needed.
        # For example, if each image is 1 x 128 x 128 and we return a sequence of n_sequence images,
        # you might want to return (n_sequence, 1, 128, 128).
        return (self.n_sequence, 1, 128, 128)


# def collate_fn(batch):
#     if 'images' in batch[0]:
#         return {
#             'times': torch.stack([item['times'] for item in batch]),
#             'images': torch.stack([item['images'] for item in batch]),
            
#         }
#     else:
#         return {
#             'images': torch.stack([item['image'] for item in batch]).unsqueeze(1),
#         }



def create_dataloaders(image_size=(64, 64), batch_size=16, n_sequences=4, 
                      num_workers=0,num_patients=500,sequence_mode="contiguous",
                      stride=1,modes="sequence"): 
    

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = StarmenDataset(
        n_sequence=n_sequences if split in ["train","val"] else 10,
        num_patients=num_patients,
        trainvaltest=[split], 
        ae_training=True if modes == "autoencoder" else False,
        sequence_mode=sequence_mode, 
        stride=stride,
    )
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, )
    test_loader = DataLoader(datasets['test'], batch_size=1,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    # Test the dataset
    train_loader, val_loader, test_loader=create_dataloaders(batch_size=16)
    data=next(iter(train_loader))
    # print(data['pid'])
    print(data['images'].shape)
    print(data['times'])