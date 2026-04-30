'''
A longitudinal brain Glioblastoma dataset.
"The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation"
'''


import copy
import itertools
import random
from typing import Literal
from glob import glob
from typing import List, Tuple

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import ipdb
import pandas as pd
import torch
from utils.general_utils import *
ROOT = str(Path(__file__).resolve().parent.parent.parent)


SEX_ENCODING = {
    'M': torch.tensor([1, 0], dtype=torch.float32),  # Male
    'F': torch.tensor([0, 1], dtype=torch.float32)   # Female
}



def normalize_image(image: np.array) -> np.array:
    '''
    Image already normalized on scan level.
    Just transform to [-1, 1] and clipped to [-1, 1].
    '''
    assert image.min() >= 0 and image.max() <= 255
    image = image / 255.0 * 2 - 1
    image = np.clip(image, -1.0, 1.0)
    return image

def normalize_image_zero_to_one(image: np.array) -> np.array:
    '''
    Normalize an image to the range [0, 1] and clip values to [0, 1].
    '''
    assert image.min() >= 0 and image.max() <= 255, "Image values should be in the range [0, 255]."
    image = image / 255.0  # Normalize to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    return image



def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    if normalize:
        image = normalize_image(image)

    return image

def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    # Add the channel dimension to comply with Torch.
    array = array[None, :, :]
    return array

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = os.path.basename(path).replace('week_', '').split('-')[0].replace('.png', '')
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time

def sample_N_slices_from_a_path_new(path, N=3, min_gap=2, max_gap=5, equal_gaps=True):
    """Helper function to sample N slices with specified gaps"""
    # Implementation needed - this seems to be used but was missing from the provided code
    pass

def sample_slices_from_ref(path, N=3):
    """Helper function to sample slices from reference"""
    # Implementation needed - this seems to be used but was missing from the provided code
    pass


class BrainGBMDataset(Dataset):

    def __init__(self,
                 base_path: str = ROOT + '/data/brain_LUMIERE/',
                 image_folder: str = '',
                 max_slice_per_patient: int = 200,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - When a patient's index is queried, we return images from all visits of that patient.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.

        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.

        NOTE: This dataset is structured like this:
        LUMIERE_images_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ.png

        LUMIERE_masks_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ_GBM_mask.png

        So we will organize outputs on the unit of slices.
        Each slice is essentially treated as a separate trajectory.
        But importantly, data partitioning is done on the unit of patients.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient
        self.all_patient_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))
        self.all_patient_ids = [os.path.basename(item.rstrip('/')) for item in self.all_patient_folders]
        self.patient_id_to_slice_id = []  # maps the patient id to a list of corresponding slice ids.
        self.image_by_slice = []
        self.max_t = 0


        curr_slice_idx = 0
        for folder in self.all_patient_folders:

            num_slices_curr_patient = 0
            slice_arr = np.array(sorted(glob('%s/slice*/' % (folder))))
            np.random.shuffle(slice_arr)
            #max_slice is how many slice you want to use from each patients
            if self.max_slice_per_patient is not None \
                and len(slice_arr) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(slice_arr)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                slice_arr = slice_arr[subset_ids]

            for curr_slice in slice_arr:
                paths = sorted(glob('%s/week*.png' % curr_slice))

                '''
                Ignore week 0!!!
                Week 0 is pre-operation, which means tumors will be cut!
                This dynamics may be too complicated to learn.
                If we ignore week 0, the remaining will likely be natural growth of tumor.
                '''
                paths = [p for p in paths if 'week_000' not in p]

                if len(paths) >= 2:
                    self.image_by_slice.append(paths)
                    num_slices_curr_patient += 1
                for p in paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(np.arange(curr_slice_idx, curr_slice_idx + num_slices_curr_patient))
            curr_slice_idx += num_slices_curr_patient


    def return_statistics(self) -> None:
        print('max time (weeks):', self.max_t)

        unique_patient_list = np.unique(self.all_patient_ids)
        print('Number of unique patients:', len(unique_patient_list))
        print('Number of unique slices:', len(self.image_by_slice))

        num_visit_map = {}
        for item in self.image_by_slice:
            num_visit = len(item)
            if num_visit not in num_visit_map.keys():
                num_visit_map[num_visit] = 1
            else:
                num_visit_map[num_visit] += 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d slices.' % (k, v))
        return

    def __len__(self) -> int:
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainGBMSequenceDataset(BrainGBMDataset):
    """
    A specialized dataset for handling sequence data formats:
    - n_len_subsequences: sequences of specific length
    - full_sequence: complete patient sequences
    """

    def __init__(self,
                 main_dataset: BrainGBMDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['n_len_subsequences', 'full_sequence'],
                 transforms = None,
                 normalize_0to1 = False,
                 subsequence_length = 4,
                 sequence_mode='all_combinations'):
        """
        Initialize a dataset that focuses on sequence data.
        
        Args:
            main_dataset: The parent dataset to extract sequences from
            subset_indices: Which indices to use from the main dataset
            return_format: Either 'n_len_subsequences' or 'full_sequence'
            transforms: Data transforms to apply
            normalize_0to1: Whether to normalize to [0,1] range
            subsequence_length: Length of subsequences to extract
            sequence_mode: How to extract sequences ('contiguous' or 'all_combinations')
        """
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.subsequence_length = subsequence_length
        self.sequence_mode = sequence_mode
        self.rano_grade = pd.read_csv(ROOT + '/ImageFlowNet/data/LUMIERE_RANO.csv')
        self.mr_info = pd.read_csv(ROOT + '/ImageFlowNet/data/LUMIERE-MRinfo_filled.csv')
        
        if normalize_0to1:
            self.normalize_image = normalize_image_zero_to_one
        else:
            self.normalize_image = normalize_image
            
        # Extract image sequences from the main dataset based on subset indices
        self.image_by_slice = []
        for patient_id in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[patient_id]
            self.image_by_slice.extend([main_dataset.image_by_slice[i] for i in slice_ids])
        
        # Initialize data structures for sequence handling
        self.all_subsequences = []
        self.n_len_subsequnces = []
        self.all_consecutive_subsequences = []
        
        # For improved batching in full_sequence mode - group by sequence length
        self.sequence_length_groups = {}
        
        # Process each image list to generate sequences
        for i, image_list in enumerate(self.image_by_slice):
            # Group sequences by length for efficient batching
            seq_len = len(image_list)
            if seq_len not in self.sequence_length_groups:
                self.sequence_length_groups[seq_len] = []
            self.sequence_length_groups[seq_len].append(i)
            
            # For subsequences of specific length
            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                subsequence_indices_list = sorted(subsequence_indices_list)
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])
            
            # For contiguous subsequences
            for num_items in range(2, len(image_list) + 1):
                for start_idx in range(0, len(image_list) - num_items + 1):
                    subsequence = image_list[start_idx : start_idx + num_items]
                    self.all_consecutive_subsequences.append(subsequence)
        
        # Filter subsequences based on sequence mode and length
        if self.sequence_mode == 'contiguous':
            if return_format == 'n_len_subsequences' and subsequence_length is not None:
                self.n_len_subsequnces = [seq for seq in self.all_consecutive_subsequences if len(seq) == self.subsequence_length]
        else:
            if return_format == 'n_len_subsequences' and subsequence_length is not None:
                self.n_len_subsequnces = [seq for seq in self.all_subsequences if len(seq) == self.subsequence_length]
        
        print(f'Sequence mode: {self.sequence_mode}')
        if return_format == 'n_len_subsequences':
            print(f'Number of Subsequences of len {subsequence_length} :', len(self.n_len_subsequnces))
        elif return_format == 'full_sequence':
            print(f'Sequence length distribution:')
            for length, indices in sorted(self.sequence_length_groups.items()):
                print(f'  Length {length}: {len(indices)} sequences')

    def rano_label_to_onehot(self, rano_label):
        """
        Map a RANO label to a 4-dimensional one-hot vector:
        0: Progress   (PD)
        1: No-Progress (SD, Post-Op)
        2: Recovery   (PR, CR)
        3: None/Unknown (Pre-Op or any unrecognized label)
        """
        mapping_dict = {
            "PD":       0,   # Progress
            "SD":       1,   # No-Progress
            "Post-Op":  1,   # No-Progress
            "PR":       2,   # Recovery
            "CR":       2    # Recovery
            # note: we do NOT map Pre-Op here, so it defaults to 3
        }

        # Determine the class index (default to 3 for 'None'/'Unknown')
        class_index = mapping_dict.get(rano_label, 3)

        # Create a one-hot vector of size 4
        one_hot = torch.zeros(4)
        one_hot[class_index] = 1.0

        return one_hot

    def get_info(self, p, week):
        match = self.rano_grade[(self.rano_grade['Patient']==p) & (self.rano_grade['Date']==week)]  
        grade = match['Rating'].values[0] if len(match)>0 else 'Unknown'
        match_mr_info = self.mr_info[(self.mr_info['Patient']==p) & (self.mr_info['Timepoint']==week)]

        age = match_mr_info['AgeFilled'].values[0] if len(match_mr_info)>0 else 0.0
        sex_str= match_mr_info['Sex'].values[0].strip()if len(match_mr_info)>0 else 'unknown'
        sex_enc = SEX_ENCODING.get(sex_str, torch.tensor([0, 0], dtype=torch.float32))
        return self.rano_label_to_onehot(grade),age,sex_enc

    def __len__(self) -> int:
        if self.return_format == 'full_sequence':
            return len(self.image_by_slice)
        elif self.return_format == 'n_len_subsequences':
            return len(self.n_len_subsequnces)
        else:
            raise ValueError(f"Unsupported return format: {self.return_format}")

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'full_sequence':
            queried_sequence = self.image_by_slice[idx]
            p = queried_sequence[-1].split('/')[-3]
            weeks = [queried_sequence[i].split('/')[-1].replace('.png','').replace('_','-') for i in range(len(queried_sequence))]
            # Get all info items and unzip them into separate lists
            info_items = [self.get_info(p, week) for week in weeks]
            grade_hots, ages, sexes = zip(*info_items)  # Unzip the list of tuples into separate lists

            # Stack each list separately into tensors
            grade_hot = torch.stack(list(grade_hots))
            age = torch.tensor(list(ages))
            sex = torch.stack(list(sexes))
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])
            
        elif self.return_format == 'n_len_subsequences':
            queried_sequence = self.n_len_subsequnces[idx]
            tp = queried_sequence[-1].split('/')[-1].replace('.png','').replace('_','-')
            p = queried_sequence[-1].split('/')[-3]
            grade_hot,age,sex = self.get_info(p, tp)
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])
        else:
            raise ValueError(f"Unsupported return format: {self.return_format}")

        # Process the images
        num_images = len(images)
        assert num_images >= 2
        assert num_images < 20  # NOTE: see `additional_targets` in `transform`.

        # Unpack the subsequence
        image_list = np.rollaxis(images, axis=0)

        data_dict = {'image': image_list[0]}
        for idx in range(num_images - 1):
            data_dict['image_other%d' % (idx + 1)] = image_list[idx + 1]

        if self.transforms is not None:
            data_dict = self.transforms(**data_dict)

        images = self.normalize_image(add_channel_dim(data_dict['image']))[None, ...]
        for idx in range(num_images - 1):
            images = np.vstack((images,
                                self.normalize_image(add_channel_dim(data_dict['image_other%d' % (idx + 1)]))[None, ...]))
                                
        slice_n = physical_dist = 0
        return images, timestamps, torch.tensor(slice_n), torch.tensor(physical_dist), grade_hot,age,sex

    def get_batch_sampler(self, batch_size=4):
        """
        Returns a batch sampler that groups sequences by length for efficient batching.
        
        Args:
            batch_size: Number of sequences per batch
            
        Returns:
            A BatchSampler object that can be passed to DataLoader
        """
        if self.return_format != 'full_sequence':
            raise ValueError("get_batch_sampler is only available for 'full_sequence' mode")
            
        return BrainGBMSequenceBatchSampler(self.sequence_length_groups, batch_size)
    
    def get_sequence_lengths(self):
        """Returns a list of available sequence lengths"""
        return sorted(self.sequence_length_groups.keys())
    
    def get_sequences_of_length(self, length, max_samples=None):
        """
        Returns indices of sequences with the specified length.
        
        Args:
            length: The sequence length to filter by
            max_samples: If specified, return at most this many randomly sampled indices
            
        Returns:
            List of dataset indices
        """
        if length not in self.sequence_length_groups:
            return []
            
        indices = self.sequence_length_groups[length]
        if max_samples is not None and max_samples < len(indices):
            return random.sample(indices, max_samples)
        return indices


class BrainGBMSequenceBatchSampler:
    """
    Custom batch sampler that groups sequences by length.
    This ensures all sequences in a batch have the same length.
    """
    
    def __init__(self, sequence_length_groups, batch_size):
        """
        Initialize the sampler.
        
        Args:
            sequence_length_groups: Dictionary mapping sequence lengths to indices
            batch_size: Number of sequences per batch
        """
        self.sequence_length_groups = sequence_length_groups
        self.batch_size = batch_size
        self.length_keys = list(self.sequence_length_groups.keys())
        
    def __iter__(self):
        # Make a copy of the groups to avoid modifying the original
        available_indices = {k: v.copy() for k, v in self.sequence_length_groups.items()}
        
        # Shuffle the indices within each length group
        for indices in available_indices.values():
            random.shuffle(indices)
        
        # Create batches by selecting from each length group
        batches = []
        
        # Randomize the order of length groups
        length_keys = self.length_keys.copy()
        random.shuffle(length_keys)
        
        for length in length_keys:
            indices = available_indices[length]
            # Create as many complete batches as possible from this length group
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:  # Only use complete batches
                    batches.append(batch)
                
        # Shuffle the order of batches
        random.shuffle(batches)
        
        # Yield each batch
        for batch in batches:
            yield batch
    
    def __len__(self):
        # Count how many complete batches we can create
        count = 0
        for indices in self.sequence_length_groups.values():
            count += len(indices) // self.batch_size
        return count


class BrainGBMSubset(BrainGBMDataset):

    def __init__(self,
                 main_dataset: BrainGBMDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['all_single','one_pair', 'all_pairs', 'all_subsequences', 'all_subarrays', 'triplet'],
                 transforms = None,
                 transforms_aug = None,
                 normalize_0to1 = False,
                 subsequence_length = 4,
                 load_in_triplet = False,
                 sequence_mode='all_combinations'):
        '''
        A subset of BrainGBMDataset.
        
        Note: For 'n_len_subsequences' or 'full_sequence' formats,
        use the BrainGBMSequenceDataset class instead.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug
        self.subsequence_length = subsequence_length
        self.load_in = load_in_triplet
        self.sequence_mode = sequence_mode
        self.rano_grade= pd.read_csv(ROOT + '/ImageFlowNet/data/LUMIERE_RANO.csv')
        if normalize_0to1:
            self.normalize_image = normalize_image_zero_to_one
        else:
            self.normalize_image = normalize_image
            
        self.image_by_slice = []
        # Patient_id to available slice_id 
        # then to available time point for each slice  
        for patient_id in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[patient_id]
            self.image_by_slice.extend([main_dataset.image_by_slice[i] for i in slice_ids])

        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []
        self.all_single_images = [] # this is to pretrain the AE
        self.n_len_subsequnces = []
        self.all_consecutive_subsequences = []
        # Get the pair-wise list
        for image_list in self.image_by_slice:
            self.all_single_images.extend(image_list)
            pair_indices = list(itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])
                self.all_subarrays.append(image_list[idx1 : idx2+1])
            # Get sequence of all possible lengths  
            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                subsequence_indices_list = sorted(subsequence_indices_list)
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])
            for num_items in range(2, len(image_list) + 1):
                for start_idx in range(0, len(image_list) - num_items + 1):
                    subsequence = image_list[start_idx : start_idx + num_items]
                    self.all_consecutive_subsequences.append(subsequence)
        if self.sequence_mode=='contiguous':
            if return_format=='n_len_subsequences' and subsequence_length is not None:
                self.n_len_subsequnces= [seq for seq in self.all_consecutive_subsequences if len(seq) ==  self.subsequence_length]
        else:
            if return_format=='n_len_subsequences' and subsequence_length is not None:
                self.n_len_subsequnces= [seq for seq in self.all_subsequences if len(seq) ==  self.subsequence_length]
        print(f'Sequence mode: {self.sequence_mode}')
        print(f'Number of Subsequences of len {subsequence_length} :', len( self.n_len_subsequnces))
    
    def rano_label_to_onehot(self,rano_label):
        """
        Map a RANO label to a 4-dimensional one-hot vector:
        0: Progress   (PD)
        1: No-Progress (SD, Post-Op)
        2: Recovery   (PR, CR)
        3: None/Unknown (Pre-Op or any unrecognized label)
        
        Examples:
        rano_label_to_onehot_4class("PD")      -> tensor([1., 0., 0., 0.])
        rano_label_to_onehot_4class("SD")      -> tensor([0., 1., 0., 0.])
        rano_label_to_onehot_4class("CR")      -> tensor([0., 0., 1., 0.])
        rano_label_to_onehot_4class("Pre-Op")  -> tensor([0., 0., 0., 1.])
        rano_label_to_onehot_4class("Unknown") -> tensor([0., 0., 0., 1.])
        """
        mapping_dict = {
            "PD":       0,   # Progress
            "SD":       1,   # No-Progress
            "Post-Op":  1,   # No-Progress
            "PR":       2,   # Recovery
            "CR":       2    # Recovery
            # note: we do NOT map Pre-Op here, so it defaults to 3
        }

        # Determine the class index (default to 3 for 'None'/'Unknown')
        class_index = mapping_dict.get(rano_label, 3)

        # Create a one-hot vector of size 4
        one_hot = torch.zeros(4)
        one_hot[class_index] = 1.0

        return one_hot

    def get_info(self,p,week):
        match=self.rano_grade[(self.rano_grade['Patient']==p) & (self.rano_grade['Date']==week)]  
        grade = match['Rating'].values[0] if len(match)>0 else 'Unknown'
        return self.rano_label_to_onehot(grade)

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_slice)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)
        elif self.return_format == 'all_subsequences':
            # If we return all subsequences of images per patient...
            return len(self.all_subsequences)
        elif self.return_format == 'all_subarrays':
            # If we return all subarrays of images per patient...
            return len(self.all_subarrays)
        elif self.return_format == 'full_sequence':
            # If we return the full sequences of images per patient...
            return len(self.image_by_slice)
        elif self.return_format == 'all_single':
            # If we return all single images per patient...
            return len(self.all_single_images)
        elif self.return_format == 'n_len_subsequences':
            return len(self.n_len_subsequnces)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_slice[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))

            sampled_pair = [
                image_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in sampled_pair
            ])
            timestamps = np.array([get_time(img) for img in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_pair
            ])
            timestamps = np.array([get_time(img) for img in queried_pair])

        elif self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'all_subarrays':
            queried_sequence = self.all_subarrays[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'full_sequence':
            queried_sequence = self.image_by_slice[idx]
            p=queried_sequence[-1].split('/')[-3]
            weeks=[queried_sequence[i].split('/')[-1].replace('.png','').replace('_','-') for i in range(len(queried_sequence))]
            grade_hot= torch.stack([self.get_info(p,week) for week in weeks])
            ipdb.set_trace()
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])
            

        elif self.return_format == 'all_single':
            queried_sequence = self.all_single_images[idx]
           
            if self.load_in=='triplet':
                slice_n,paths,physical_dist = sample_N_slices_from_a_path_new(queried_sequence,N=3,min_gap=2,
                                                                max_gap=5,equal_gaps=True)
                images = np.array([
                    load_image(img, target_dim=self.target_dim, normalize=False) for img in paths
                ])
                timestamps = np.array([get_time(img) for img in paths])
            elif self.load_in=='n_slices':
                N = 4
                if self.transforms is not None:
                   N =  random.randint(3,7)   
                slice_n,paths = sample_slices_from_ref(queried_sequence,N=N)
                images = np.array([
                    load_image(img, target_dim=self.target_dim, normalize=False) for img in paths
                ])
                timestamps = np.array([get_time(img) for img in [queried_sequence]])
                physical_dist=0
            else:
                images = np.array([
                    load_image(img, target_dim=self.target_dim, normalize=False) for img in [queried_sequence]
                ])
                timestamps = np.array([get_time(img) for img in [queried_sequence]])
                paths = queried_sequence
                slice_n=physical_dist= 0

        elif self.return_format == 'n_len_subsequences':
            queried_sequence = self.n_len_subsequnces[idx]
            tp=queried_sequence[-1].split('/')[-1].replace('.png','').replace('_','-')
            p=queried_sequence[-1].split('/')[-3]
            match=self.rano_grade[(self.rano_grade['Patient']==p) & (self.rano_grade['Date']==tp)]
            grade = match['Rating'].values[0] if len(match)>0 else 'Unknown'
            grade_hot = self.rano_label_to_onehot(grade)
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        if self.return_format in ['one_pair', 'all_pairs']:
            assert len(images) == 2
            image1, image2 = images[0], images[1]
            if self.transforms is not None:
                transformed = self.transforms(image=image1, image_other=image2)
                image1 = transformed["image"]
                image2 = transformed["image_other"]

            if self.transforms_aug is not None:
                transformed_aug = self.transforms_aug(image=image1, image_other=image1)
                image1_aug = transformed_aug["image"]
                image1_aug = self.normalize_image(image1_aug)
                image1_aug = add_channel_dim(image1_aug)

            image1 = self.normalize_image(image1)
            image2 = self.normalize_image(image2)

            image1 = add_channel_dim(image1)
            image2 = add_channel_dim(image2)

            if self.transforms_aug is not None:
                images = np.vstack((image1[None, ...], image2[None, ...], image1_aug[None, ...]))
            else:
                images = np.vstack((image1[None, ...], image2[None, ...]))
        
        elif self.return_format in ['all_single']:
       
            batch = images
            if self.transforms is not None:
                if self.load_in == 'triplet':
                    tmp= self.transforms(image=batch[0],image1=batch[1],image2=batch[2])
                    batch=np.stack([tmp['image'],tmp['image1'],tmp['image2']])
                elif self.load_in == 'n_slices':
                    raise NotImplementedError
                else:
                    batch= self.transforms(image=batch)['image']
            images = np.stack([self.normalize_image(im) for im in batch])
     
        elif self.return_format in ['all_subsequences', 'all_subarrays', 'full_sequence','n_len_subsequences']:
            num_images = len(images)
            assert num_images >= 2
            assert num_images < 20  # NOTE: see `additional_targets` in `transform`.

            # Unpack the subsequence.
            image_list = np.rollaxis(images, axis=0)

            data_dict = {'image': image_list[0]}
            for idx in range(num_images - 1):
                data_dict['image_other%d' % (idx + 1)] = image_list[idx + 1]

            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)

            images = self.normalize_image(add_channel_dim(data_dict['image']))[None, ...]
            for idx in range(num_images - 1):
                images = np.vstack((images,
                                    self.normalize_image(add_channel_dim(data_dict['image_other%d' % (idx + 1)]))[None, ...]))
            slice_n=physical_dist= 0
        return images, timestamps, torch.tensor(slice_n), torch.tensor(physical_dist),grade_hot


class BrainGBMSegDataset(Dataset):

    def __init__(self,
                 base_path: str = ROOT + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 mask_folder: str = 'LUMIERE_masks_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        This dataset is for segmentation.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        all_patient_folders = sorted(glob('%s/%s/Patient-*/' % (base_path, image_folder)))

        self.image_by_patient = []
        self.mask_by_patient = []

        for patient_folder in all_patient_folders:
            curr_patient_slice_folders = np.array(sorted(glob('%s/slice*/' % patient_folder)))

            if self.max_slice_per_patient is not None \
                and len(curr_patient_slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(curr_patient_slice_folders)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                curr_patient_slice_folders = curr_patient_slice_folders[subset_ids]

            for im_folder in curr_patient_slice_folders:
                image_paths = sorted(glob('%s/*.png' % im_folder))
                mask_paths = []
                for image_path_ in image_paths:
                    mask_path_ = image_path_.replace('.png', '').replace(
                        image_folder, mask_folder) + '_GBM_mask.png'
                    assert os.path.isfile(mask_path_)
                    mask_paths.append(mask_path_)
                self.image_by_patient.append(image_paths)
                self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainGBMSegSubset(BrainGBMSegDataset):

    def __init__(self,
                 main_dataset: BrainGBMSegDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of BrainGBMSegDataset.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim

        image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]
        mask_by_patient = [
            main_dataset.mask_by_patient[i] for i in subset_indices
        ]

        self.image_list = [image for patient_folder in image_by_patient for image in patient_folder]
        self.mask_list = [mask for patient_folder in mask_by_patient for mask in patient_folder]
        assert len(self.image_list) == len(self.mask_list)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = load_image(self.image_list[idx], target_dim=self.target_dim, normalize=False)
        mask = load_image(self.mask_list[idx], target_dim=self.target_dim, normalize=False)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = self.normalize_image(image)

        # I believe this means necrosis and contrast enhancement.
        # necrosis: 85, contrast enhancement: 170, edema: 255.
        assert mask.min() == 0 and mask.max() <= 255
        mask = np.logical_and(mask > 0, mask < 250)

        image = add_channel_dim(image)
        mask = add_channel_dim(mask)

        return image, mask






if __name__ == '__main__':
    print('Full set.')
    dataset = BrainGBMDataset(max_slice_per_patient=None)
    dataset.return_statistics()

    print('Subset with max 20 slices per patient.')
    dataset = BrainGBMDataset(max_slice_per_patient=20)
    dataset.return_statistics()
