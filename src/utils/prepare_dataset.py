from utils.extend import ExtendedDataset
from utils.split import split_indices

from datasets.brain_ms import BrainMSDataset, BrainMSSubset, BrainMSSegDataset, BrainMSSegSubset, brain_MS_split
from datasets.brain_gbm import BrainGBMDataset, BrainGBMSubset, BrainGBMSegDataset, BrainGBMSegSubset, BrainGBMSequenceDataset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset_AE(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from one earlier timepoint.
    '''
    if config.dataset_name == 'mario':
        dataset=MarioDataset(target_dim=config.image_size,only_unhealthy=config.use_only_pathological_case)
        Subset=MarioSubset
        train_indices, val_indices, test_indices = dataset.get_split()
    elif config.dataset_name == 'retina_areds':
        dataset = RetinaAREDSDataset(eye_mask_folder=config.eye_mask_folder)
        Subset = RetinaAREDSSubset

    elif config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.image_size,
                                  base_path=config.dataset_path,
                                  image_folder=config.image_folder,
                                  max_slice_per_patient=config.max_slice_per_patient)
        Subset = BrainGBMSubset

    elif config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                   image_folder=config.image_folder,
                                   target_dim=config.target_dim)
        Subset = SyntheticSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    if config.dataset_name != 'mario':
        # Load into DataLoader
        ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        indices = list(range(len(dataset)))
        train_indices, val_indices, test_indices = \
            split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_aug = None
    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_single',
                       transforms=transforms_train,
                       transforms_aug=transforms_aug,
                       normalize_0to1=config.normalize_zero_to_one,
                       load_in_triplet=config.load_in_triplet)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_single',
                     transforms=transforms_val,
                     normalize_0to1=config.normalize_zero_to_one,
                     load_in_triplet=config.load_in_triplet)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_single',
                      transforms=transforms_test,
                      normalize_0to1=config.normalize_zero_to_one,
                      load_in_triplet=config.load_in_triplet)

    min_sample_per_epoch = 5

    if  config.max_training_samples is not None:
        min_sample_per_epoch = config.max_training_samples

    # desired_len = max(len(train_set), min_sample_per_epoch)
    # desired_len = len(train_set)
    # train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t,train_indices,val_indices,test_indices


def prepare_dataset(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from one earlier timepoint.
    '''
    
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.image_size,base_path=config.dataset_path,image_folder=config.image_folder)
        Subset = BrainGBMSubset
    
    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    if config.dataset_name != 'mario':
        # Load into DataLoader
        ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        indices = list(range(len(dataset)))
        train_indices, val_indices, test_indices = \
            split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_aug = None
    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='one_pair',
                       transforms=transforms_train,
                       transforms_aug=transforms_aug)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='one_pair',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='one_pair',
                      transforms=transforms_test)

    min_sample_per_epoch = 5

    if  config.max_training_samples is not None:
        min_sample_per_epoch = config.max_training_samples

    # desired_len = max(len(train_set), min_sample_per_epoch)
    # train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t,train_indices,val_indices,test_indices


def prepare_dataset_npt(config: AttributeHashmap, transforms_list = [None, None, None],
                        batch_size=3, train_seq_len=4, test_seq_len=4, 
                        sequence_mode='contiguous', return_format='full_sequence'):
    '''
    Prepare the dataset for predicting one future timepoint from potentially multiple earlier timepoints.
    
    Args:
        config: Configuration parameters
        transforms_list: List of transforms for train/val/test sets
        batch_size: Batch size for dataloader
        train_seq_len: Sequence length for training (used for n_len_subsequences)
        test_seq_len: Sequence length for testing (used for n_len_subsequences)
        sequence_mode: How to sample sequences ('contiguous' or 'all_combinations')
        return_format: Dataset return format ('n_len_subsequences' or 'full_sequence')
    '''
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.image_size, base_path=config.dataset_path,
                                  image_folder=config.image_folder, max_slice_per_patient=config.max_slice_per_patient)
        
        # Use different dataset classes based on return format
        if return_format in ['n_len_subsequences', 'full_sequence']:
            SequenceDataset = BrainGBMSequenceDataset
        else:
            Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    
    # Create datasets based on return_format
    if config.dataset_name == 'brain_gbm' and return_format in ['n_len_subsequences', 'full_sequence']:
        train_set = SequenceDataset(
            main_dataset=dataset,
            subset_indices=train_indices,
            return_format=return_format,
            transforms=transforms_train,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=train_seq_len,
            sequence_mode=sequence_mode
        )
        
        val_set = SequenceDataset(
            main_dataset=dataset,
            subset_indices=val_indices,
            return_format=return_format,
            transforms=transforms_val,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=train_seq_len,
            sequence_mode='contiguous'
        )
        
        test_set = SequenceDataset(
            main_dataset=dataset,
            subset_indices=test_indices,
            return_format=return_format,
            transforms=transforms_test,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=test_seq_len,
            sequence_mode='contiguous'
        )
    else:
        train_set = Subset(
            main_dataset=dataset,
            subset_indices=train_indices,
            return_format=return_format,
            transforms=transforms_train,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=train_seq_len,
            sequence_mode=sequence_mode
        )
        
        val_set = Subset(
            main_dataset=dataset,
            subset_indices=val_indices,
            return_format=return_format,
            transforms=transforms_val,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=train_seq_len,
            sequence_mode='contiguous'
        )
        
        test_set = Subset(
            main_dataset=dataset,
            subset_indices=test_indices,
            return_format=return_format,
            transforms=transforms_test,
            normalize_0to1=config.normalize_zero_to_one,
            subsequence_length=test_seq_len,
            sequence_mode='contiguous'
        )

    # Handle data extension if needed
    min_sample_per_epoch = config.max_training_samples
    desired_len = min(len(train_set), min_sample_per_epoch)
    train_set_extended = ExtendedDataset(dataset=train_set, desired_len=desired_len)
    
    # Create DataLoaders with appropriate batching strategies
    if config.dataset_name == 'brain_gbm' and return_format == 'full_sequence':
        # Use batch sampler for efficient sequence batching
        batch_sampler = train_set.get_batch_sampler(batch_size=batch_size)
        train_loader = DataLoader(
            dataset=train_set_extended,
            batch_sampler=batch_sampler,
            num_workers=config.num_workers
        )
        
        val_batch_sampler = val_set.get_batch_sampler(batch_size=batch_size)
        val_loader = DataLoader(
            dataset=val_set,
            batch_sampler=val_batch_sampler,
            num_workers=config.num_workers
        )
        
        test_batch_sampler = test_set.get_batch_sampler(batch_size=batch_size)
        test_loader = DataLoader(
            dataset=test_set,
            batch_sampler=test_batch_sampler,
            num_workers=config.num_workers
        )
    else:
        # Standard DataLoader for fixed-length sequences or other formats
        train_loader = DataLoader(
            dataset=train_set_extended,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

    return train_loader, val_loader, test_loader, dataset.num_image_channel(), dataset.max_t, train_indices, val_indices, test_indices


def prepare_dataset_specific_seq_len(config: AttributeHashmap, transforms_list = [None, None, None],npt=None):
    '''
    Prepare the dataset for predicting one future timepoint from potentially multiple earlier timepoints.
    '''
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.image_size,
                                  base_path=config.dataset_path,
                                  image_folder=config.image_folder,
                                  max_slice_per_patient=config.max_slice_per_patient)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='n_len_subsequences',
                       transforms=transforms_train,subsequence_length=npt,
                       normalize_0to1=config.normalize_zero_to_one )
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='n_len_subsequences',
                     transforms=transforms_val,subsequence_length=npt,
                     normalize_0to1=config.normalize_zero_to_one)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='n_len_subsequences',
                      transforms=transforms_test,normalize_0to1=config.normalize_zero_to_one ,subsequence_length=npt)

    min_sample_per_epoch = 5
    min_sample_per_epoch = config.max_training_samples
    desired_len = min(len(train_set), min_sample_per_epoch)
    desired_len = len(train_set)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t,train_indices,val_indices,test_indices


def prepare_dataset_full_sequence(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all full sequences.
    '''

    # Read dataset.
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    full_set = Subset(main_dataset=dataset,
                      subset_indices=indices,
                      return_format='full_sequence',
                      transforms=None)

    full_set = DataLoader(dataset=full_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return full_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_all_subarrays(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all subarrays.
    This means we will not consider subsequences with dropped out intermediate values.
    '''

    # Read dataset.
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subarrays',
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subarrays',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subarrays',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_segmentation(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'brain_ms':
        dataset = BrainMSSegDataset(target_dim=config.target_dim)
        Subset = BrainMSSegSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMSegDataset(target_dim=config.target_dim)
        Subset = BrainGBMSegSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    if config.dataset_name == 'brain_ms':
        train_indices, val_indices, test_indices = brain_MS_split()

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set) / config.batch_size, min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel()

