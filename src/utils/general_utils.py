from collections import defaultdict
from glob import glob
from itertools import combinations
import pandas as pd
import math,os,time,re,h5py,torch,tracemalloc
from typing import List, Tuple
from matplotlib import pyplot as plt
from sklearn.manifold import Isomap
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import plotly.graph_objects as go
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.sparse.csgraph import  dijkstra
import networkx as nx
import jax.numpy as jnp
from scipy.sparse import csr_matrix
import random as rand
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent.parent)

def to_np(x):
    return x.detach().cpu().numpy()


def set_seeds(seed=42):
    rand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DynamicSubsampledDataLoader:
    def __init__(self, dataset, batch_size, shuffle, collate_fn, num_workers, fraction=1.0):
        """
        Custom DataLoader wrapper for dynamic subsampling.

        Args:
            dataset (Dataset): The dataset to load.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            collate_fn (callable): Collate function for batching.
            num_workers (int): Number of subprocesses for data loading.
            fraction (float): Fraction of the dataset to use (default: 1.0).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.fraction = fraction
        self.num_samples = len(dataset)

    def _get_random_sampler(self):
        """Create a new SubsetRandomSampler with random indices."""
        subset_size = int(self.fraction * self.num_samples)
        random_indices = np.random.permutation(self.num_samples)[:subset_size]
        return SubsetRandomSampler(random_indices)

    def get_dataloader(self):
        """Create a DataLoader with the current subsampling strategy."""
        sampler = self._get_random_sampler()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Sampler handles shuffling
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) * 1.0 / (self.warmup + 1e-6)
        return lr_factor

class MemmapWrapper:
    def __init__(self, memmap_path):
        """
        Initializes the wrapper for a memory-mapped dataset.
        
        Args:
            memmap_path (str): Path to the memory-mapped file.
            chunk_size (int): Number of rows to process at a time for min/max computations.
        """
        self.path = memmap_path
        if not os.path.isfile(self.path+'.memmap'):
            raise FileNotFoundError(f"The memory-mapped file '{self.path+'.memmap'}' does not exist.")
        
        # Load the memory-mapped array
        try:
            self.memmap = np.memmap(self.path+'.memmap', mode='r', dtype='float32')
            # Infer the shape from the file size
            file_size = os.path.getsize(self.path+'.memmap')
            # Assuming 2D data; adjust if necessary
            self.shape_ = self._infer_shape(file_size, self.memmap.dtype)
            self.memmap = self.memmap.reshape(self.shape_)
        except Exception as e:
            raise IOError(f"Error loading memory-mapped file '{self.path}': {e}")
        
        self.chunk_size = 2048
        self.min_value = None
        self.max_value = None
    
    def _infer_shape(self, file_size, dtype):
        """
        Infers the shape of the 2D array based on file size and dtype.
        
        Args:
            file_size (int): Size of the file in bytes.
            dtype (np.dtype): Data type of the array.
        
        Returns:
            tuple: Inferred shape (rows, columns).
        
        Raises:
            ValueError: If the file size does not correspond to a valid 2D shape.
        """
        bytes_per_element = dtype.itemsize
        total_elements = file_size // bytes_per_element
        # Attempt to find a square matrix
        side = int(np.sqrt(total_elements))
        if side * side != total_elements:
            raise ValueError("File size does not correspond to a square 2D array.")
        return (side, side)
    
    def __getitem__(self, key):
        """
        Retrieves a slice of the dataset as a PyTorch tensor.
        
        Args:
            key (int, slice, tuple): Indexing key.
        
        Returns:
            torch.Tensor: Retrieved data as a PyTorch tensor.
        """
        data = self.memmap[key]
        return data
    
    @property
    def shape(self):
        """Returns the shape of the dataset."""
        return self.memmap.shape
    
    def min(self):
        """
        Computes the minimum value of the dataset efficiently using chunks.
        Utilizes cached value if available.
        
        Returns:
            torch.Tensor: Minimum value as a tensor.
        """
        if self.min_value is None:
            cache_path = self.path + '_min_max.npy'
            if os.path.exists(cache_path):
                self.min_value = np.load(cache_path)[0]
            else:
                print("Computing minimum value...")
                self.min_value = np.inf
                for start in range(0, self.shape[0], self.chunk_size):
                    end = min(start + self.chunk_size, self.shape[0])
                    chunk = self.memmap[start:end, :]
                    current_min = np.min(chunk)
                    if current_min < self.min_value:
                        self.min_value = current_min
                print(f"Minimum value computed and saved to '{cache_path}'.")
        return self.min_value
    
    def max(self):
        """
        Computes the maximum value of the dataset efficiently using chunks.
        Utilizes cached value if available.
        
        Returns:
            torch.Tensor: Maximum value as a tensor.
        """
        if self.max_value is None:
            cache_path = self.path + '_min_max.npy'
            if os.path.exists(cache_path):
                self.max_value = np.load(cache_path)[1]
            else:
                print("Computing maximum value...")
                self.max_value = -np.inf
                for start in range(0, self.shape[0], self.chunk_size):
                    end = min(start + self.chunk_size, self.shape[0])
                    chunk = self.memmap[start:end, :]
                    current_max = np.max(chunk)
                    if current_max > self.max_value:
                        self.max_value = current_max
        return self.max_value
    
    
    def close(self):
        """
        Closes the memory-mapped file by deleting the memmap object.
        """
        try:
            del self.memmap
            print(f"Memory-mapped file '{self.path}' successfully closed.")
        except Exception as e:
            print(f"Error closing memory-mapped file '{self.path}': {e}")

class TorchH5Wrapper:
    def __init__(self, h5_path, chunk_size=1024):
        """
        Initializes the wrapper for an HDF5 dataset.
        
        Args:
            h5_path (str): Path to the HDF5 file.
            chunk_size (int): Number of rows to process at a time for min/max computations.
        """
        self.path = h5_path
        self.file = h5py.File(h5_path+'.h5', 'r')
        self.dataset = self.file['dist']
        self.chunk_size = chunk_size
        self.min_value = None
        self.max_value = None

    def __getitem__(self, key):
        return torch.tensor(self.dataset[key])

    @property
    def shape(self):
        return self.dataset.shape

    def min(self):
        """Compute the minimum value of the dataset efficiently using chunks."""
        if self.min_value is None:
            if os.path.exists(self.path+'_min_max.npy'):
                self.min_value= np.load(self.path+'_min_max.npy')[0]
            else:
                print("Computing min")
                self.min_value = np.inf
                for start in range(0, self.dataset.shape[0], self.chunk_size):
                    chunk = self.dataset[start:start + self.chunk_size]
                    self.min_value = min(self.min_value, np.min(chunk))
            return torch.tensor(self.min_value)
        else:
            return torch.tensor(self.min_value)
        
    def max(self):
        """Compute the maximum value of the dataset efficiently using chunks."""
        if self.max_value is None:
            if os.path.exists(self.path+'_min_max.npy'):
                self.max_value = np.load(self.path+'_min_max.npy')[1]   
            else:
                print("Computing max")
                self.max_value = -np.inf
                for start in range(0, self.dataset.shape[0], self.chunk_size):
                    chunk = self.dataset[start:start + self.chunk_size]
                    self.max_value = max(self.max_value, np.max(chunk))
            return torch.tensor(self.max_value)
        else:
            return torch.tensor(self.max_value)

    def close(self):
        self.file.close()

# def normalize_image(image: np.array) -> np.array:
#     '''
#     Image already normalized on scan level.
#     Just transform to [-1, 1] and clipped to [-1, 1].
#     '''
#     assert image.min() >= 0 and image.max() <= 255
#     image = image / 255.0 * 2 - 1
#     image = np.clip(image, -1.0, 1.0)
#     return image

def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors

def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return torch.nn.functional.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

def pytorch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    
    """
        Convert a PyTorch tensor to a JAX-compatible array.
        
        Handles:
        - 4D tensors (B, C, H, W): Converts to channel-last format (B, H, W, C).
        - 2D tensors: Directly converts without transposing.
        
    """
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()  
    
    numpy_array = tensor.detach().cpu().numpy() 

    # if tensor.ndimension() == 4:
    #     numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))  # Rearrange to channel-last format (B, H, W, C) for jax

    jax_array = jnp.asarray(numpy_array)
    return jax_array

def convert_variables(images: torch.Tensor,
                      timestamps: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_start = images[:, 0, ...].float().to(device)
    x_end = images[:, 1, ...].float().to(device)
    if images.shape[1] == 3:
        x_start_aug = images[:, 2, ...].float().to(device)
    t_list = timestamps[0].float().to(device)
    if images.shape[1] == 3:
        return [x_start, x_end, x_start_aug], t_list
    else:
        return [x_start, x_end], t_list

def normalize_image_zero_to_one(image: np.array) -> np.array:
    '''
    Normalize an image to the range [0, 1] and clip values to [0, 1].
    '''
    assert image.min() >= 0 and image.max() <= 255, "Image values should be in the range [0, 255]."
    image = image / 255.0  # Normalize to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    return image

def load_png_as_tensor(image_path: str, model_type="ae", transform=None, numpy=False,normalize=False):
    """
    Load an image from the given path and convert it to a tensor or a PIL Image for foundation models.
    
    Args:
        image_path (str): The path to the image file.
        model_type (str): Specify the model type: 'unet', 'ae', or 'foundation'.
        transform (callable, optional): Optional transform for 'ae' and 'resnet' (default is basic tensor conversion).
        processor: The image processor for foundation models (optional).
    
    Returns:
        torch.Tensor or PIL.Image: The processed image (Tensor for 'ae' and 'resnet', PIL Image for 'foundation').
    """
    if model_type == "foundation":
        # Foundation model expects a PIL image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image  # Return PIL image for processor to handle

    # For UNet and ResNet models, return a tensor
    # Open the image file
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    if normalize:
        image = normalize_image_zero_to_one(image =np.asarray(image))
    else:
        image = np.asarray(image)
    if not numpy:
        # If no transform is provided, apply a default transform (for 'unet' and 'resnet')
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()  # Converts to [0, 1] range, (H, W, C) -> (C, H, W)
            ])

        # Apply the transform and return tensor for 'unet' and 'resnet'
        image = transform(image).float()
    return image

def applynnModuleList(ModList,x):
    with torch.no_grad():
        # out = x.float()
        out = x
        for layer in ModList:
            out = layer(out)
    return out

def interpolate_images(encoder, decoder, img1_path, img2_path, step=10, p=4):
    """
    Interpolate between two images in latent space and generate interpolated images.

    Args:
        encoder: The encoder model that generates latent representations.
        decoder: The decoder model (or ModuleList) that reconstructs images from latent representations.
        img1: First input image (tensor).
        img2: Second input image (tensor).
        step: The number of interpolation steps between the two latents.
        p: The number of interpolated images to display in each row.
    
    Returns:
        None
    """
    encoder.eval()  # Set encoder to eval mode
    decoder.eval()  # Set decoder to eval mode
    

     # Load images
    img1 = load_png_as_tensor(img1_path)
    img2 = load_png_as_tensor(img2_path)

    # Encode the images to get latent representations
    with torch.no_grad():
        latent1 = img1.unsqueeze(0).float()
        latent2 = img2.unsqueeze(0).float()
        for layer in encoder:
            latent1 = layer(latent1)
            latent2 = layer(latent2)
 
    # Create steps for interpolation
    latents = [latent1 * (1 - alpha) + latent2 * alpha for alpha in torch.linspace(0, 1, step)]
    # Decode the interpolated latents to generate images
    interpolated_images = []
    with torch.no_grad():
        for latent in latents:
            # Check if decoder is nn.ModuleList() or a single module
            if isinstance(decoder, torch.nn.ModuleList):
                # Sequentially pass through each module in the ModuleList
                output = latent
                for layer in decoder:
                    output = layer(output)
                generated_img = output
            else:
                # If it's a regular nn.Module, just apply forward pass
                generated_img = decoder(latent)
                
            interpolated_images.append(generated_img.squeeze().cpu().numpy())
    
  
    # Plot the original and interpolated images
    num_rows = (step // p) + 1  # Determine the number of rows needed for interpolated images
    total_rows = num_rows + 1   # Add an extra row for the original images

    fig, axs = plt.subplots(total_rows, p, figsize=(15, total_rows * 5))
    axs = axs.flatten()  # Flatten the axes array for easy indexing

    # Plot the original images in the first row (only 2 columns)
    axs[0].imshow(img1.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title("Image 1", fontsize=10)
    axs[0].axis('off')

    axs[1].imshow(img2.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title("Image 2", fontsize=10)
    axs[1].axis('off')

    # Turn off any extra subplots in the first row (if more than 2 columns exist)
    for i in range(2, p):
        axs[i].axis('off')

    # Plot the interpolated images starting from the second row
    start_idx = p  # Start plotting interpolated images from the second row
    for idx, interpolated_img in enumerate(interpolated_images):
        axs[start_idx + idx].imshow(interpolated_img, cmap='gray')
        axs[start_idx + idx].set_title(f'Interpolated {idx + 1}', fontsize=10)
        axs[start_idx + idx].axis('off')

    # Turn off any remaining empty subplots
    for i in range(start_idx + len(interpolated_images), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    return interpolated_images

def compute_batch_distances(model, img_batch1, img_batch2, model_type="resnet", cls_token=False, processor=None,device="cpu"):
    """
    Compute latent space norms between two batches of images.
    
    Args:
        model: The neural network model used to generate latent vectors.
        img_batch1: First batch of images.
        img_batch2: Second batch of images.
        model_type (str): Specify the model type: 'resnet', 'unet', or 'foundation'.
        processor: The image processor for foundation models (optional).
    
    Returns:
        torch.Tensor: A tensor of distances between the two batches.
    """
    model.eval()  # Set model to evaluation mode
    
    if model_type == "resnet":
        img_batch1 = img_batch1.unsqueeze(0).repeat(1, 3, 1, 1).to(device)  # Repeat across 3 channels for ResNet
        img_batch2 = img_batch2.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        latent1 = model(img_batch1)[-1]  # Extract the last layer output
        latent2 = model(img_batch2)[-1]
    elif model_type == "foundation":
        in1 = processor(images=img_batch1, return_tensors="pt").to(device)
        in2 = processor(images=img_batch2, return_tensors="pt").to(device)
        outputs1 = model(**in1)
        outputs2 = model(**in2)
        if cls_token:
            latent1 = outputs1[0][:, 0, :]  # Extract the CLS token for comparison
            latent2 = outputs2[0][:, 0, :]
        else:
            latent1 = outputs1[0][:, 1:, :]  # Use remaining tokens if needed
            latent2 = outputs2[0][:, 1:, :]
    else:
        _, latent1 = model.forward(img_batch1.unsqueeze(0).to(device))  # Use the forward pass for UNet
        _, latent2 = model.forward(img_batch2.unsqueeze(0).to(device))

    distances = torch.norm(latent1 - latent2, dim=-1)  # Compute Euclidean distance for the entire batch
    return distances

def get_latent_path(img_path,model_type="resnet"):
    if model_type == "resnet":
        return img_path.replace('LUMIERE_images_tumor-3px_256x256','LUMIERE_images_tumor1200px_256x256_latent_resnet').replace('.png','.pt')
    elif model_type == "foundation":
        return img_path.replace('LUMIERE_images_tumor1200px_256x256','LUMIERE_images_tumor1200px_256x256_latent_foundation').replace('.png','.pt')
    elif model_type == "ae":
        return img_path.replace('LUMIERE_images_tumor-3px_256x256','LUMIERE_images_tumor1200px_256x256_latent_ae').replace('.png','.pt')
    elif model_type == "ae_small":
        return img_path.replace('LUMIERE_images_tumor-3px_256x256','LUMIERE_images_tumor-3px_256x256_latent').replace('.png','.pt')
    else:
        print("Model type not supported")

def compute_batch_distances(model, img_batch1_paths, img_batch2_paths, model_type="resnet", cls_token=False, processor=None,device="cpu"):
    """
    Compute latent space norms between all pairs of images in two batches (given as image paths) using full feature maps.
    
    Args:
        model: The neural network model used to generate latent vectors.
        img_batch1_paths: List of image paths for the first batch of images.
        img_batch2_paths: List of image paths for the second batch of images.
        model_type (str): Specify the model type: 'resnet', 'unet', or 'foundation'.
        processor: The image processor for foundation models (optional).
    
    Returns:
        torch.Tensor: A tensor of distances between all pairs of samples in the two batches.
    """
    model.eval()  # Set model to evaluation mode
    latent1=get_latents(model, img_batch1_paths,  model_type, cls_token, processor,device)
    latent2=get_latents(model, img_batch2_paths,  model_type, cls_token, processor,device)
    

        # Compute the pairwise distance between every pair of latents (1 vs all in the other batch)
    pairwise_distances = torch.cdist(latent1.view(latent1.size(0), -1), latent2.view(latent2.size(0), -1))  # Shape: (batch_size1, batch_size2)
    
    return pairwise_distances
      
def get_images(img_batch1_paths):
    """
    Compute latent space norms between all pairs of images in two batches (given as image paths) using full feature maps.
    
    Args:
        model: The neural network model used to generate latent vectors.
        img_batch1_paths: List of image paths for the first batch of images.
        img_batch2_paths: List of image paths for the second batch of images.
        model_type (str): Specify the model type: 'resnet', 'ae', or 'foundation'.
        processor: The image processor for foundation models (optional).
    
    Returns:
        torch.Tensor: A tensor of distances between all pairs of samples in the two batches.
    """

    latent_paths1 = [img_path for img_path in img_batch1_paths]

    # Check if the latent is already computed and saved

    if os.path.exists(latent_paths1[0]):
        latent1 = [torch.load(latent_path) for latent_path in latent_paths1]
        latent1 = torch.stack(latent1)

    return latent1

def get_latents(model, img_batch1_paths,  model_type="ae_small", cls_token=False, processor=None,device="cpu"):
    """
    Compute latent space norms between all pairs of images in two batches (given as image paths) using full feature maps.
    
    Args:
        model: The neural network model used to generate latent vectors.
        img_batch1_paths: List of image paths for the first batch of images.
        img_batch2_paths: List of image paths for the second batch of images.
        model_type (str): Specify the model type: 'resnet', 'ae', or 'foundation'.
        processor: The image processor for foundation models (optional).
    
    Returns:
        torch.Tensor: A tensor of distances between all pairs of samples in the two batches.
    """
    if model is not None:
        model.eval()  # Set model to evaluation mode
        model=model.to(device)

    latent_paths1 = [get_latent_path(img_path,model_type) for img_path in img_batch1_paths]

    if os.path.exists(latent_paths1[0]):
        latent1 = [torch.load(latent_path,weights_only=False) for latent_path in latent_paths1]
        latent1 = torch.stack(latent1)
    else:
        if model_type == "foundation":
            img_batch1 = [load_png_as_tensor(img_path, model_type, processor) for img_path in img_batch1_paths]
            img_batch1 = processor(images=img_batch1, return_tensors="pt").to(device)
        else:
            # Load images from paths and convert them to tensors
            img_batch1 = [load_png_as_tensor(img_path, model_type, processor) for img_path in img_batch1_paths]
       
        
            # Convert list of tensors into batch tensors
            img_batch1 = torch.stack(img_batch1)  # Shape: (batch_size1, C, H, W)
       

        if model_type == "resnet":
            img_batch1 = img_batch1.repeat(1, 3, 1, 1).to(device)  # Repeat across 3 channels for ResNet
           
            latent1 = model(img_batch1)[-1]  # Latent shape: (batch_size1, 2048, 8, 8)
         
            
        elif model_type == "foundation":
            outputs1 = model(**img_batch1)
            if cls_token:
                latent1 = outputs1[0][:, 0, :]  # Latent shape: (batch_size1, latent_dim)
                
            else:
                latent1 = outputs1[0][:, 1:, :]
               
        else:
            # _, latent1 = model.forward(img_batch1.to(device))  # Latent shape: (batch_size1, latent_dim)
            latent1 = None

        # Compute the pairwise distance between every pair of latents (1 vs all in the other batch)
    # pairwise_distances = torch.cdist(latent1.view(latent1.size(0), -1), latent2.view(latent2.size(0), -1))  # Shape: (batch_size1, batch_size2)
    
    return latent1

def compute_and_plot_latent_norms(model, image_paths, sequential=False, images_per_row=3, model_type="ae", cls_token=False, processor=None):
    """
    Compute latent space norms of images and plot them with image names and norms in the title.
    
    Args:
        model: The neural network model used to generate latent vectors.
        image_paths: List of paths to images.
        sequential (bool): If True, compute the norm sequentially pairwise.
                           If False, compute the norm of all images to the first image.
        images_per_row (int): Number of images to display per row.
        model_type (str): Specify the model type: 'unet', 'resnet', or 'foundation'.
        processor: The image processor for foundation models (like Swin Transformer).
    
    Returns:
        None
    """
    model.eval()  # Set model to evaluation mode
    
    # Load the first image and get its latent representation
    reference_img_path = image_paths[0]
    img1 = load_png_as_tensor(reference_img_path, model_type, processor)
    
    if model_type == "resnet":
        img1 = img1.unsqueeze(0).repeat(1, 3, 1, 1)  # Repeat across 3 channels for ResNet
        latent1 = model(img1)[-1]  # Extract the last layer output
    elif model_type == "foundation":
        inputs = processor(images=img1, return_tensors="pt")
        outputs = model(**inputs)
        if cls_token:
            latent1 = outputs[0][:, 0, :]  # Extract the CLS token for comparison
        else:
            latent1 = outputs[0][:, 1:, :]  # Use remaining tokens if needed
    else:
        _, latent1 = model.forward(img1.unsqueeze(0))  # Use the forward pass for UNet
    
    norms = [(reference_img_path, reference_img_path, 0.0)]  # First image norm against itself is zero
    
    # Loop through all images starting from the second one
    for i in range(1, len(image_paths)):
        img2 = load_png_as_tensor(image_paths[i], model_type, processor)
        
        if model_type == "resnet":
            img2 = img2.unsqueeze(0).repeat(1, 3, 1, 1)  # Repeat across 3 channels for ResNet
            latent2 = model(img2)[-1]  # Extract the last layer output
        elif model_type == "foundation":
            inputs = processor(images=img2, return_tensors="pt")
            outputs = model(**inputs)
            if cls_token:
                latent2 = outputs[0][:, 0, :]  # Extract the CLS token for comparison
            else:
                latent2 = outputs[0][:, 1:, :]
        else:
            _, latent2 = model.forward(img2.unsqueeze(0))  # Use the forward pass for UNet
        
        if sequential:
            # If sequential is True, compute norm sequentially
            norm = torch.norm(latent1 - latent2)
            latent1 = latent2  # Update the first latent to the next for pairwise computation
        else:
            # Compute norm against the first image
            norm = torch.norm(latent1 - latent2)
        
        norms.append((reference_img_path, image_paths[i], norm.item()))
    
    # Determine the number of rows needed for plotting
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / images_per_row)
    
    fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, num_rows * 5))
    axs = axs.flatten()  # Flatten the axis array for easy indexing

    for idx, (ref_img, img_path, norm) in enumerate(norms):
        if model_type == "foundation":
            img = Image.open(img_path)  # For foundation models, load and display PIL image directly
        else:
            img = load_png_as_tensor(img_path, model_type, processor).squeeze().numpy()  # For other models, load and convert to numpy
        
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(f'{ref_img[-25:]} vs {img_path[-25:]}\nNorm: {norm:.4f}', fontsize=10)
        axs[idx].axis('off')
    
    # Turn off any remaining empty subplots
    for i in range(len(norms), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_latent_space(encoder, patient_slice_week, processor=None, model_type="ae", device="cpu", cls_token=False):
   
    encoder = encoder.to(device)
    encoder=encoder.eval()
    for slice_path in tqdm(patient_slice_week, desc="Processing Patients"):
        img = load_png_as_tensor(slice_path, model_type=model_type)
        
    
        with torch.no_grad():
            if model_type == "resnet":
                img = img.unsqueeze(0).float().to(device)
                latent = encoder(img.repeat(1, 3, 1, 1))[-1]
            elif model_type == "foundation":
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = encoder(**inputs)
                if cls_token:
                    latent = outputs[0][:, 0, :]
                else:
                    latent = outputs[0][:, 1:, :]
            else:
                img = img.unsqueeze(0).float().to(device)
                latent = encoder(img).squeeze()

        path_parts = slice_path.split('/')
        if len(path_parts) >= 4:
            path_parts[-4] = path_parts[-4] + '_latent'
        else:
            raise ValueError(f"Unexpected path structure: {slice_path}")

        save_dir = '/'+os.path.join(*path_parts[:-1])
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.splitext(path_parts[-1])[0] + '.pt'
        save_path = os.path.join(save_dir, filename)
        # print(f"Saving to: {save_path}")
        # print(latent.shape)
        torch.save(latent.cpu(), save_path)
        
                
                # print(f"Saved latent representation: {save_path}")

def get_all_patient_ids(image_folder: str) -> List[str]:
    all_patient_folders = sorted(glob(f'{image_folder}/*/'))
    all_patient_ids = [os.path.basename(item.rstrip('/')) for item in all_patient_folders]
    return all_patient_ids

def get_patient_images(image_folder: str, patient_id: str) -> List[List[str]]:
    patient_folder = os.path.join(image_folder, patient_id)
    all_slices = sorted(glob(f'{patient_folder}/*'))  # Get all slices in the patient's folder
    all_slices_weeks = [sorted(glob(f'{slice_path}/*')) for slice_path in all_slices]  # Get weeks for each slice
    # print(f'Patient ID: {patient_id}, Number of slices: {len(all_slices)}, Number of weeks: {len(all_slices_weeks)}')
    return all_slices_weeks

def get_nested_patient_slices(image_folder: str) -> List[List[List[str]]]:
    all_patient_ids = get_all_patient_ids(image_folder)  # Get all patient IDs
    all_patient_data = []
    
    # For each patient, get their slice and week image paths
    for patient_id in all_patient_ids:
        patient_slices_weeks = get_patient_images(image_folder,patient_id)  # Get all slices and weeks for the patient
        # print(f'Patient ID: {patient_id}, Number of slices: {len(patient_slices_weeks)}')
        all_patient_data.append(sorted(patient_slices_weeks))  # Append this patient's slice and week data
    return all_patient_data

def construct_geodist_faiss_scipy(
    features_np: np.ndarray,
    k: int,
    distance_matrix_path: str,
    device: str = "cpu",
    use_memmap: bool = True,
    memmap_dtype: str = 'float32',
    memmap_chunk_size: int = 1000,
    h5_compression: bool = True,
    compression_opts: int = 4
):
    """
    Constructs a geodesic distance matrix using FAISS for nearest neighbor search,
    NetworkX for graph construction, and SciPy for shortest path computation.
    Stores the distance matrix either as an HDF5 file or a memory-mapped file based on the `use_memmap` flag.
    Tracks min/max values and saves the graph.

    Args:
        features_np (np.ndarray): Feature vectors of shape (N, D).
        k (int): Number of nearest neighbors.
        distance_matrix_path (str): Base path to store the distance matrix.
                                    If saving as HDF5, it appends '.h5'.
                                    If saving as memmap, it appends '.memmap'.
        device (str, optional): 'cpu' or 'cuda'. Defaults to "cpu".
        use_memmap (bool, optional): If True, saves the distance matrix as a memory-mapped file.
                                      If False, saves as an HDF5 file. Defaults to False.
        memmap_dtype (str, optional): Data type for memmap. Defaults to 'float32'.
        memmap_chunk_size (int, optional): Number of rows to write at a time for memmap. Defaults to 1000.
        h5_compression (bool, optional): If True and saving as HDF5, applies GZIP compression.
                                          Ignored if use_memmap is True. Defaults to True.
        compression_opts (int, optional): Compression level for HDF5 (0-9). Relevant only if h5_compression is True.
                                          Defaults to 4.
    """

    N, d = features_np.shape
    print(f"Number of nodes: {N}, Feature dimension: {d}")

    # Initialize the graph
    G = nx.Graph()

    # Build Faiss index and search for nearest neighbors
    print("Building Faiss index and searching for nearest neighbors...")
    if device == 'cuda':
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(features_np)

    # Search for k+1 nearest neighbors (including self)
    distances, indices = index.search(features_np, k + 1)

    # Ensure distances are non-negative and convert to Euclidean distances
    distances = np.maximum(distances, 0)
    distances = np.sqrt(distances)

    # Build initial k-NN graph
    print("Building initial k-NN graph...")
    for idx in tqdm(range(N), desc="Adding edges to graph"):
        idx_neighbors = indices[idx]
        idx_distances = distances[idx]

        # Exclude self (first neighbor is the query point itself)
        neighbor_info = [
            (int(n_idx), float(dist))
            for n_idx, dist in zip(idx_neighbors, idx_distances)
            if n_idx != idx
        ]
        for neighbor_idx, distance in neighbor_info:
            # Add edges in both directions to ensure symmetry
            G.add_edge(idx, neighbor_idx, weight=distance)
            G.add_edge(neighbor_idx, idx, weight=distance)

    # Step to connect disconnected components
    print("Checking for disconnected components...")
    components = list(nx.connected_components(G))

    if len(components) > 1:
        print(f"Found {len(components)} disconnected components. Connecting components...")
        while len(components) > 1:
            component_a = components[0]
            component_b = components[1]

            idx_a = np.array(list(component_a))
            idx_b = np.array(list(component_b))

            features_a = features_np[idx_a]
            features_b = features_np[idx_b]

            if device == 'cuda':
                res = faiss.StandardGpuResources()
                index_b = faiss.GpuIndexFlatL2(res, d)
            else:
                index_b = faiss.IndexFlatL2(d)
            index_b.add(features_b)

            distances_ab, indices_ab = index_b.search(features_a, 1)

            min_idx = np.argmin(distances_ab)
            node_a = idx_a[min_idx]
            node_b = idx_b[indices_ab[min_idx][0]]
            min_distance = np.sqrt(max(distances_ab[min_idx][0], 0))

            G.add_edge(node_a, node_b, weight=min_distance)
            G.add_edge(node_b, node_a, weight=min_distance)
            print(f"Connected node {node_a} and node {node_b} with distance {min_distance}")

            components = list(nx.connected_components(G))

    print("Creating adjacency matrix...")
    adjacency = nx.to_scipy_sparse_array(G, nodelist=range(N), weight='weight', format='csr')
    adjacency = adjacency.astype(np.float32)

    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)

    print("Ensuring adjacency matrix is symmetric...")
    adjacency = (adjacency + adjacency.T) / 2

    if (adjacency.data < 0).any():
        num_negatives = np.sum(adjacency.data < 0)
        print(f"Warning: Adjacency matrix contains {num_negatives} negative weights.")
        adjacency.data = np.maximum(adjacency.data, 0)

    # Initialize variables for min and max
    min_value = np.inf
    max_value = -np.inf

    if use_memmap:
        # Define memmap file path
        memmap_file_path = distance_matrix_path + '.memmap'

        # Remove existing memmap file if it exists
        if os.path.exists(memmap_file_path):
            os.remove(memmap_file_path)

        print(f"Saving distance matrix to {memmap_file_path} as a memory-mapped file...")
        # Create a memory-mapped file
        memmap = np.memmap(memmap_file_path, dtype=memmap_dtype, mode='w+', shape=(N, N))

        # Compute shortest paths and write to memmap
        print("Computing shortest paths and storing distance matrix as memmap...")
        for i in tqdm(range(0, N, memmap_chunk_size), desc="Computing shortest paths"):
            idx_start = i
            idx_end = min(i + memmap_chunk_size, N)
            indices_range = np.arange(idx_start, idx_end, dtype=np.int32)

            dist_chunk = dijkstra(csgraph=adjacency, indices=indices_range, directed=False)

            dist_chunk = np.maximum(dist_chunk, 0)
            memmap[idx_start:idx_end, :] = dist_chunk.astype(memmap_dtype)
            memmap.flush()

            # Update min and max
            min_value = min(min_value, np.nanmin(dist_chunk))
            max_value = max(max_value, np.nanmax(dist_chunk))

        # Delete memmap object to flush remaining data and close the file
        del memmap

        print(f"Distance matrix saved to {memmap_file_path}")
    else:
        # Define HDF5 file path
        h5_file_path = distance_matrix_path + '.h5'

        # Remove existing HDF5 file if it exists
        if os.path.exists(h5_file_path):
            os.remove(h5_file_path)

        # Determine compression settings
        compression = 'gzip' if h5_compression else None
        compression_kwargs = {'compression': compression} if compression else {}

        if h5_compression:
            compression_kwargs['compression_opts'] = compression_opts

        print(f"Saving distance matrix to {h5_file_path} with chunk size (1, {N})...")
        with h5py.File(h5_file_path, 'w') as h5f:
            # Create the dataset with fixed chunk size (1, N)
            dset = h5f.create_dataset(
                'dist',
                shape=(N, N),
                dtype='float32',
                chunks=(1, N),
                **compression_kwargs
            )

            # Compute shortest paths and write to HDF5
            print("Computing shortest paths and storing distance matrix as HDF5...")
            for i in tqdm(range(0, N, memmap_chunk_size), desc="Computing shortest paths"):
                idx_start = i
                idx_end = min(i + memmap_chunk_size, N)
                indices_range = np.arange(idx_start, idx_end, dtype=np.int32)

                dist_chunk = dijkstra(csgraph=adjacency, indices=indices_range, directed=False)

                dist_chunk = np.maximum(dist_chunk, 0)
                dset[idx_start:idx_end, :] = dist_chunk.astype('float32')
                h5f.flush()

                # Update min and max
                min_value = min(min_value, np.nanmin(dist_chunk))
                max_value = max(max_value, np.nanmax(dist_chunk))

        print(f"Distance matrix saved to {h5_file_path}")

    # Save min/max values to a file
    min_max_path = distance_matrix_path + '_min_max.npy'
    np.save(min_max_path, np.array([min_value, max_value], dtype=np.float32))
    print(f"Min/max values saved to {min_max_path}")

    # Save the graph
    graph_path = distance_matrix_path + '_graph.graphml'
    nx.write_graphml(G, graph_path)
    print(f"Graph saved to {graph_path}")

def construct_geodist_isomap(features_np, k, distance_matrix_path, use_memmap=True, h5_compression=False, compression_opts=2):
    """
    Constructs a geodesic distance matrix using Isomap.
    Saves the distance matrix either as a memory-mapped file or an HDF5 file based on flags.
    Tracks min/max values and saves the k-NN graph.

    Args:
        features_np (np.ndarray): Feature vectors of shape (N, D).
        k (int): Number of nearest neighbors.
        distance_matrix_path (str): Base path to store the distance matrix.
                                    If saving as HDF5, it appends '.h5'.
                                    If saving as memmap, it appends '.memmap'.
        use_memmap (bool, optional): If True, saves the distance matrix as a memory-mapped file.
                                      If False, saves as an HDF5 file. Defaults to False.
        h5_compression (bool, optional): If True and saving as HDF5, applies GZIP compression.
                                          Ignored if use_memmap is True. Defaults to True.
        compression_opts (int, optional): Compression level for HDF5 (0-9). Relevant only if h5_compression is True.
                                          Defaults to 4.

    Raises:
        ValueError: If an unsupported save method is specified.
    """
    N, d = features_np.shape
    print(f"Number of nodes: {N}, Feature dimension: {d}")

    print("Fitting Isomap model and computing geodesic distances...")
    isomap = Isomap(
        n_neighbors=k,
        n_components=2,
        eigen_solver='auto',
        path_method='auto',
        neighbors_algorithm='auto',
        metric='euclidean',
        n_jobs=-1
    )
    isomap.fit(features_np)
    dist_matrix = isomap.dist_matrix_
    print("Geodesic distance matrix computed.")

    if use_memmap:
        # Define memmap file path
        memmap_file_path = distance_matrix_path + '.memmap'

        # Remove existing memmap file if it exists
        if os.path.exists(memmap_file_path):
            os.remove(memmap_file_path)

        print(f"Saving distance matrix to {memmap_file_path} as a memory-mapped file...")
        # Create a memory-mapped file
        memmap = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=(N, N))

        # Write data one row at a time
        for i in range(N):
            if i % 1000 == 0 or i == N - 1:
                print(f"Writing row {i + 1} of {N}...")
            memmap[i, :] = dist_matrix[i, :].astype('float32')
            memmap.flush()  # Ensure the row is written to disk

        # Delete memmap object to flush remaining data and close the file
        del memmap

        print(f"Distance matrix saved to {memmap_file_path}")
    else:
        # Define HDF5 file path
        h5_file_path = distance_matrix_path + '.h5'

        # Remove existing HDF5 file if it exists
        if os.path.exists(h5_file_path):
            os.remove(h5_file_path)

        # Determine compression settings
        compression = 'gzip' if h5_compression else None
        compression_kwargs = {'compression': compression} if compression else {}

        if h5_compression:
            compression_kwargs['compression_opts'] = compression_opts

        print(f"Saving distance matrix to {h5_file_path} with chunk size (1, {N})...")
        with h5py.File(h5_file_path, 'w') as h5f:
            # Create the dataset with fixed chunk size (1, N)
            dset = h5f.create_dataset(
                'dist',
                shape=(N, N),
                dtype='float32',
                chunks=(1, N),
                **compression_kwargs
            )

            # Write data one row at a time
            for i in range(N):
                if i % 1000 == 0 or i == N - 1:
                    print(f"Writing row {i + 1} of {N}...")
                dset[i, :] = dist_matrix[i, :].astype('float32')
                h5f.flush()  # Ensure the row is written to disk

        print(f"Distance matrix saved to {h5_file_path}")

    # Compute min and max values
    print("Computing min and max values of the distance matrix...")
    min_value = np.nanmin(dist_matrix)
    max_value = np.nanmax(dist_matrix)
    min_max_path = distance_matrix_path + '_min_max.npy'
    np.save(min_max_path, np.array([min_value, max_value], dtype=np.float32))
    print(f"Min/max values saved to {min_max_path}")

    # Extract the k-NN graph
    print("Extracting k-NN graph...")
    knn_graph = isomap.nbrs_.kneighbors_graph(features_np)
    knn_graph_symmetric = (knn_graph + knn_graph.T) / 2
    knn_graph_symmetric = csr_matrix(knn_graph_symmetric)
    G = nx.from_scipy_sparse_array(knn_graph_symmetric, edge_attribute='weight')

    # Save the graph
    graph_path = distance_matrix_path + '_graph.graphml'
    nx.write_graphml(G, graph_path)
    print(f"Graph saved to {graph_path}")

def slerp_interpolation(low, high, n_steps):
    """
    Performs spherical linear interpolation (SLERP) between two latent vectors.

    Parameters:
    - low (np.ndarray): The starting latent vector.
    - high (np.ndarray): The ending latent vector.
    - n_steps (int): The number of interpolation steps (including the start and end vectors).

    Returns:
    - List[np.ndarray]: A list of interpolated latent vectors.
    """

    # Ensure the inputs are numpy arrays
    low = torch.tensor(low)
    high = torch.tensor(high)
    
    # Normalize the vectors
    low_norm = low / torch.linalg.norm(low)
    high_norm = high / torch.linalg.norm(high)
    
    # Compute the cosine of the angle between the vectors
    dot = torch.dot(low_norm.squeeze(), high_norm.squeeze())
    dot = torch.clip(dot, -1.0, 1.0)  # Clamp to avoid numerical errors

    # Compute the angle between the vectors
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    
    # Handle the case where the vectors are the same or opposite
    if sin_omega == 0:
        # Return a list of linearly interpolated vectors
        return [((1 - t) * low + t * high) for t in torch.linspace(0, 1, n_steps)]
    
    # Generate interpolation coefficients
    interpolated_vectors = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        factor1 = torch.sin((1 - t) * omega) / sin_omega
        factor2 = torch.sin(t * omega) / sin_omega
        interpolated_vector = factor1 * low + factor2 * high
        interpolated_vectors.append(interpolated_vector)
    
    return interpolated_vectors

def interactive_plot(embedding,file_name,color_map='blue'):
    # Create an interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_map,
            opacity=0.7
        )
    )])

    # Add title and axis labels, and adjust the figure size
    fig.update_layout(
        title=file_name,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        width=1200,  # Set the width of the plot
        height=800,  # Set the height of the plot
    )

    # Display the plot in the notebook
    fig.show()

    # Save the interactive plot as an HTML file
    html_file_path = file_name+".html"
    fig.write_html(html_file_path)

    print(f"Interactive 3D  {html_file_path}")

def plot_graph_nodes(path_data,all_samples,title):
    
    n_images = len(path_data)
    n_cols = 4  
    n_rows = math.ceil(n_images / n_cols)


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  


    for i, (ax, img) in enumerate(zip(axes, path_data)):
        ax.imshow(img.reshape(256,256))  
        ax.axis('off')  
        ax.set_title(f"{all_samples[path_data[i]][-28:]}")


    for i in range(n_images, len(axes)):
        axes[i].axis('off')


    plt.suptitle(title, fontsize=16)


    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

def test_implementations():
    import sys

    # Generate test data
    N = 5000  # Number of data points (adjust for testing)
    d = 4096   # Dimensionality of the feature vectors
    k = 7    # Number of nearest neighbors
    device = 'cpu'  # Change to 'cuda' if GPU is available

    print("Generating random test data...")
    features_np = np.random.rand(N, d).astype(np.float32)

    # Test original implementation
    print("\nTesting original implementation...")
    tracemalloc.start()
    start_time = time.time()
    dij_original, G_original = construct_geodist_faiss_scipy(features_np, k,'nope')
    original_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    original_memory = peak / 10**6  # Convert bytes to megabytes
    tracemalloc.stop()
    print(f"Original implementation took {original_time:.2f} seconds and used {original_memory:.2f} MB of memory.")

    # Test optimized implementation
    print("\nTesting optimized implementation...")
    tracemalloc.start()
    start_time = time.time()
    dij_optimized, adjacency_optimized = construct_geodist_faiss_scipy_new(features_np, k,'nope')
    optimized_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    optimized_memory = peak / 10**6  # Convert bytes to megabytes
    tracemalloc.stop()
    print(f"Optimized implementation took {optimized_time:.2f} seconds and used {optimized_memory:.2f} MB of memory.")

    # Compare the distance matrices
    print("\nComparing the outputs...")
    difference = np.abs(dij_original - dij_optimized)
    max_difference = np.nanmax(difference)
    print(f"Maximum difference between distance matrices: {max_difference:.6f}")

    if np.allclose(dij_original, dij_optimized, atol=1e-5, equal_nan=True):
        print("The distance matrices are approximately equal.")
    else:
        print("The distance matrices differ.")

    # Summary
    print("\nSummary:")
    print(f"Original implementation: Time = {original_time:.2f}s, Peak Memory = {original_memory:.2f} MB")
    print(f"Optimized implementation: Time = {optimized_time:.2f}s, Peak Memory = {optimized_memory:.2f} MB")

def process_file_paths_with_indices(file_paths: List[str]) -> Tuple[
    List[Tuple[str, int, str, int]],  # Task 1: Pairs (path1, index1, path2, index2)
    List[Tuple[str, int, str, int, str, int]],  # Task 2: Triplets (path_top, index_top, path_mid, index_mid, path_bottom, index_bottom)
    List[Tuple[str, int, str, int]]   # Task 3: Pairs (path1, index1, path2, index2)
]:
    """
    Processes a list of file paths and generates:
    1. Pairs with the same week and slice but different patients.
    2. Triplets with top, mid, and bottom slices for each patient and each week.
    3. Pairs with the same patient and slice but different weeks.

    Each pair/triplet includes the actual file paths and their indices in the original list.

    Parameters:
    - file_paths (List[str]): List of file path strings.

    Returns:
    - Tuple containing three lists:
        1. List of tuples: (path1, index1, path2, index2)
        2. List of tuples: (path_top, index_top, path_mid, index_mid, path_bottom, index_bottom)
        3. List of tuples: (path1, index1, path2, index2)
    """
    
    # Updated regular expression pattern to handle optional suffixes after week number
    pattern = r".*/Patient-(\d{3})/slice_(\d+)/week_(\d+)(?:-\d+)?\.png$"
    
    # Parsed data storage: List of dicts with path, index, patient_id, slice_num, week_num
    parsed_data = []
    
    # Collect problematic paths (if any)
    problematic_paths = []
    
    for idx, path in enumerate(file_paths):
        match = re.match(pattern, path)
        if match:
            patient_id = f"Patient-{match.group(1)}"
            slice_number = int(match.group(2))
            week_number = int(match.group(3))
            parsed_data.append({
                'path': path,
                'index': idx,
                'patient_id': patient_id,
                'slice_number': slice_number,
                'week_number': week_number
            })
        else:
            # Collect problematic paths
            problematic_paths.append(path)
    
    if problematic_paths:
        print(f"Warning: {len(problematic_paths)} path(s) did not match the expected format and were skipped.")
        # Optionally, you can print them or handle them as needed
        # for p_path in problematic_paths:
        #     print(f"Skipped: {p_path}")
    
    # Organize data into dictionaries for efficient grouping
    week_slice_to_patients = defaultdict(list)      # For Task 1
    patient_week_to_slices = defaultdict(list)      # For Task 2
    patient_slice_to_weeks = defaultdict(list)      # For Task 3
    
    # Additionally, map (patient, week, slice) to path and index for Task 2
    patient_week_to_slice_details = defaultdict(list)
    
    for entry in parsed_data:
        patient = entry['patient_id']
        slice_num = entry['slice_number']
        week = entry['week_number']
        path = entry['path']
        idx = entry['index']
        
        # Populate for Task 1
        week_slice_to_patients[(week, slice_num)].append((path, idx, patient))
        
        # Populate for Task 2
        patient_week_to_slices[(patient, week)].append(slice_num)
        patient_week_to_slice_details[(patient, week)].append((slice_num, path, idx))
        
        # Populate for Task 3
        patient_slice_to_weeks[(patient, slice_num)].append((week, path, idx))
    
    # Task 1: Pairs with the same week and slice but different patients
    pairs_same_week_slice_diff_patients = []
    
    for (week, slice_num), patient_entries in week_slice_to_patients.items():
        # Group patients by patient_id to ensure different patients
        patients = defaultdict(list)
        for path, idx, patient in patient_entries:
            patients[patient].append((path, idx))
        
        patient_ids = list(patients.keys())
        if len(patient_ids) < 2:
            continue  # Need at least two different patients
        
        # Generate all unique patient pairs
        for patient1, patient2 in combinations(patient_ids, 2):
            # For each pair of patients, pair all their corresponding paths
            for path1, idx1 in patients[patient1]:
                for path2, idx2 in patients[patient2]:
                    pairs_same_week_slice_diff_patients.append((path1, idx1, path2, idx2))
    
    # Task 2: Triplets with top, mid, and bottom slices for each patient and week
    triplets_top_mid_bottom = []
    
    for (patient, week), slices in patient_week_to_slice_details.items():
        if len(slices) < 3:
            continue  # Need at least three slices to form a triplet
        
        # Sort the slices by slice_number
        sorted_slices = sorted(slices, key=lambda x: x[0])  # x[0] is slice_number
        
        top_slice, top_path, top_idx = sorted_slices[0]
        bottom_slice, bottom_path, bottom_idx = sorted_slices[-1]
        mid_index = len(sorted_slices) // 2
        mid_slice, mid_path, mid_idx = sorted_slices[mid_index]
        
        triplets_top_mid_bottom.append((top_path, top_idx, mid_path, mid_idx, bottom_path, bottom_idx))
    
    # Task 3: Pairs with the same patient and slice but different weeks
    pairs_same_patient_slice_diff_weeks = []
    
    for (patient, slice_num), week_entries in patient_slice_to_weeks.items():
        if len(week_entries) < 2:
            continue  # Need at least two weeks
        
        # Group weeks by week_number to ensure different weeks
        weeks = defaultdict(list)
        for week, path, idx in week_entries:
            weeks[week].append((path, idx))
        
        week_numbers = list(weeks.keys())
        if len(week_numbers) < 2:
            continue  # Need at least two different weeks
        
        # Generate all unique week pairs
        for week1, week2 in combinations(week_numbers, 2):
            # For each pair of weeks, pair all their corresponding paths
            for path1, idx1 in weeks[week1]:
                for path2, idx2 in weeks[week2]:
                    pairs_same_patient_slice_diff_weeks.append((path1, idx1, path2, idx2))
    
    return pairs_same_week_slice_diff_patients, triplets_top_mid_bottom, pairs_same_patient_slice_diff_weeks

def extract_patient_volumes(all_slices, patients_of_interest):
    patient_pattern = re.compile(r"Patient-(\d+)")
    slice_pattern   = re.compile(r"slice_(\d+)")
    week_pattern    = re.compile(r"week_(\d+(?:-\d+)?)")
    patient_week_slices = defaultdict(list)
    
    for path in all_slices:
        p = patient_pattern.search(path)
        s = slice_pattern.search(path)
        w = week_pattern.search(path)
        if p and s and w:
            patient_num = int(p.group(1))
            if patient_num in patients_of_interest:
                patient_week_slices[(patient_num, w.group(1))].append(int(s.group(1)))
    
    return [(p, w, min(sl), max(sl)) for (p, w), sl in patient_week_slices.items()]

def sample_two_slices(patient_tuple, 
                      min_slice_diff=32,
                      base_path=ROOT+'/ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256'):
    """
    Given a tuple (patient_number, week_str, start_slice, end_slice), 
    this function picks two slices that are at least 32 slices apart,
    reconstructs the corresponding file paths, and returns them.
    """
    patient_number, week_str, start_slice, end_slice = patient_tuple
    if end_slice - start_slice < min_slice_diff:
        raise ValueError("Not enough range to pick two slices 32 apart.")
    slice_a = rand.randint(start_slice, end_slice - min_slice_diff)
    slice_b = rand.randint(slice_a + min_slice_diff, end_slice)
    path_a = f"{base_path}/Patient-{patient_number:03d}/slice_{slice_a}/week_{week_str}.png"
    path_b = f"{base_path}/Patient-{patient_number:03d}/slice_{slice_b}/week_{week_str}.png"
    return path_a, path_b, slice_a,slice_b

def plot_multiple_grids(image_dict, cols=5, img_size=(2, 2)):
    """
    Plots multiple sets of images side-by-side, each set with its own label.
    
    Parameters:
    - image_dict: dict, where keys are labels (strings) and values are arrays (N x H x W) of images.
    - cols: number of columns per set.
    - img_size: tuple (width, height) for each subplot.

    Returns:
    - fig: Matplotlib figure object.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure dictionary preserves order (Python 3.7+ should by default)
    labels = list(image_dict.keys())
    sets = len(labels)

    # Calculate number of rows needed
    # For each set: rows = ceil(len(images)/cols)
    # Global rows = max of all these rows
    max_images = max(len(images) for images in image_dict.values())
    rows = (max_images + cols - 1) // cols

    # Figure size: sets * cols horizontally and rows vertically
    figsize = (sets * cols * img_size[0], rows * img_size[1])
    fig, axes = plt.subplots(rows, sets * cols, figsize=figsize)

    # If there's only one row, axes might not be a 2D array
    if rows == 1:
        axes = np.array([axes])  # Make it 2D for consistency

    # Flatten for easy indexing
    # We'll treat the figure as a grid of rows x (sets*cols)
    axes_flat = axes.ravel()

    # Plot each set
    for s_idx, label in enumerate(labels):
        images = image_dict[label]
        n_images = len(images)
        
        # Starting column for this set: s_idx * cols
        start_col = s_idx * cols

        # Title: place above the first row of this set
        # The top-left subplot of this set is at (0, start_col)
        axes[0, start_col].set_title(label, fontsize=14)

        for i in range(rows * cols):
            ax_idx = (0 * sets * cols) + start_col + i  # linear index offset by row in the loop below
            # Actually we need to consider row as well, better to compute from row, col
            # row = i // cols, col = i % cols + start_col
            row = i // cols
            col = i % cols + start_col
            ax = axes[row, col]

            if i < n_images:
                ax.imshow(images[i].squeeze(), cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')

    plt.tight_layout()
    return fig


from mpl_toolkits.axes_grid1 import make_axes_locatable  # Required import for colorbar placement

def plot_images_with_distances(images, distances=None, cols=5, img_size=(2, 2), save_path=None):
    """
    Plots images with their corresponding deformation distances as titles and returns
    the matplotlib figure object. Each image includes an individual color scale (colorbar).

    Parameters:
    - images: List or NumPy array of images.
    - distances: List or NumPy array of distances, aligned with images.
    - cols: Number of columns in the plot grid.
    - img_size: Tuple specifying the size of each subplot (width, height).

    Returns:
    - fig: Matplotlib figure object.
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Compute the number of rows needed

    # Dynamically adjust figsize based on the number of rows and columns
    figsize = (cols * img_size[0], rows * img_size[1])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes properly and handle single row/column cases
    if rows == 1 and cols == 1:
        axes = [axes]  # Single subplot
    elif rows == 1 or cols == 1:
        axes = axes.flatten()  # Single row or single column
    else:
        axes = axes.ravel()  # Multiple rows and columns

    for i, ax in enumerate(axes):
        if i < n_images:
            # Display image and store the returned image object
            img = ax.imshow(images[i].squeeze(), cmap='gray')
            
            # Add distance as title if provided
            if distances is not None:
                ax.set_title(f'Distance: {distances[i]:.2f}', fontsize=10)
            
            # Add colorbar to the right of the image
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax)
            
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off unused subplots

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return fig



# def plot_images_with_distances(images, distances=None, cols=5, img_size=(2, 2),save_path=None):
#     """
#     Plots images with their corresponding deformation distances as titles and returns
#     the matplotlib figure object.

#     Parameters:
#     - images: List or NumPy array of images.
#     - distances: List or NumPy array of distances, aligned with images.
#     - cols: Number of columns in the plot grid.
#     - img_size: Tuple specifying the size of each subplot (width, height).

#     Returns:
#     - fig: Matplotlib figure object.
#     """
#     n_images = len(images)
#     rows = (n_images + cols - 1) // cols  # Compute the number of rows needed

#     # Dynamically adjust figsize based on the number of rows and columns
#     figsize = (cols * img_size[0], rows * img_size[1])
#     fig, axes = plt.subplots(rows, cols, figsize=figsize)

#     # Flatten axes properly and handle single row/column cases
#     if rows == 1 and cols == 1:
#         axes = [axes]  # Single subplot
#     elif rows == 1 or cols == 1:
#         axes = axes.flatten()  # Single row or single column
#     else:
#         axes = axes.ravel()  # Multiple rows and columns

#     for i, ax in enumerate(axes):
#         if i < n_images:
#             ax.imshow(images[i].squeeze(), cmap='gray')
#             if distances is not None:
#                 ax.set_title(f'Distance: {distances[i]:.2f}', fontsize=10)
#             ax.axis('off')
#         else:
#             ax.axis('off')  # Turn off unused subplots

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     # Do not call plt.show(), just return the figure for logging
#     plt.close()

def compute_slice_distance(slices_apart, slice_thickness, slice_spacing):
    if slices_apart == 0:
        return 0  # Same slice, no distance
    if slices_apart == 1:
        return slice_thickness  # Direct neighbors, no spacing
    
    # Corrected formula
    return (slices_apart - 1) * slice_spacing + slice_thickness * (slices_apart - 2)


def extract_spacing_and_compute_distances(file_paths):
        """
        Extracts patient, week, slice, spacing, and slice thickness from file paths and computes NxN distance matrix.
        If a specific week is not found, it replaces spacing and slice thickness with the average of all other weeks for the same patient.

        Args:
            file_paths (list of str): List of file paths containing slice information.
            csv_data (str): Path to the CSV file with patient and slice metadata.

        Returns:
            results (list): List of tuples with extracted metadata for each file path.
            distance_matrix (np.ndarray): NxN matrix of distances between slices.
        """
        
        # Read the CSV data
        # df = pd.read_csv(csv_data, delimiter=",")
        # df.columns = df.columns.str.strip()
        df = pd.read_csv(
            ROOT+"/ImageFlowNet/data/LUMIERE-MRinfo.csv",
            delimiter=","
        )
        # Constants
        MAX_SLICE_DISTANCE = 385.0  # Maximum slice distance in mm for 3D CT1 sequence
        MIN_SLICE_DISTANCE = 0  # Minimum slice distance in mm for 3D CT1 sequence
        sequence = 'CT1'

        results = []
        slice_numbers = []
        spacings = []
        slice_thicknesses = []

        # Extract metadata and match with CSV
        for path in file_paths:
            parts = path.split("/")
            patient = parts[-3]  # e.g., Patient-023
            week = parts[-1].replace("_", "-").replace(".png", "")  # e.g., week_000 -> week-000
            slice_number = int(parts[-2].split("_")[1])  # e.g., slice_80 -> 80

            # Match patient and week in the CSV
            match = df[(df['Patient'] == patient) & (df['Timepoint'] == week) & (df['Sequence'] == sequence)]
            
            if not match.empty:
                spacing = float(match['Spacing'].iloc[0])
                slice_thickness = float(match['Slice thickness'].iloc[0])
            else:
                # If the specific week is not found, compute averages for the patient
                patient_data = df[(df['Patient'] == patient) & (df['Sequence'] == sequence)]
                if not patient_data.empty:
                    spacing = patient_data['Spacing'].astype(float).mean()
                    slice_thickness = patient_data['Slice thickness'].astype(float).mean()
                    # print(f"Using average spacing and slice thickness for patient {patient}: "
                    #       f"spacing={spacing}, slice_thickness={slice_thickness}")
                else:
                    # Default to None if no data for the patient is available
                    spacing = None
                    slice_thickness = None
                    print(f"No data available for patient {patient} in CSV file.")

            results.append((path, patient, week, slice_number, spacing, slice_thickness))
            slice_numbers.append(slice_number)
            spacings.append(spacing)
            slice_thicknesses.append(slice_thickness)

        # Compute NxN distance matrix
        n = len(file_paths)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                elif spacings[i] is not None and spacings[j] is not None:
                    slices_apart = abs(slice_numbers[i] - slice_numbers[j])
                    slice_thickness_avg = (slice_thicknesses[i] + slice_thicknesses[j]) / 2
                    distance_matrix[i, j] = compute_slice_distance(
                        slices_apart, slice_thickness_avg, spacings[i]
                    )
                else:
                    distance_matrix[i, j] = 0  # Assign 0 if metadata is missing

        # Normalize the distance matrix
        normalized_distance_matrix = distance_matrix / MAX_SLICE_DISTANCE 

        # normalized_distance_matrix = (distance_matrix - MIN_SLICE_DISTANCE) / (MAX_SLICE_DISTANCE - MIN_SLICE_DISTANCE)

        return results, normalized_distance_matrix


def sample_N_slices_from_a_path_new(
    image_path, 
    min_gap=2, 
    max_gap=2, 
    N=3, 
    latents=False,
    base_path=None,
    equal_gaps=False  # <--- new flag
):
    """
    Samples N slices from a given image path ensuring that:
      - The consecutive slice gaps are within [min_gap, max_gap].
      - Optionally, all these gaps are exactly the same (equal_gaps=True).

    Args:
        image_path (str): Path to the reference image.
        min_gap (int, optional): Minimum gap between consecutive slices. Defaults to 2.
        max_gap (int, optional): Maximum gap between consecutive slices. Defaults to 2.
        N (int, optional): Number of slices to sample. Defaults to 3.
        latents (bool, optional): Whether to return latents or image tensors. Defaults to True.
        base_path (str, optional): Base directory path for images. 
            If None, defaults to ROOT + '/ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256'.
        equal_gaps (bool, optional): Whether the consecutive slice gaps must all be the same.
            Defaults to False.

    Returns:
        tuple: (sampled_slices, paths)
            - sampled_slices: list of slice indices that were chosen
            - paths: list of file paths for those slices
    """
    if base_path is None:
        base_path = os.path.join(
            ROOT, 'ImageFlowNet', 'data', 'brain_LUMIERE', 'LUMIERE_images_tumor-3px_256x256'
        )
    
    # Validate gap parameters
    if min_gap > max_gap:
        raise ValueError(f"min_gap ({min_gap}) cannot be greater than max_gap ({max_gap}).")
    
    # Extract patient number, week_str, and slice number from the image path
    patient_match = re.search(r"Patient-(\d+)", image_path)
    slice_match   = re.search(r"slice_(\d+)", image_path)
    week_match    = re.search(r"week_(\d+(?:-\d+)?)", image_path)
    
    if not patient_match:
        raise ValueError(f"Cannot extract patient number from path: {image_path}")
    if not slice_match:
        raise ValueError(f"Cannot extract slice number from path: {image_path}")
    
    patient_number = int(patient_match.group(1))
    slice_number   = int(slice_match.group(1))
    week_str       = week_match.group(1) if week_match else "unknown"
    
    # Construct patient_folder using base_path and patient_number
    patient_folder = os.path.join(base_path, f"Patient-{patient_number:03d}")
    if not os.path.exists(patient_folder):
        raise ValueError(f"Patient folder does not exist: {patient_folder}")
    
    all_files = os.listdir(patient_folder)
    
    # Extract all slice numbers
    slice_pattern = re.compile(r"slice_(\d+)")
    all_slices = sorted(
        int(slice_pattern.search(f).group(1)) 
        for f in all_files if slice_pattern.search(f)
    )
    
    if not all_slices:
        raise ValueError(f"No valid slices found in: {patient_folder}")
    
    # Ensure the reference slice exists
    if slice_number not in all_slices:
        raise ValueError(f"Reference slice_number {slice_number} not in folder for patient {patient_number}.")
    
    # Filter out the reference slice from valid_slices
    valid_slices = [s for s in all_slices if s != slice_number]
    
    if len(valid_slices) < N - 1:
        raise ValueError(f"Not enough slices to sample {N} slices for patient {patient_number}.")
    
    # Generate all possible combinations of N-1 slices
    possible_combinations = list(combinations(valid_slices, N - 1))
    
    # Keep only combos that: 
    #   1) include the reference slice
    #   2) have consecutive gaps in [min_gap, max_gap]
    #   3) (optionally) have equal gaps if equal_gaps=True
    valid_combinations = []
    for combo in possible_combinations:
        combo_with_ref = list(combo) + [slice_number]
        combo_with_ref_sorted = sorted(combo_with_ref)
        
        # Compute the gaps
        gaps = [
            combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
            for i in range(N-1)
        ]
        
        # Condition A: all gaps within [min_gap, max_gap]
        all_in_range = all(min_gap <= g <= max_gap for g in gaps)
        
        if not all_in_range:
            continue
        
        if equal_gaps:
            # Condition B (equal gaps): all(gaps) must be the same
            if len(set(gaps)) == 1:  # i.e., all gaps are identical
                valid_combinations.append(combo_with_ref_sorted)
        else:
            # If not requiring equal gaps, any valid gap sequence is fine
            valid_combinations.append(combo_with_ref_sorted)
    
    # If none found, we do the fallback logic in your original code
    if not valid_combinations:
        # Attempt a small fallback: extend max_gap a bit if not found
        extended_max_gap = max_gap + 2
        print(
            f"No valid combos found with gaps [{min_gap}, {max_gap}] "
            f"(equal_gaps={equal_gaps}). Trying extended_max_gap={extended_max_gap}."
        )
        for combo in possible_combinations:
            combo_with_ref = list(combo) + [slice_number]
            combo_with_ref_sorted = sorted(combo_with_ref)
            gaps = [
                combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
                for i in range(N-1)
            ]
            all_in_extended_range = all(min_gap <= g <= extended_max_gap for g in gaps)
            if all_in_extended_range:
                if equal_gaps:
                    if len(set(gaps)) == 1:
                        valid_combinations.append(combo_with_ref_sorted)
                else:
                    valid_combinations.append(combo_with_ref_sorted)
        
        if not valid_combinations:
            raise ValueError(
                f"No valid combos with min_gap={min_gap}, max_gap={max_gap} "
                f"(extended to {extended_max_gap}), equal_gaps={equal_gaps} "
                f"for patient={patient_number}."
            )
    
    # Randomly select one valid combination
    sampled_slices = rand.choice(valid_combinations)
    
    # Final check of gaps
    final_gaps = [
        sampled_slices[i+1] - sampled_slices[i] 
        for i in range(N-1)
    ]
    if not all(min_gap <= g <= max_gap for g in final_gaps):
        raise ValueError(
            f"Final chosen gaps {final_gaps} are outside the bounds "
            f"[{min_gap}, {max_gap}] for patient {patient_number}."
        )
    if equal_gaps and len(set(final_gaps)) != 1:
        raise ValueError(
            f"Final chosen slices {sampled_slices} do not have equal gaps as required."
        )
    
    # Construct paths
    paths = [
        os.path.join(
            base_path, f"Patient-{patient_number:03d}",
            f"slice_{slice_num}",
            f"week_{week_str}.png"
        )
        for slice_num in sampled_slices
    ]
    
    # if latents:
    #     # call your get_latents(...) function or similar
    #     pass
    # else:
    #     # call your load_png_as_tensor(...) function
    #     pass
    #Distance for the first and last slice
    physical_dist=extract_spacing_and_compute_distances(paths)[1][0,-1]
    return sampled_slices, paths,physical_dist




def sample_N_slices_from_a_path_new(
    image_path, 
    min_gap=2, 
    max_gap=5, 
    N=3, 
    latents=False,
    base_path=None,
    equal_gaps=True  # <--- new flag
):
    """
    Samples N slices from a given image path ensuring that:
      - The consecutive slice gaps are within [min_gap, max_gap].
      - Optionally, all these gaps are exactly the same (equal_gaps=True).

    Args:
        image_path (str): Path to the reference image.
        min_gap (int, optional): Minimum gap between consecutive slices. Defaults to 2.
        max_gap (int, optional): Maximum gap between consecutive slices. Defaults to 2.
        N (int, optional): Number of slices to sample. Defaults to 3.
        latents (bool, optional): Whether to return latents or image tensors. Defaults to True.
        base_path (str, optional): Base directory path for images. 
            If None, defaults to ROOT + '/ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256'.
        equal_gaps (bool, optional): Whether the consecutive slice gaps must all be the same.
            Defaults to False.

    Returns:
        tuple: (sampled_slices, paths)
            - sampled_slices: list of slice indices that were chosen
            - paths: list of file paths for those slices
    """
    if base_path is None:
        base_path = os.path.join(
            ROOT, 'ImageFlowNet', 'data', 'brain_LUMIERE', 'LUMIERE_images_tumor-3px_256x256'
        )
    
    # Validate gap parameters
    if min_gap > max_gap:
        raise ValueError(f"min_gap ({min_gap}) cannot be greater than max_gap ({max_gap}).")
    
    # Extract patient number, week_str, and slice number from the image path
    patient_match = re.search(r"Patient-(\d+)", image_path)
    slice_match   = re.search(r"slice_(\d+)", image_path)
    week_match    = re.search(r"week_(\d+(?:-\d+)?)", image_path)
    
    if not patient_match:
        raise ValueError(f"Cannot extract patient number from path: {image_path}")
    if not slice_match:
        raise ValueError(f"Cannot extract slice number from path: {image_path}")
    
    patient_number = int(patient_match.group(1))
    slice_number   = int(slice_match.group(1))
    week_str       = week_match.group(1) if week_match else "unknown"
    
    # Construct patient_folder using base_path and patient_number
    patient_folder = os.path.join(base_path, f"Patient-{patient_number:03d}")
    if not os.path.exists(patient_folder):
        raise ValueError(f"Patient folder does not exist: {patient_folder}")
    
    all_files = os.listdir(patient_folder)
    
    # Extract all slice numbers
    slice_pattern = re.compile(r"slice_(\d+)")
    all_slices = sorted(
        int(slice_pattern.search(f).group(1)) 
        for f in all_files if slice_pattern.search(f)
    )
    
    if not all_slices:
        raise ValueError(f"No valid slices found in: {patient_folder}")
    
    # Ensure the reference slice exists
    if slice_number not in all_slices:
        raise ValueError(f"Reference slice_number {slice_number} not in folder for patient {patient_number}.")
    
    # Filter out the reference slice from valid_slices
    valid_slices = [s for s in all_slices if s != slice_number]
    
    if len(valid_slices) < N - 1:
        raise ValueError(f"Not enough slices to sample {N} slices for patient {patient_number}.")
    
    # Generate all possible combinations of N-1 slices
    possible_combinations = list(combinations(valid_slices, N - 1))
    
    # Keep only combos that: 
    #   1) include the reference slice
    #   2) have consecutive gaps in [min_gap, max_gap]
    #   3) (optionally) have equal gaps if equal_gaps=True
    valid_combinations = []
    for combo in possible_combinations:
        combo_with_ref = list(combo) + [slice_number]
        combo_with_ref_sorted = sorted(combo_with_ref)
        
        # Compute the gaps
        gaps = [
            combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
            for i in range(N-1)
        ]
        
        # Condition A: all gaps within [min_gap, max_gap]
        all_in_range = all(min_gap <= g <= max_gap for g in gaps)
        
        if not all_in_range:
            continue
        
        if equal_gaps:
            # Condition B (equal gaps): all(gaps) must be the same
            if len(set(gaps)) == 1:  # i.e., all gaps are identical
                valid_combinations.append(combo_with_ref_sorted)
        else:
            # If not requiring equal gaps, any valid gap sequence is fine
            valid_combinations.append(combo_with_ref_sorted)
    
    # If none found, we do the fallback logic in your original code
    if not valid_combinations:
        # Attempt a small fallback: extend max_gap a bit if not found
        extended_max_gap = max_gap + 2
        print(
            f"No valid combos found with gaps [{min_gap}, {max_gap}] "
            f"(equal_gaps={equal_gaps}). Trying extended_max_gap={extended_max_gap}."
        )
        for combo in possible_combinations:
            combo_with_ref = list(combo) + [slice_number]
            combo_with_ref_sorted = sorted(combo_with_ref)
            gaps = [
                combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
                for i in range(N-1)
            ]
            all_in_extended_range = all(min_gap <= g <= extended_max_gap for g in gaps)
            if all_in_extended_range:
                if equal_gaps:
                    if len(set(gaps)) == 1:
                        valid_combinations.append(combo_with_ref_sorted)
                else:
                    valid_combinations.append(combo_with_ref_sorted)
        
        if not valid_combinations:
            raise ValueError(
                f"No valid combos with min_gap={min_gap}, max_gap={max_gap} "
                f"(extended to {extended_max_gap}), equal_gaps={equal_gaps} "
                f"for patient={patient_number}."
            )
    
    # Randomly select one valid combination
    sampled_slices = rand.choice(valid_combinations)
    
    # Final check of gaps
    final_gaps = [
        sampled_slices[i+1] - sampled_slices[i] 
        for i in range(N-1)
    ]
    if not all(min_gap <= g <= max_gap for g in final_gaps):
        raise ValueError(
            f"Final chosen gaps {final_gaps} are outside the bounds "
            f"[{min_gap}, {max_gap}] for patient {patient_number}."
        )
    if equal_gaps and len(set(final_gaps)) != 1:
        raise ValueError(
            f"Final chosen slices {sampled_slices} do not have equal gaps as required."
        )
    
    # Construct paths
    paths = [
        os.path.join(
            base_path, f"Patient-{patient_number:03d}",
            f"slice_{slice_num}",
            f"week_{week_str}.png"
        )
        for slice_num in sampled_slices
    ]
    
    # if latents:
    #     # call your get_latents(...) function or similar
    #     pass
    # else:
    #     # call your load_png_as_tensor(...) function
    #     pass
    #Distance for the first and last slice
    physical_dist=extract_spacing_and_compute_distances(paths)[1][0,-1]
    return sampled_slices, paths,physical_dist




def sample_N_slices_from_a_path(
    image_path, 
    min_gap=3, 
    max_gap=80, 
    N=3, 
    latents=False,
    base_path=None,
    equal_gaps=False  # <--- new flag
):
    """
    Samples N slices from a given image path ensuring that:
      - The consecutive slice gaps are within [min_gap, max_gap].
      - Optionally, all these gaps are exactly the same (equal_gaps=True).

    Args:
        image_path (str): Path to the reference image.
        min_gap (int, optional): Minimum gap between consecutive slices. Defaults to 2.
        max_gap (int, optional): Maximum gap between consecutive slices. Defaults to 2.
        N (int, optional): Number of slices to sample. Defaults to 3.
        latents (bool, optional): Whether to return latents or image tensors. Defaults to True.
        base_path (str, optional): Base directory path for images. 
            If None, defaults to ROOT + '/ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256'.
        equal_gaps (bool, optional): Whether the consecutive slice gaps must all be the same.
            Defaults to False.

    Returns:
        tuple: (sampled_slices, paths)
            - sampled_slices: list of slice indices that were chosen
            - paths: list of file paths for those slices
    """
    if base_path is None:
        base_path = os.path.join(
            ROOT, 'ImageFlowNet', 'data', 'brain_LUMIERE', 'LUMIERE_images_tumor-3px_256x256'
        )
    
    # Validate gap parameters
    if min_gap > max_gap:
        raise ValueError(f"min_gap ({min_gap}) cannot be greater than max_gap ({max_gap}).")
    
    # Extract patient number, week_str, and slice number from the image path
    patient_match = re.search(r"Patient-(\d+)", image_path)
    slice_match   = re.search(r"slice_(\d+)", image_path)
    week_match    = re.search(r"week_(\d+(?:-\d+)?)", image_path)
    
    if not patient_match:
        raise ValueError(f"Cannot extract patient number from path: {image_path}")
    if not slice_match:
        raise ValueError(f"Cannot extract slice number from path: {image_path}")
    
    patient_number = int(patient_match.group(1))
    slice_number   = int(slice_match.group(1))
    week_str       = week_match.group(1) if week_match else "unknown"
    
    # Construct patient_folder using base_path and patient_number
    patient_folder = os.path.join(base_path, f"Patient-{patient_number:03d}")
    if not os.path.exists(patient_folder):
        raise ValueError(f"Patient folder does not exist: {patient_folder}")
    
    all_files = os.listdir(patient_folder)
    
    # Extract all slice numbers
    slice_pattern = re.compile(r"slice_(\d+)")
    all_slices = sorted(
        int(slice_pattern.search(f).group(1)) 
        for f in all_files if slice_pattern.search(f)
    )
    
    if not all_slices:
        raise ValueError(f"No valid slices found in: {patient_folder}")
    
    # Ensure the reference slice exists
    if slice_number not in all_slices:
        raise ValueError(f"Reference slice_number {slice_number} not in folder for patient {patient_number}.")
    
    # Filter out the reference slice from valid_slices
    valid_slices = [s for s in all_slices if s != slice_number]
    
    if len(valid_slices) < N - 1:
        raise ValueError(f"Not enough slices to sample {N} slices for patient {patient_number}.")
    
    # Generate all possible combinations of N-1 slices
    possible_combinations = list(combinations(valid_slices, N - 1))
    
    # Keep only combos that: 
    #   1) include the reference slice
    #   2) have consecutive gaps in [min_gap, max_gap]
    #   3) (optionally) have equal gaps if equal_gaps=True
    valid_combinations = []
    for combo in possible_combinations:
        combo_with_ref = list(combo) + [slice_number]
        combo_with_ref_sorted = sorted(combo_with_ref)
        
        # Compute the gaps
        gaps = [
            combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
            for i in range(N-1)
        ]
        
        # Condition A: all gaps within [min_gap, max_gap]
        all_in_range = all(min_gap <= g <= max_gap for g in gaps)
        
        if not all_in_range:
            continue
        
        if equal_gaps:
            # Condition B (equal gaps): all(gaps) must be the same
            if len(set(gaps)) == 1:  # i.e., all gaps are identical
                valid_combinations.append(combo_with_ref_sorted)
        else:
            # If not requiring equal gaps, any valid gap sequence is fine
            valid_combinations.append(combo_with_ref_sorted)
    
    # If none found, we do the fallback logic in your original code
    if not valid_combinations:
        # Attempt a small fallback: extend max_gap a bit if not found
        extended_max_gap = max_gap + 2
        print(
            f"No valid combos found with gaps [{min_gap}, {max_gap}] "
            f"(equal_gaps={equal_gaps}). Trying extended_max_gap={extended_max_gap}."
        )
        for combo in possible_combinations:
            combo_with_ref = list(combo) + [slice_number]
            combo_with_ref_sorted = sorted(combo_with_ref)
            gaps = [
                combo_with_ref_sorted[i+1] - combo_with_ref_sorted[i] 
                for i in range(N-1)
            ]
            all_in_extended_range = all(min_gap <= g <= extended_max_gap for g in gaps)
            if all_in_extended_range:
                if equal_gaps:
                    if len(set(gaps)) == 1:
                        valid_combinations.append(combo_with_ref_sorted)
                else:
                    valid_combinations.append(combo_with_ref_sorted)
        
        if not valid_combinations:
            raise ValueError(
                f"No valid combos with min_gap={min_gap}, max_gap={max_gap} "
                f"(extended to {extended_max_gap}), equal_gaps={equal_gaps} "
                f"for patient={patient_number}."
            )
    
    # Randomly select one valid combination
    sampled_slices = rand.choice(valid_combinations)
    
    # Final check of gaps
    final_gaps = [
        sampled_slices[i+1] - sampled_slices[i] 
        for i in range(N-1)
    ]
    if not all(min_gap <= g <= max_gap for g in final_gaps):
        raise ValueError(
            f"Final chosen gaps {final_gaps} are outside the bounds "
            f"[{min_gap}, {max_gap}] for patient {patient_number}."
        )
    if equal_gaps and len(set(final_gaps)) != 1:
        raise ValueError(
            f"Final chosen slices {sampled_slices} do not have equal gaps as required."
        )
    
    # Construct paths
    paths = [
        os.path.join(
            base_path, f"Patient-{patient_number:03d}",
            f"slice_{slice_num}",
            f"week_{week_str}.png"
        )
        for slice_num in sampled_slices]

    if latents:
        # Get latents for the selected paths
        latents = get_latents(None, paths).squeeze(1)
        out = latents.reshape(latents.size(0), -1)
    else:
        # Load images as tensors
        out = torch.stack([load_png_as_tensor(path) for path in paths])


    return out, sampled_slices


def sample_slices_from_ref(image_path, N=5, latents=False, 
                  base_path=os.path.join(ROOT, 
                                         'ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256')
                                         ):
    pnum = int(re.search(r"Patient-(\d+)", image_path).group(1))
    wstr = re.search(r"week_(\d+(?:-\d+)?)", image_path).group(1)
    snum = int(re.search(r"slice_(\d+)", image_path).group(1))
    folder = '/'.join(image_path.split('/')[:-2])
    fs = [int(re.search(r"slice_(\d+)", f).group(1)) for f in os.listdir(folder) if re.search(r"slice_(\d+)", f)]
    if snum not in fs: raise ValueError("Reference slice not found.")
    fs.remove(snum)
    if len(fs) < N-1: raise ValueError("Not enough slices to sample.")
    chosen = rand.sample(fs, N-1)
    chosen.append(snum)
    chosen.sort()
    paths = [f"{base_path}/Patient-{pnum:03d}/slice_{s}/week_{wstr}.png" for s in chosen]
    return  chosen,paths


def save_as_memmap_with_minmax(array, save_path):
    """
    Save a numpy array as a memory-mapped file and save its min/max values.
    
    Args:
        array (np.ndarray): Input numpy array to save
        save_path (str): Path where to save the memmap file
        
    Returns:
        tuple: (memmap_path, minmax_path) - paths to the saved files
    """
    # Ensure the directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get the base path without extension
    base_path = os.path.splitext(save_path)[0]
    
    # Define paths for memmap and min-max files
    memmap_file_path = f"{base_path}.memmap"
    minmax_path = f"{base_path}_min_max.npy"
    
    # Calculate min and max
    array_min = np.min(array)
    array_max = np.max(array)
    N=array.shape[0]
    # Save min-max values
    np.save(minmax_path, np.array([array_min, array_max]))
    

    # Remove existing memmap file if it exists
    if os.path.exists(memmap_file_path):
        os.remove(memmap_file_path)

    print(f"Saving distance matrix to {memmap_file_path} as a memory-mapped file...")
    # Create a memory-mapped file
    memmap = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=(N, N))

    # Write data one row at a time
    for i in range(N):
        if i % 1000 == 0 or i == N - 1:
            print(f"Writing row {i + 1} of {N}...")
        memmap[i, :] = array[i, :].astype('float32')
        memmap.flush()  # Ensure the row is written to disk

    # Delete memmap object to flush remaining data and close the file
    
    return memmap_file_path, minmax_path


if __name__ == "__main__":
    path='/ivi/zfs/s0/original_homes/mislam/ImageFlowNet/data/brain_LUMIERE/LUMIERE_images_tumor-3px_256x256/Patient-015/slice_156/week_126.png'
    slices, id = sample_slices_from_ref(path,N=5)
    print(id)
    dummy_var=np.random.rand(256,256)
    save_as_memmap_with_minmax(dummy_var,'/ivi/zfs/s0/original_homes/mislam/ae_isomap_study/saved_full_dataset/test')