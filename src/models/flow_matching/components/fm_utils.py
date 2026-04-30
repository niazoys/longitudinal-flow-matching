import torch
import numpy as np

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

def time_weighted_mse_loss(pred, true, time):
    
    # print(time.shape, pred.shape, true.shape)
    # ipdb.set_trace()
    return torch.mean( (time.view(-1,1) *(pred - true) )** 2)

def l1_loss(pred, true):
    return torch.mean(torch.abs(pred - true))

def metrics_calculation(pred, true, metrics=['mse_loss'], normalize=False):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().squeeze().numpy()
        true = true.detach().cpu().squeeze().numpy()

    loss_D = {key: None for key in metrics}
    if normalize:
        # We normalize the data to be between 0 and 1
        # for each sample individually
        
        # Create copies to store normalized results
        pred_normalized = np.zeros_like(pred)
        true_normalized = np.zeros_like(true)
        
        # Process each sample independently
        for i in range(pred.shape[0]):
            # Get current sample
            pred_sample = pred[i]
            true_sample = true[i]
            
            # Compute min/max for this sample
            pred_min, pred_max = pred_sample.min(), pred_sample.max()
            true_min, true_max = true_sample.min(), true_sample.max()
            
            # Avoid division by zero
            pred_range = max(pred_max - pred_min, 1e-8)
            true_range = max(true_max - true_min, 1e-8)
            
            # Normalize this sample
            pred_normalized[i] = (pred_sample - pred_min) / pred_range
            true_normalized[i] = (true_sample - true_min) / true_range
        
        # Replace with normalized versions
        pred = pred_normalized
        true = true_normalized
    
    for metric in metrics:
        if metric == 'mse_loss':
            loss_D['mse_loss'] = np.mean((pred - true)**2)
        if metric == 'l1_loss':
            loss_D['l1_loss'] = np.mean(np.abs(pred - true))

    return loss_D

def positional_encoding_tensor(time_tensor, num_frequencies, base=0.012):
    time_tensor = time_tensor.clamp(0, 1).unsqueeze(1)
    frequencies = torch.pow(base, -torch.arange(0, num_frequencies, dtype=torch.float32) / num_frequencies).to(time_tensor.device)
    angles = time_tensor * frequencies
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    pos_encoding = torch.stack((sine, cosine), dim=-1)
    pos_encoding = pos_encoding.flatten(start_dim=2)
    pos_encoding = (pos_encoding + 1) / 2
    return pos_encoding