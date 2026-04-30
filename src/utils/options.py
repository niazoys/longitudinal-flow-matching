import argparse

def ae_arg():
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--log', type=eval, default=True)
    parser.add_argument('--enable_progress_bar', type=eval, default=True)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--loss_switch', type=int, default=15)
    # Test and Re-run
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--save_results', type=bool, default=False)
    # parser.add_argument('--resume_ckpt', type=str, default="/ae_isomap_study/logs/AE-ISOMAP-IMPLICIT/rgx4o3k5/checkpoints/last.ckpt")

    parser.add_argument('--resume_ckpt', type=str, default=None)

    # Train settings
    parser.add_argument('--train_augm', type=eval, default=False)

    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default="brain_gbm")
    parser.add_argument('--dataset_path', type=str, default='/ivi/zfs/s0/original_homes/mislam/ImageFlowNet/data/brain_LUMIERE')# This would be handled for snellius
    parser.add_argument('--image_folder', type=str, default='LUMIERE_images_tumor-3px_256x256')
    parser.add_argument('--use_only_pathological_case', type=eval, default=True)
    parser.add_argument('--train-val-test-ratio', default='8.0:0.5:1.5', type=str)
    parser.add_argument('--max_training_samples', type=eval, default=1e6)


    # For autoencoder this should a big number of all the slices are included 
    # for NeuralODE this should be something like 20
    parser.add_argument('--max_slice_per_patient', type=eval, default=200)
    
    # Model settings
    parser.add_argument('--image_size', default=(256, 256), type=eval)
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_channels', type=int, default=64, help='Number of channels in the model')
    parser.add_argument('--num_res_blocks', type=int, default=1, help='Number of residual blocks in each downsample block')
    parser.add_argument('--channel_mult', type=eval, default=(2, 4, 4, 4, 4, 4, 4), help='Channel multiplier at each resolution')
    parser.add_argument('--resblock_updown', type=bool, default=True)

    parser.add_argument('--learn_sigma', type=bool, default=False, help='Flag to learn sigma (noise level)')
    parser.add_argument('--num_classes', type=eval, default=None, help='Use class conditioning')
    parser.add_argument('--use_checkpoint', type=bool, default=False, help='Use gradient checkpointing')
    parser.add_argument('--attention_resolutions', type=str, default='64,32,16,8,4,2,1', help='Comma-separated list of resolutions with attention')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_head_channels', type=int, default=16, help='Number of channels per attention head')
    parser.add_argument('--num_heads_upsample', type=int, default=-1, help='Number of heads during upsample')
    parser.add_argument('--use_scale_shift_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--use_fp16', type=bool, default=False)
    parser.add_argument('--use_new_attention_order', type=bool, default=True)

    # Swithc between UNet and AE and Time paramterized models
    parser.add_argument('--use_skip_connection', type=bool, default=False)
    parser.add_argument('--use_time_embed', type=bool, default=False)

    
    # Disentanglement settings
    parser.add_argument('--use_disentangle', type=bool, default=False)
    # Train VAE instead of AE
    parser.add_argument('--make_vae', type=bool, default=False)
    parser.add_argument('--normalize_zero_to_one', type=bool, default=True) # otherwise -1 to 1

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int)
    
    # Base Losses    
    parser.add_argument('--w_mse', type=float, default=1.0, help='Weight for L2 loss')   
    parser.add_argument('--w_ssim', type=float, default=0.5, help='Weight for SSIM loss') 
    parser.add_argument('--w_edge', type=float, default=0.0, help='Weight for edge loss')
    # Additional Losses 1
    parser.add_argument('--w_iso', type=float, default=0.0, help='global loss weight')
    parser.add_argument('--w_gm', type=float, default=0.0, help='graph matching loss weight')
    parser.add_argument('--w_interp', type=float, default=0.0, help='Interpolation loss weight')

    # Additional Losses 2
    parser.add_argument('--w_relax', type=float, default=0.0, help='Relaxed distortion loss weight') # run with 0.1

    parser.add_argument('--w_bend', type=float, default=0.0, help='Latent Space flatning loss weight') # run with 1e-5
    parser.add_argument('--additional_dist', type=str, default='loss_residual', help='additional distance w/ registration')# w_euclid,loss_residual,None
    parser.add_argument('--w_additional_dist', type=float, default=0.0, help='additional distance weight') # run with 0.5
    parser.add_argument('--w_jcob', type=float, default=0.0, help='Jaccobian loss weight') # run with 1e-4
    # Misc
    parser.add_argument('--load_in_triplet', type=str, default='triplet', help='Load data in triplet')
    parser.add_argument('--log_interp', type=int, default=10, help='log interpolation')
    parser.add_argument('--log_edge_loss', type=bool, default=False, help='Log edge loss during validation and testing')
    parser.add_argument('--combinations', type=eval, default=None)
    args, unknown = parser.parse_known_args()

    # Overwrite default settings with values from combinations
    if args.combinations is not None:
        for key, value in args.combinations.items():
            setattr(args, key, value)

    return args

def get_neural_ode_args():
    parser = argparse.ArgumentParser()
    
    # Define command-line arguments for NeuralODEModel
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--alpha", type=float, default=1, help="Regularization parameter alpha")
    parser.add_argument("--input_dim", type=int, default=4096, help="Dimension of the input layer")
    parser.add_argument("--time_dim", type=int, default=128, help="Dimension of time embedding")
    parser.add_argument("--direction_dim", type=int, default=4096, help="Dimension of direction embedding")
    parser.add_argument("--hidden_dim", type=int, default=4096, help="Dimension of hidden layers")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-12, help="Weight decay")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument('--image_folder', type=str, default='LUMIERE_images_tumor1200px_256x256')
    args, unknown = parser.parse_known_args()
    return args

def get_vae_loss_args():
    parser = argparse.ArgumentParser()
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-l', '--loss',
                       default='btcvae',
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default='gaussian',
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=10000,
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

     # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=10,
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=0,
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=50,
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=100,
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=6,
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=5e-5,
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=0.7,
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=10,
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=5,
                        help="Weight of the TC term (beta in the paper).")
    
    args = parser.parse_args()
    return args

def get_isometry_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=50, help='number of epochs')
    parser.add_argument("--lr", type=float,default=1e-5, help='learning rate') #this was 0.0001
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--n_steps",type=int,default=15,help="number of steps")
    parser.add_argument("--seed",type=int,default=10,help="seed")

    parser.add_argument("--n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--hidden_nf",type=int,default=512,help="hidden units")  #this was 64
    parser.add_argument("--warmup",type=int,default=10,help="warmup")
    parser.add_argument("--alpha1",type=float,default=2,help="global loss weight")
    parser.add_argument("--alpha2",type=float,default=2,help="graph matching loss weight") #this was 5
    parser.add_argument("--alpha3",type=float,default=1.0,help="low dimensional loss weight") #this was 1
    parser.add_argument("--alpha4",type=float,default=0.1,help="jacobian loss weight") #this was 0.0010
    parser.add_argument("--alpha5",type=float,default=0.0,help="inverse loss weight") #this was 0.001
    parser.add_argument("--alpha6",type=float,default=0.25,help="interpolation supervision loss") #this was 0.001
    parser.add_argument("--n_dim",type=int,default=1024,help="number of dimensions") #this was 1
    
    parser.add_argument("--n_neighbors",type=int,default=3,help="number of neighbors")
    parser.add_argument("--batch_size",type=int,default=8,help="batch size")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    parser.add_argument("--use_latents",type=bool,default=True,help="use latents")
    parser.add_argument("--use_img_geodist",type=bool,default=False, help="This will use PHATE or HeatGeo on images")
    parser.add_argument("--geodist_model",type=str,default='ae_small',help="geodist model") # this will be ignored if use_img_geodist is True
    parser.add_argument('--normalize_geodist', type=bool, default=True, help='normalize geodist')
    parser.add_argument('--use_custom_code', type=bool, default=False, help='use custom code for connecting components in the graph')
    parser.add_argument('--d_weight', type=eval, default=[1.0,0.0,0.0], help='distance weight')# Feature, Slice distance, Registration distance
    parser.add_argument('--additional_dist', type=str, default='w_euclid', help='additional distance')# w_euclid,loss_residual,None
    parser.add_argument('--additional_distance_weight', type=float, default=0.5, help='additional distance weight') # run with 0.5
    parser.add_argument('--low_bend_supervised',type=int,default=False)
    parser.add_argument("--use_pca",type=int,default=False)
    parser.add_argument("--pca_rank",type=int,default=1024)
    parser.add_argument('--test_ckpt', type=str, default=None) #4096
    parser.add_argument('--save_n_epoch', type=int, default=25)
    parser.add_argument('--log_interp', type=int, default=10)
    args = parser.parse_known_args()[0]
    return args

def get_isometry_cnn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=500, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.00001, help='learning rate') # this was 0.0001
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--n_steps",type=int,default=10,help="number of steps")
    parser.add_argument("--seed",type=int,default=11,help="seed")
    parser.add_argument("--n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--alpha1",type=float,default=5.0,help="global loss weight") #this was 5 
    parser.add_argument("--alpha2",type=float,default=5,help="graph matching loss weight") # this was 5
    parser.add_argument("--alpha3",type=float,default=1.0,help="low dimensional loss weight") # this was 1
    parser.add_argument("--alpha4",type=float,default=0.0001,help="jacobian loss weight") # this was 0.001
    parser.add_argument("--n_dim",type=int,default=4096,help="number of dimensions") #this was 1
    parser.add_argument("--hidden_nf",type=eval,default=512 ,help="hidden units")  
    parser.add_argument("--n_neighbors",type=int,default=5,help="number of neighbors")
    parser.add_argument("--batch_size",type=int,default=32,help="batch size")
    parser.add_argument("--warmup",type=int,default=60,help="warmup")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    parser.add_argument("--use_latents",type=bool,default=True,help="use latents")
    parser.add_argument("--height",type=int,default=8,help="height")
    parser.add_argument("--width",type=int,default=8,help="width")
    parser.add_argument("--channels",type=int,default=256,help="channels")
    args = parser.parse_args()
    return args

def get_isometry_args_original():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=2000, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.0001, help='learning rate')
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--n_steps",type=int,default=10,help="number of steps")
    parser.add_argument("--seed",type=int,default=0,help="seed")
    parser.add_argument("--n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--alpha1",type=float,default=5.0,help="global loss weight")
    parser.add_argument("--alpha2",type=float,default=5.0,help="graph matching loss weight")
    parser.add_argument("--alpha3",type=float,default=1.0,help="low dimensional loss weight")
    parser.add_argument("--alpha4",type=float,default=0.001,help="jacobian loss weight")
    parser.add_argument("--n_dim",type=int,default=1,help="number of dimensions")
    parser.add_argument("--hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--n_neighbors",type=int,default=5,help="number of neighbors")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    parser.add_argument("--warmup",type=int,default=50,help="warmup")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    args = parser.parse_args()
    return args

def get_VAE_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=2000, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.0001, help='learning rate')
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--seed",type=int,default=0,help="seed")
    parser.add_argument("--n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--beta",type=float,default=1.0,help="beta")
    parser.add_argument("--n_dim",type=int,default=1,help="number of dimensions")
    parser.add_argument("--hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--n_neighbors",type=int,default=5,help="number of neighbors")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    args = parser.parse_args()
    return args

def get_CFM_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=5000, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.0005, help='learning rate')
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--seed",type=int,default=0,help="seed")
    parser.add_argument("--n_layers",type=int,default=10,help="number of layers")
    parser.add_argument("--hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    args = parser.parse_args()
    return args

def get_PFM_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=5000, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.0005, help='learning rate')
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--seed",type=int,default=0,help="seed")
    parser.add_argument("--varphi_n_steps",type=int,default=10,help="number of steps")
    parser.add_argument("--varphi_n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--varphi_hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--n_layers",type=int,default=10,help="number of layers")
    parser.add_argument("--hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    parser.add_argument("--path",type=str,default="models/interpolation/run-20240728_130602-cxjuyx7v/files/parameters",help="path to CNF model")
    args = parser.parse_args()
    return args

def get_d_PFM_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,default=5000, help='number of epochs')
    parser.add_argument("--lr", type=float,default=0.0005, help='learning rate')
    parser.add_argument("--shuffle",type=bool,default=True,help="shuffle true or false")
    parser.add_argument("--train_size",type=float,default=0.8,help="train size")
    parser.add_argument("--seed",type=int,default=0,help="seed")
    parser.add_argument("--varphi_n_steps",type=int,default=10,help="number of steps")
    parser.add_argument("--varphi_n_layers",type=int,default=5,help="number of layers")
    parser.add_argument("--varphi_hidden_nf",type=int,default=64,help="hidden units")
    parser.add_argument("--n_layers",type=int,default=10,help="number of layers")
    parser.add_argument("--hidden_nf",type=int,default=16,help="hidden units")
    parser.add_argument("--split",type=bool,default=True,help="split true or false")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size")
    parser.add_argument("--n_dim",type=int,default=1,help="number of dimensions")
    parser.add_argument("--path",type=str,default="models/interpolation/run-20240728_130602-cxjuyx7v/files/parameters",help="path to CNF model")
    args = parser.parse_args()
    return args
