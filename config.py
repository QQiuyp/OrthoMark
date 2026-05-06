# =============================================================================
# OrthoMark Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
epochs = 200000
lr = 1e-3
batch_size = 50
cropsize = 128
message_length = 64

# -----------------------------------------------------------------------------
# Validation/Test Settings
# -----------------------------------------------------------------------------
batchsize_val = 16
cropsize_val = 128
test_message_length = 64
val_freq = 1
test_runs = 1
test_checkpoint_path = "./OrthoMark/results/QIM/models/400.pt"
test_output_root = "test_results"
test_save_intermediate = False
test_vis_interval = 100
test_seed = None

# -----------------------------------------------------------------------------
# Data Paths
# -----------------------------------------------------------------------------
TRAIN_PATH = ''
VAL_PATH = ''
format_train = 'jpg'
format_val = 'png'

# -----------------------------------------------------------------------------
# Checkpoint Settings
# -----------------------------------------------------------------------------
MODEL_PATH = 'results'
PROJECT_NAME = "QIM"
CONTINUE_PATH = 'QIM/'
CONTINUE_EPOCH = 400
tain_next = True
SAVE_freq = 1

# -----------------------------------------------------------------------------
# Device Settings
# -----------------------------------------------------------------------------
ndevice = 0  # GPU device index

# -----------------------------------------------------------------------------
# QIM Watermark Settings
# -----------------------------------------------------------------------------
qim_Delta = 2          # QIM quantization step size
qim_seed = 12345       # Random seed for carrier generation

# -----------------------------------------------------------------------------
# Optimizer Settings (Adversarial/Embedding)
# -----------------------------------------------------------------------------
adv_opt_name = "adam"
adv_lr = 1e-2
adv_lr_final = 1e-5
adv_warmup_ratio = 0.1
adv_grad_clip = 0.0

# -----------------------------------------------------------------------------
# Loss Weights
# -----------------------------------------------------------------------------
adv_mse_w = 300        # MSE weight for image quality
acc_w = 1              # Accuracy weight

# -----------------------------------------------------------------------------
# Decoder Optimizer Settings
# -----------------------------------------------------------------------------
dec_lr = 1e-5
dec_grad_clip = 0

# -----------------------------------------------------------------------------
# Training Steps
# -----------------------------------------------------------------------------
joint_steps = 500           # Training inner loop steps
test_joint_steps = 1000     # Test inner loop steps

# -----------------------------------------------------------------------------
# Loss Mode Settings
# -----------------------------------------------------------------------------
loss_mode = "qim"           # Training loss mode: "qim" / "mse" / "mix"
test_loss_mode = "qim"      # Test loss mode: "qim" / "mse"
mix_epochs = 0              # Epochs for MSE->QIM transition (for "mix" mode)

# -----------------------------------------------------------------------------
# Noise/Attack Settings
# -----------------------------------------------------------------------------
noise_type = "NGMIX"        # Noise type: "JP", "NGMIX", etc.

# -----------------------------------------------------------------------------
# Carrier Settings
# -----------------------------------------------------------------------------
V_mode = "ortho"            # Carrier mode: "ortho" / "rand_unit" / "rand_unit_corr"
V_rho = 0.5                 # Correlation coefficient (for "rand_unit_corr")

# -----------------------------------------------------------------------------
# Other Settings
# -----------------------------------------------------------------------------
target_psnr = 41
method_name = "OrthoMark"
mode = "qim"                # Model mode (used in some conditionals)
