import time, wandb
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import logging
memory_logger = logging.getLogger("ticl")

import ticl.utils as utils
from ticl.utils import ExponentialLR, ReduceLROnSpike, init_dist, get_autocast_context, IGNORE_INDEX

import pdb


def eval_criterion(criterion, targets, output, device, n_out, batch_info=None):
    """
    Evaluate the criterion based on model output and targets.
    
    Parameters:
    -----------
    criterion : nn.Module
        Loss function to use
    targets : torch.Tensor
        Target values
    output : torch.Tensor or dict
        Model output (either tensor or dictionary for semantic models)
    device : str
        Device to use
    n_out : int
        Number of output classes
    batch_info : dict, optional
        Additional batch information (for semantic models)
        
    Returns:
    --------
    tuple
        (Loss, NaN share)
    """
    
    # Log output structure
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                memory_logger.debug(f"Model output['{k}']: shape={v.shape}, dtype={v.dtype}")
                if k == 'class_logits':
                    # Log more details about class predictions
                    class_probs = torch.softmax(v, dim=-1)
                    predicted_classes = torch.argmax(class_probs, dim=-1)
                    memory_logger.debug(f"Predicted classes: {predicted_classes}")
                    memory_logger.debug(f"Max class probability: {torch.max(class_probs, dim=-1)[0]}")
    else:
        memory_logger.debug(f"Model output: shape={output.shape}, dtype={output.dtype}")
        # Log class predictions for standard output
        if len(output.shape) > 1 and output.shape[-1] > 1:  # Multi-class case
            class_probs = torch.softmax(output, dim=-1)
            predicted_classes = torch.argmax(class_probs, dim=-1)
            memory_logger.debug(f"Predicted classes: {predicted_classes}")
    
    # Log batch_info if present
    if batch_info is not None:
        if 'semantic_targets' in batch_info:
            sem_targets = batch_info['semantic_targets']
    
    # Check if this is a semantic model with dictionary output
    is_semantic_model = isinstance(output, dict) and 'class_logits' in output and 'semantic_logits' in output
    
    if is_semantic_model:
        # For semantic models with our custom loss
        from ticl.models.semantic_aware_model import SemanticConsistencyLoss
        if isinstance(criterion, SemanticConsistencyLoss):
            # Extract semantic targets from batch info
            semantic_targets = None
            
            # Check if batch contains semantic targets
            has_semantic_features = False
            if batch_info is not None:
                if 'semantic_targets' in batch_info:
                    semantic_targets = batch_info['semantic_targets'].to(device)
                    has_semantic_features = True
                
            # Create target dictionary
            target_dict = {
                'class_targets': targets.to(device).long(),
                'semantic_targets': semantic_targets
            }
            
            # Log targets dictionary
            memory_logger.debug(f"Target dict - class_targets: {target_dict['class_targets'].shape}")
            if semantic_targets is not None:
                memory_logger.debug(f"Target dict - semantic_targets: {target_dict['semantic_targets'].shape}")
                
                # Check for valid semantic targets
                valid_count = (semantic_targets != -100).sum().item()
                if valid_count == 0:
                    memory_logger.debug("WARNING: Semantic targets present but all are -100 (ignored)")
            
            # Compute loss
            memory_logger.debug("Computing loss with SemanticConsistencyLoss")
            loss = criterion(output, target_dict)
            memory_logger.debug(f"Loss value: {loss.item()}")
            
            # Return loss as a tensor for compatibility
            return loss.unsqueeze(0).unsqueeze(0), 0.0
        else:
            # Fallback to just using class logits with standard loss
            memory_logger.debug("Using standard loss with semantic model class_logits output")
            output = output['class_logits']
    
    # Standard loss functions
    if isinstance(criterion, nn.GaussianNLLLoss):
        memory_logger.debug(f"Using GaussianNLLLoss")
        assert output.shape[-1] == 2, \
            'need to write a little bit of code to handle multiple regression targets at once'

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        loss_name = 'MSELoss' if isinstance(criterion, nn.MSELoss) else 'BCEWithLogitsLoss'
        memory_logger.debug(f"Using {loss_name}")
        losses = criterion(output.flatten(), targets.to(device).flatten())
    elif isinstance(criterion, nn.CrossEntropyLoss):
        memory_logger.debug(f"Using CrossEntropyLoss")
        # Debug reshape dimensions
        reshaped_output = output.reshape(-1, n_out)[:, :int(targets.max()) + 1]
        flattened_targets = targets.to(device).long().flatten()
        memory_logger.debug(f"Reshaped output for CE loss: {reshaped_output.shape}")
        memory_logger.debug(f"Flattened targets for CE loss: {flattened_targets.shape}")
        
        losses = criterion(reshaped_output, flattened_targets)
    else:
        memory_logger.debug(f"Using custom criterion: {type(criterion).__name__}")
        losses = criterion(output, targets)
    
    # Log final loss values
    if isinstance(losses, torch.Tensor):
        memory_logger.debug(f"Raw loss tensor: shape={losses.shape}, mean={losses.mean().item()}")
    
    losses = losses.view(*output.shape[0:2])
    loss_mean, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
    memory_logger.debug(f"Final loss: {loss_mean.item()}, nan_share: {nan_share}")
    
    return loss_mean, nan_share


def train_epoch(
    model, 
    aggregate_k_gradients, 
    using_dist, 
    scaler, 
    dl, 
    device, 
    optimizer, 
    criterion, 
    n_out, 
    progress_bar=False
):
    model.train()  # Turn on the train mode
    total_loss = torch.tensor(0., device = device)
    nan_steps = torch.tensor(0., device = device)
    ignore_steps = torch.tensor(0., device = device)
    steps_per_epoch = len(dl)
    assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
    
    # Detect device type for backend-specific operations
    is_cuda = device.startswith('cuda')
    is_mps = device == 'mps'
    is_cpu = device == 'cpu'
    
    # Initialize progress bar with more informative metrics
    if progress_bar:
        dl = tqdm(dl, desc='Epoch Progress', leave=True)
        # Set initial progress bar formatting
        dl.set_description('Training')
    
    # For tracking GPU utilization
    gpu_util = 0.0
    batch_loss_history = []
    
    for batch, batch_data in enumerate(dl):
        # Debug log batch information
        memory_logger.debug(f"Processing batch {batch}/{steps_per_epoch}")
        memory_logger.debug(f"Batch data type: {type(batch_data)}, length: {len(batch_data)}")
        
        # Unpack batch data - could now include info dict for semantic features
        if len(batch_data) == 3:
            # Standard format: (data, targets, single_eval_pos)
            data, targets, single_eval_pos = batch_data
            batch_info = None
            memory_logger.debug("Standard batch format (no semantic info)")
        elif len(batch_data) == 4:
            # Extended format: (data, targets, single_eval_pos, info)
            data, targets, single_eval_pos, batch_info = batch_data
            memory_logger.debug(f"Extended batch format with semantic info: {list(batch_info.keys() if batch_info else [])}")
        else:
            raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
        
        # For semantic models, the data may be a tuple containing (info, x, y)
        # Let's check and extract the correct components
        if isinstance(data, tuple) and len(data) == 3:
            memory_logger.debug(f"Data is tuple with {len(data)} elements")
            
            # Check if the first element is a dictionary (info)
            if isinstance(data[0], dict):
                memory_logger.debug("Found nested batch_info in data tuple")
                # The nested info may have the class token patterns
                nested_info, x, y = data
                
                # Merge nested_info into batch_info (prioritize nested_info)
                if batch_info is None:
                    batch_info = nested_info
                elif nested_info is not None:
                    # Create a new dict to avoid modifying the original
                    batch_info = {**batch_info, **nested_info}
                    memory_logger.debug(f"Merged batch_info, now contains: {list(batch_info.keys())}")
                
                # Set data to just the tensor part
                data = (x, y)
            
            # Log each tensor component
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    memory_logger.debug(f"  data[{i}]: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(data, torch.Tensor):
            memory_logger.debug(f"Data shape: {data.shape}, dtype: {data.dtype}")
            
        memory_logger.debug(f"Raw targets shape: {targets.shape}, dtype: {targets.dtype}")
        memory_logger.debug(f"single_eval_pos: {single_eval_pos}")
        
        # Log semantic targets if present
        if batch_info is not None and 'semantic_targets' in batch_info:
            sem_targets = batch_info['semantic_targets']
            
        
        # Get GPU utilization if available
        if is_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming first GPU
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = info.gpu  # GPU utilization percentage
            except (ImportError, pynvml.NVMLError):
                gpu_util = -1  # Unable to get GPU utilization
                
        # Update the progress bar with detailed metrics
        if progress_bar:
            progress_desc = f'Batch {batch}/{steps_per_epoch}'
            if batch_loss_history:
                avg_loss = sum(batch_loss_history[-10:]) / min(len(batch_loss_history), 10)
                progress_desc += f' | Loss: {avg_loss:.4f}'
            
            progress_desc += f' | TrainSeq/TestSeq: {single_eval_pos}/{data[1].shape[0] - single_eval_pos}'
            
            if gpu_util > 0:
                progress_desc += f' | GPU: {gpu_util}%'
            
            dl.set_description(progress_desc)
            
        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log({
                'train_train_sample_number': single_eval_pos, 
                'train_test_sample_number': data[1].shape[0] - single_eval_pos,
                'batch': batch,
                'steps_per_epoch': steps_per_epoch,
                'gpu_utilization': gpu_util
            })

        if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
            cm = model.no_sync()
        else:
            cm = nullcontext()
            
        with cm:
            
            autocast_context = get_autocast_context(
                device=device,
                dtype=None,
                scaler=scaler,
            )
                
            with autocast_context:
                # Move data to the appropriate device
                # At this point, we should have a clean data tuple or tensor
                if isinstance(data, tuple):
                    device_data = tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                    memory_logger.debug(f"Moved tuple data to device: {device}")
                else:
                    device_data = data.to(device)
                    memory_logger.debug(f"Moved tensor data to device: {device}")
                
                # Forward pass
                memory_logger.debug(f"Running model forward pass with single_eval_pos={single_eval_pos}")
                
                # Check if we have semantic information to pass to the model
                class_texts = None
                
                if batch_info is not None and 'class_token_patterns' in batch_info:
                    # Extract class texts from token patterns
                    class_token_patterns = batch_info['class_token_patterns']
                    memory_logger.debug(f"Found class_token_patterns for {len(class_token_patterns)} classes")
                    
                    # Debug the token patterns
                    for class_idx in sorted(class_token_patterns.keys())[:2]:  # Look at first few
                        pattern = class_token_patterns[class_idx]
                        memory_logger.debug(f"Class {class_idx} pattern keys: {list(pattern.keys())}")
                        if 'tokens' in pattern:
                            memory_logger.debug(f"Class {class_idx} tokens: {pattern['tokens'].shape} - Type: {type(pattern['tokens'])}")
                    
                    # Generate text descriptions from the class token patterns
                    class_texts = []
                    
                    # Try to import CLIP tokenizer for token decoding
                    try:
                        from transformers import CLIPTokenizerFast
                        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
                        has_tokenizer = True
                        memory_logger.debug("Using CLIP tokenizer to decode tokens for class descriptions")
                    except (ImportError, Exception):
                        has_tokenizer = False
                        memory_logger.debug("CLIP tokenizer not available, using simplified class descriptions")
                    
                    # Generate meaningful descriptions for each class
                    for class_idx in sorted(class_token_patterns.keys()):
                        pattern = class_token_patterns[class_idx]
                        semantic_class = pattern.get('semantic_class', 0)
                        
                        # First try to use the actual column_name from semantic data if available
                        if 'column_name' in pattern and pattern['column_name'] is not None:
                            # Format it more nicely by removing underscores and adding spaces
                            col_name = pattern['column_name'].replace('_', ' ').title()
                            text = f"Data with {col_name}"
                            memory_logger.debug(f"Using real column name: {text}")
                        # Fall back to class_name if available
                        elif 'class_name' in pattern:
                            text = pattern['class_name']
                            memory_logger.debug(f"Using class_name from pattern: {text}")
                        # Otherwise, try to create a more descriptive text if we have token information and tokenizer
                        elif 'tokens' in pattern and has_tokenizer and len(pattern['tokens']) > 0:
                            try:
                                # Get the tokens and try to decode them
                                tokens = pattern['tokens'].cpu().tolist()
                                token_texts = tokenizer.decode(tokens[:5])  # Use first few tokens
                                # Clean up the token text
                                token_texts = token_texts.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
                                if token_texts:
                                    text = f"Data class {class_idx}: {token_texts}"
                                else:
                                    text = f"Data class {class_idx} from semantic class {semantic_class}"
                            except Exception as e:
                                memory_logger.debug(f"Error decoding tokens: {e}")
                                text = f"Data class {class_idx} from semantic class {semantic_class}"
                        else:
                            # Fallback to simple description
                            text = f"Data class {class_idx} from semantic class {semantic_class}"
                            
                        memory_logger.debug(f"Class {class_idx} text: '{text}'")
                            
                        class_texts.append(text)
                        
                    memory_logger.debug(f"Generated {len(class_texts)} class text descriptions: {class_texts[:3]}...")
                
                # Pass class_texts to the model's forward method if available
                output = model(device_data, single_eval_pos=single_eval_pos, class_texts=class_texts)
                
                # Log model output details
                if isinstance(output, dict):
                    memory_logger.debug(f"Model output is dictionary with keys: {list(output.keys())}")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            memory_logger.debug(f"Output['{k}']: shape={v.shape}, dtype={v.dtype}")
                else:
                    memory_logger.debug(f"Model output: shape={output.shape}, dtype={output.dtype}")

                # Process targets for evaluation 
                memory_logger.debug(f"Pre-filtered targets shape: {targets.shape}")
                if single_eval_pos is not None:
                    targets = targets[single_eval_pos:]
                    memory_logger.debug(f"Filtered targets with single_eval_pos={single_eval_pos}, new shape: {targets.shape}")
                    
                    # Also adjust semantic targets if present
                    if batch_info is not None and 'semantic_targets' in batch_info:
                        orig_shape = batch_info['semantic_targets'].shape
                        batch_info['semantic_targets'] = batch_info['semantic_targets'][single_eval_pos:]
                        memory_logger.debug(f"Filtered semantic targets from {orig_shape} to {batch_info['semantic_targets'].shape}")
                        
                        # Ensure semantic targets are long tensor type (for bincount and loss functions)
                        if batch_info['semantic_targets'].dtype != torch.long:
                            memory_logger.debug(f"Converting semantic targets from {batch_info['semantic_targets'].dtype} to torch.long")
                            batch_info['semantic_targets'] = batch_info['semantic_targets'].long()

                # Check for valid labels
                valid_labels = targets != -100
                valid_count = valid_labels.sum().item()
                memory_logger.debug(f"Valid labels: {valid_count}/{targets.numel()} ({valid_count/targets.numel()*100:.1f}%)")
                
                if valid_count == 0:
                    memory_logger.debug("Skipping batch due to no valid labels")
                    continue

                # Calculate loss
                memory_logger.debug(f"Calculating loss with criterion: {type(criterion).__name__}")
                memory_logger.debug(f"Criterion details: {criterion}")
                
                # Log any special criterion properties
                if hasattr(criterion, 'semantic_weight'):
                    memory_logger.debug(f"Semantic weight in loss: {criterion.semantic_weight}")
                
                loss, nan_share = eval_criterion(
                    criterion, 
                    targets, 
                    output, 
                    device=device, 
                    n_out=n_out,
                    batch_info=batch_info
                )
                
                # Scale loss for gradient accumulation
                original_loss = loss.item()
                loss = loss / aggregate_k_gradients
                memory_logger.debug(f"Original loss: {original_loss}, scaled for accumulation: {loss.item()}")

            # Get current batch loss value and log it
            current_batch_loss = loss.mean().cpu().detach().item() * aggregate_k_gradients
            batch_loss_history.append(current_batch_loss)
            
            # Log to wandb
            if wandb.run: 
                wandb.log({'batch_loss': current_batch_loss})
            
            # Update progress bar with current loss
            if progress_bar:
                avg_loss = sum(batch_loss_history[-10:]) / min(len(batch_loss_history), 10)
                dl.set_postfix(loss=f"{avg_loss:.4f}", refresh=True)
            
            # Backward pass with scaler if applicable
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # print("LOSS")
            # print(loss.mean().cpu().detach().item() * aggregate_k_gradients)

            # Update weights after accumulating gradients
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                # Gradient clipping with backend-specific handling
                # 'foreach=True' is not supported on MPS (Apple Silicon) in some PyTorch versions
                if device == 'mps':
                    try:
                        # Try first with foreach (newer PyTorch versions may support it)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                    except (RuntimeError, TypeError):
                        # Fallback to standard gradient clipping without foreach
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
                        if batch == 0:  # Only print warning once
                            print("Note: Using slower gradient clipping method for MPS device")
                else:
                    # For CUDA, CPU, ROCm, use the faster foreach version
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                
                # Use mixed precision optimizer step if enabled
                if scaler is not None and (is_cuda or is_mps):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad()                

            # Check for NaN loss
            if torch.isnan(loss):
                raise ValueError("NAN loss encountered")
            else:
                total_loss += loss.mean().cpu().detach().item()
                
            nan_steps += nan_share
            ignore_steps += (targets == -100).float().mean()
            
    return (total_loss / steps_per_epoch * aggregate_k_gradients,
            nan_steps.cpu().item() / steps_per_epoch,
            ignore_steps.cpu().item()/steps_per_epoch)


def train(dl, model, criterion, optimizer_state=None, scheduler=None,
          epochs=10, stop_after_epochs=None, learning_rate=None, min_lr=None, weight_decay=0.0, warmup_epochs=10,
          device='cuda:0',
          aggregate_k_gradients=1, verbose=True, epoch_callback=None, train_mixed_precision=False, adaptive_batch_size=False,
          learning_rate_schedule='cosine', lr_decay=0.99, adam_beta1=0.9, reduce_lr_on_spike=False,
          spike_tolerance=4, progress_bar=False,
          ):
    """
    Training function that supports various hardware backends (CUDA, MPS, ROCm, CPU).
    
    Args:
        dl: DataLoader for training
        model: Model to train
        criterion: Loss function
        optimizer_state: Optional state to restore optimizer from
        scheduler: Optional learning rate scheduler
        epochs: Number of epochs to train
        stop_after_epochs: Optional early stopping point
        learning_rate: Initial learning rate
        min_lr: Minimum learning rate for schedulers
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs for learning rate
        device: Device to train on ('cuda', 'cuda:N', 'mps', 'cpu')
        aggregate_k_gradients: Number of gradients to accumulate before update
        verbose: Whether to print progress
        epoch_callback: Callback function for each epoch
        train_mixed_precision: Whether to use mixed precision training
        adaptive_batch_size: Whether to adaptively increase batch size
        learning_rate_schedule: Type of learning rate schedule
        lr_decay: Learning rate decay factor for exponential schedule
        adam_beta1: Beta1 parameter for AdamW optimizer
        reduce_lr_on_spike: Whether to reduce learning rate on loss spikes
        spike_tolerance: Tolerance for loss spikes
        progress_bar: Whether to show progress bar during training
        
    Returns:
        total_loss: Final loss value
        model: Trained model (moved to CPU)
        dl: DataLoader used for training
        epoch: Final epoch number
    """
    # Initialize device and distributed setup if applicable
    using_dist, rank, device = init_dist(device)
    if rank == 0 and verbose:
        print(f'Using {device} device')

    # Detect device type for backend-specific operations
    is_cuda = device.startswith('cuda')
    is_mps = device == 'mps'
    is_cpu = device == 'cpu'
    is_rocm = is_cuda and torch.version.hip is not None

    # Move model and criterion to the target device
    model.to(device)
    criterion.to(device)

    # Store the number of output classes
    n_out = model.n_out
    
    # Set up distributed training if applicable
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        if rank == 0:
            print("Distributed training")
    elif is_cuda:
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "Unknown"
        backend_type = "ROCm (AMD)" if is_rocm else "CUDA (NVIDIA)"
        print(f"Single GPU training using {backend_type} on {gpu_name}")
    elif is_mps:
        print("Single GPU training using MPS (Apple Silicon)")
    elif is_cpu:
        print("Training on CPU")
    else:
        raise ValueError(f"Invalid device: {device}")

    # Initialize model metadata
    if rank == 0:
        model.learning_rates = getattr(model, 'learning_rates', [])
        model.losses = getattr(model, 'losses', [])
        model.wallclock_times = getattr(model, 'wallclock_times', [])
        model.start_time = time.time()
        if len(model.wallclock_times):
            model.start_time -= model.wallclock_times[-1]
        if epoch_callback is not None:
            epoch_callback(model, None, None, "start")

    # Set up model reference in dataloader
    dl.model = model
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(adam_beta1, 0.999))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    # Initialize schedulers
    spike_scheduler = None
    if scheduler is None:
        # Ensure the cosine annealing period is at least 1 epoch to avoid division by zero
        cosine_period = max(1, epochs - warmup_epochs)
        
        if learning_rate_schedule == 'cosine':
            base_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_period, eta_min=min_lr)
        elif learning_rate_schedule == 'exponential':
            base_scheduler = ExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)
        elif learning_rate_schedule == 'constant':
            base_scheduler = ExponentialLR(optimizer, gamma=1, min_lr=min_lr)
        else:
            raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
        
        # Add linear warmup to scheduler
        scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs),
                                             base_scheduler], milestones=[warmup_epochs])
        start_epoch = 1
    else:
        start_epoch = scheduler.last_epoch + 1

    # Set up optional spike-based learning rate reduction
    if reduce_lr_on_spike:
        spike_scheduler = ReduceLROnSpike(optimizer, smoothing=10, factor=0.5, min_lr=min_lr, tolerance=spike_tolerance, verbose=True)
    
    # Initialize mixed precision training if applicable
    # Different backend types have different mixed precision capabilities
    if train_mixed_precision:
        if is_cuda:
            # CUDA backend supports mixed precision
            scaler = GradScaler()
            if verbose:
                precision_type = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
                print(f"Using mixed precision training ({precision_type}) with CUDA backend")
        elif is_mps:
            # MPS supports float16 mixed precision
            scaler = GradScaler()
            if verbose:
                print("Using mixed precision training (float16) with MPS backend")
        else:
            # CPU doesn't benefit much from mixed precision, but we'll use it if requested
            scaler = None
            if verbose:
                print("Mixed precision requested but not used for CPU training")
    else:
        scaler = None
        if verbose:
            print("Using full precision training")

    # Check compatibility
    utils.check_compatibility(dl)

    # Initialize training variables
    total_loss = float('inf')
    increased_batch_size = 0
    epoch = start_epoch
    if stop_after_epochs is not None:
        epochs = min(epochs, stop_after_epochs)
    
    # Set up timing events for GPU backends
    if is_cuda:
        gpu_start_time = torch.cuda.Event(enable_timing=True)
        gpu_end_time = torch.cuda.Event(enable_timing=True)

    try:
        train_time, inference_time, train_gpu_time = [], [], []
        for epoch in range(start_epoch, epochs + 1):
            if verbose:
                # More informative epoch start message
                progress_banner = f"{'='*20} Epoch {epoch}/{epochs} {'='*20}"
                print(f"\n{progress_banner}")
                
                # Print current hyperparameters
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.8f}")
                
                # Print GPU memory usage if on CUDA
                if is_cuda:
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info()
                        free_mem_gb = free_mem / (1024**3)
                        total_mem_gb = total_mem / (1024**3)
                        used_mem_gb = total_mem_gb - free_mem_gb
                        print(f"GPU memory: {used_mem_gb:.2f}GB used / {total_mem_gb:.2f}GB total ({used_mem_gb/total_mem_gb*100:.1f}%)")
                    except:
                        # Older PyTorch versions or other issues
                        print(f"GPU: {torch.cuda.get_device_name(0)}")
                
                print(f"Steps per epoch: {len(dl)}")
                print("-" * len(progress_banner))

            # Record starting time
            epoch_start_time = time.time()
            if is_cuda:
                gpu_start_time.record()
            
            # Train for one epoch
            new_loss, nan_share, ignore_share = train_epoch(
                model, 
                aggregate_k_gradients, 
                using_dist, 
                scaler, 
                dl, 
                device, 
                optimizer, 
                criterion, 
                n_out,
                progress_bar=progress_bar,
            )

            # Update loss
            total_loss = new_loss
            
            # Get current learning rate
            if spike_scheduler is not None:
                last_lr = spike_scheduler.get_last_lr()[0]
            else:
                last_lr = scheduler.get_last_lr()[0]

            # Record training time
            train_time.append(time.time() - epoch_start_time)
            
            # Record GPU time if applicable
            if is_cuda:
                gpu_end_time.record()
                torch.cuda.synchronize()
                train_gpu_time.append(gpu_start_time.elapsed_time(gpu_end_time)/1000)
            elif is_mps:
                # MPS doesn't support Events API yet, so use wallclock time
                train_gpu_time.append(train_time[-1])
            else:
                train_gpu_time.append(0)

            # Print and log training progress
            if verbose:
                print('-' * 89)
                print(
                    f'| End of epoch {epoch:3d} | Wallclock time: {train_time[-1]:5.2f}s | ' + 
                    (f'GPU time: {train_gpu_time[-1]:5.2f}s | ' if (is_cuda or is_mps) else '') + 
                    f'Mean loss {total_loss:5.4f} |'
                )

                if wandb.run: 
                    wandb.log({
                        "avg_train_time": sum(train_time)/len(train_time), 
                        "train_time": train_time[-1],
                        "avg_train_gpu_time": sum(train_gpu_time)/len(train_gpu_time) if (is_cuda or is_mps) else 0, 
                        "train_gpu_time": train_gpu_time[-1] if (is_cuda or is_mps) else 0
                    })

                print(
                    f' lr {last_lr}'
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}')
                print('-' * 89)
                
            # Check for loss divergence
            if new_loss > 1.5 * total_loss:
                print("LOSS DIVERGED")
                return total_loss, model.to('cpu'), dl, epoch
            
            # Handle adaptive batch size if enabled
            if adaptive_batch_size:
                if increased_batch_size == 0 and epoch >= 20:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 1
                    print(f"Increased aggregate_k_gradients size to {aggregate_k_gradients}")
                elif increased_batch_size == 1 and epoch >= 50:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 2
                    print(f"Increased aggregate_k_gradients size to {aggregate_k_gradients}")
                elif increased_batch_size == 2 and epoch >= 200:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 3
                    print(f"Increased aggregate_k_gradients size to {aggregate_k_gradients}")
                elif increased_batch_size == 3 and total_loss >= 1000:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 4
                    print(f"Increased aggregate_k_gradients size to {aggregate_k_gradients}")
            
            # Update learning rate schedulers        
            scheduler.step()
            if spike_scheduler is not None:
                spike_scheduler.step(metrics=total_loss)
                
            # Handle epoch callback if provided
            if epoch_callback is not None and rank == 0:
                model.learning_rates.append(last_lr)
                model.losses.append(total_loss)
                model.wallclock_times.append(time.time() - model.start_time)
                output = epoch_callback(model, optimizer, scheduler, epoch)
                if output: 
                    inference_time.append(output)
                    if wandb.run:
                        wandb.log({
                            "avg_inference_time": sum(inference_time)/len(inference_time), 
                            "inference_time": inference_time[-1]
                        })

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Return trained model and stats
    if rank == 0:  # trivially true for non-parallel training
        return total_loss, model.to('cpu'), dl, epoch
