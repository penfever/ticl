import time, wandb
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import logging

import ticl.utils as utils
from ticl.utils import ExponentialLR, ReduceLROnSpike, init_dist, get_autocast_context, IGNORE_INDEX, memory_logger
from ticl.batch_monitoring import SemanticBatchMonitor

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
            
            # FAIL WITH INFORMATIVE ERROR if we're using SemanticConsistencyLoss but no semantic features
            if not has_semantic_features:
                error_msg = (
                    "ERROR: Missing semantic features in batch but using SemanticConsistencyLoss. "
                    "This typically happens when semantic features are not properly configured. "
                    "Possible causes: \n"
                    "1. Dataloader doesn't include semantic_targets in batch_info\n"
                    "2. Using a non-semantic prior with a semantic model\n"
                    "3. Mixed-precision setting is affecting feature generation\n"
                    "Try using --train-mixed-precision False or check dataloader configuration."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
                    
            # Create target dictionary
            target_dict = {
                'class_targets': targets.to(device).long(),
                'semantic_targets': semantic_targets
            }
            
            # Compute loss
            loss = criterion(output, target_dict)
            
            # Return loss as a tensor for compatibility
            return loss.unsqueeze(0).unsqueeze(0), 0.0
        else:
            # Fallback to just using class logits with standard loss
            output = output['class_logits']
    
    # Standard loss functions
    if isinstance(criterion, nn.GaussianNLLLoss):
        assert output.shape[-1] == 2, \
            'need to write a little bit of code to handle multiple regression targets at once'

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        losses = criterion(output.flatten(), targets.to(device).flatten())
    elif isinstance(criterion, nn.CrossEntropyLoss):
        # Ensure targets are valid and avoid out of bounds issues
        valid_targets = targets.to(device).clamp(min=0).long().flatten()
        max_target = valid_targets.max().item()
        
        # Check for valid target values and fix if needed
        if max_target >= n_out:
            valid_targets = valid_targets.clamp(max=n_out-1)
            max_target = n_out - 1
        
        # Debug reshape dimensions
        reshaped_output = output.reshape(-1, n_out)
        
        # Check for NaN or Inf values in output logits
        if torch.isnan(reshaped_output).any() or torch.isinf(reshaped_output).any():
            reshaped_output = torch.nan_to_num(reshaped_output, nan=0.0, posinf=1e4, neginf=-1e4)
        
        losses = criterion(reshaped_output, valid_targets)
    else:
        losses = criterion(output, targets)
    
    losses = losses.view(*output.shape[0:2])
    loss_mean, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
    
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
    progress_bar=False,
    batch_monitor=None,
    semantic_batch_monitoring=True,
    skip_bad_semantic_batches=False,
    semantic_batch_log_frequency=10
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
    
    # Initialize batch monitor if needed
    if batch_monitor is None and semantic_batch_monitoring:
        batch_monitor = SemanticBatchMonitor(
            enable_monitoring=semantic_batch_monitoring,
            skip_bad_batches=skip_bad_semantic_batches,
            log_frequency=semantic_batch_log_frequency
        )
    
    # Initialize progress bar with more informative metrics
    if progress_bar:
        dl = tqdm(dl, desc='Epoch Progress', leave=True)
        # Set initial progress bar formatting
        dl.set_description('Training')
    
    # For tracking GPU utilization
    gpu_util = 0.0
    batch_loss_history = []
    
    for batch, batch_data in enumerate(dl):
        # Unpack batch data - could now include info dict for semantic features
        if len(batch_data) == 3:
            # Standard format: (data, targets, single_eval_pos)
            data, targets, single_eval_pos = batch_data
            batch_info = None
        elif len(batch_data) == 4:
            # Extended format: (data, targets, single_eval_pos, info)
            data, targets, single_eval_pos, batch_info = batch_data
        else:
            raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
        
        # For semantic models, the data may be a tuple containing (info, x, y)
        # Let's check and extract the correct components
        if isinstance(data, tuple) and len(data) == 3:
            # Check if the first element is a dictionary (info)
            if isinstance(data[0], dict):
                # The nested info may have the class token patterns
                nested_info, x, y = data
                
                # Merge nested_info into batch_info (prioritize nested_info)
                if batch_info is None:
                    batch_info = nested_info
                elif nested_info is not None:
                    # Create a new dict to avoid modifying the original
                    batch_info = {**batch_info, **nested_info}
                
                # Set data to just the tensor part
                data = (x, y)
            
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
            
            # Get appropriate autocast context based on device type
            # The issue is that scaler can be active but autocast returns nullcontext
            autocast_context = get_autocast_context(
                device=device,
                dtype=None,
                scaler=scaler,
            )
                
            # Additional check to ensure we don't try to use scaler with nullcontext
            use_scaler = (scaler is not None and autocast_context is not nullcontext())
            
            # Move data to the appropriate device
            # At this point, we should have a clean data tuple or tensor
            if isinstance(data, tuple):
                device_data = tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
            else:
                device_data = data.to(device)
            
            # Let's skip batch if all labels are empty
            if isinstance(device_data, tuple) and len(device_data) == 2:
                if torch.all(device_data[1][single_eval_pos:] == -100):
                    continue
                    
            # Batch monitoring - analyze and potentially skip problematic batches
            if batch_monitor is not None:
                # Extract semantic tokens, targets and embeddings for analysis
                semantic_tokens = None
                semantic_targets = None
                semantic_embeddings = None
                
                # Try to get semantic tokens from batch info
                if batch_info is not None:
                    if 'semantic_targets' in batch_info:
                        semantic_targets = batch_info['semantic_targets']
                    
                    if 'semantic_tokens' in batch_info:
                        semantic_tokens = batch_info['semantic_tokens']
                    
                    # Extract semantic embeddings if already computed
                    if 'semantic_embeddings' in batch_info:
                        semantic_embeddings = batch_info['semantic_embeddings']
                
                # Analyze batch quality
                batch_stats, fingerprint, is_bad_batch = batch_monitor.analyze_batch(
                    batch_info, semantic_tokens, semantic_targets, semantic_embeddings
                )
                
                # Log to wandb periodically
                if batch_monitor.should_log_batch():
                    batch_monitor.log_batch_stats_to_wandb(batch_stats, fingerprint)
                
                # Skip problematic batches if enabled
                if batch_monitor.should_skip_batch(is_bad_batch):
                    memory_logger.warning(f"Skipping bad batch {batch}/{steps_per_epoch} - {fingerprint['hash']}")
                    # If using wandb, log the skip event
                    if wandb.run is not None:
                        wandb.log({
                            "semantic_batch/skipped": 1,
                            "semantic_batch/skipped_hash": fingerprint['hash'],
                            "semantic_batch/skipped_reason": "; ".join(batch_stats.get('bad_batch_reasons', ['Unknown']))
                        })
                    
                    # Log detailed info about the skipped batch
                    memory_logger.info(f"=== Skipped Batch Details ===")
                    memory_logger.info(f"Fingerprint: {fingerprint['hash']}")
                    memory_logger.info(f"Reasons: {batch_stats.get('bad_batch_reasons', ['Unknown'])}")
                    
                    # Log key statistics to help diagnose the issue
                    key_stats = {k: v for k, v in batch_stats.items() 
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    memory_logger.info(f"Key statistics: {key_stats}")
                    
                    continue

            # Check if we have semantic information to pass to the model
            class_texts = None
            
            if batch_info is not None and 'class_token_patterns' in batch_info:
                # Extract class texts from token patterns
                class_token_patterns = batch_info['class_token_patterns']
                
                # Try to import CLIP tokenizer for token decoding
                try:
                    from transformers import CLIPTokenizerFast
                    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
                    has_tokenizer = True
                except (ImportError, Exception):
                    has_tokenizer = False
                
                # Generate meaningful descriptions for each class
                class_texts = []
                for class_idx in sorted(class_token_patterns.keys()):
                    pattern = class_token_patterns[class_idx]
                    semantic_class = pattern.get('semantic_class', 0)
                    
                    # First try to use the actual column_name from semantic data if available
                    if 'column_name' in pattern and pattern['column_name'] is not None:
                        # Format it more nicely by removing underscores and adding spaces
                        col_name = pattern['column_name'].replace('_', ' ').title()
                        text = f"Data with {col_name}"
                    # Fall back to class_name if available
                    elif 'class_name' in pattern:
                        text = pattern['class_name']
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
                        except Exception:
                            text = f"Data class {class_idx} from semantic class {semantic_class}"
                    else:
                        # Fallback to simple description
                        text = f"Data class {class_idx} from semantic class {semantic_class}"
                        
                    class_texts.append(text)
            
            # Process targets for evaluation before forward pass
            if single_eval_pos is not None:
                targets = targets[single_eval_pos:]
                
                if batch_info is not None and 'semantic_targets' in batch_info and batch_info['semantic_targets'] is not None:
                    batch_info['semantic_targets'] = batch_info['semantic_targets'][single_eval_pos:]
                    
                    # Ensure semantic targets are long tensor type (for bincount and loss functions)
                    if batch_info['semantic_targets'].dtype != torch.long:
                        batch_info['semantic_targets'] = batch_info['semantic_targets'].long()
                    
                    # Print diagnostic info to verify we have valid targets
                    if batch % 50 == 0:  # Only print occasionally to avoid spam
                        unique_vals = torch.unique(batch_info['semantic_targets']).tolist()
                        print(f"Semantic targets unique values: {unique_vals}")
                        valid_semantic = (batch_info['semantic_targets'] != -100).sum().item()
                        print(f"Valid semantic targets: {valid_semantic}/{batch_info['semantic_targets'].numel()}")
            
            # breakpoint()
            # Check for valid labels
            valid_labels = targets != -100
            valid_count = valid_labels.sum().item()
            
            if valid_count == 0:
                continue
                
            with autocast_context:
                # Check if the model supports the semantic arguments (class_texts and batch_info)
                # The TabPFN model doesn't support these arguments, so we need to check dynamically
                # to prevent errors when passing unsupported arguments
                model_class_name = model.__class__.__name__
                if hasattr(model, 'module'):  # Handle DistributedDataParallel case
                    model_class_name = model.module.__class__.__name__
                
                # Only pass semantic arguments to models that support them
                supports_semantic = model_class_name not in ['TabPFN']
                
                if supports_semantic:
                    # Model supports semantic arguments
                    output = model(device_data, single_eval_pos=single_eval_pos,
                                   class_texts=class_texts, batch_info=batch_info)
                else:
                    # Basic model without semantic support - only pass required arguments
                    output = model(device_data, single_eval_pos=single_eval_pos)

                # Calculate loss
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
            # Only use scaler if it's not None AND we're using a compatible autocast context
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulating gradients
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                # Enhanced gradient clipping with more aggressive threshold and backend-specific handling
                max_norm = 0.2

                if is_cuda:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=True)
                else:
                    # This matches the more numerically stable approach used on MPS devices
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=False)
                
                # Use mixed precision optimizer step if enabled
                # Be more explicit about when to use scaler
                if use_scaler and (is_cuda or is_mps):
                    # Only use scaler if we used it in backward pass
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad()                

            # Check for NaN loss and handle it more gracefully
            if torch.isnan(loss):
                # Initialize with small loss value to avoid completely stopping training
                total_loss += 1.0
                nan_steps += 1.0  # Count the full step as NaN
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
          semantic_batch_monitoring=True, skip_bad_semantic_batches=False, semantic_batch_log_frequency=10,
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
        semantic_batch_monitoring: Whether to enable detailed monitoring of semantic batches
        skip_bad_semantic_batches: Whether to skip batches with problematic semantic data
        semantic_batch_log_frequency: How often to log batch statistics (every N batches)
        
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
    
    # Check if this is a semantic model with CLIP language transformer
    # Look for the CLIP text model in the model or its module
    has_clip_language_model = False
    clip_text_model = None
    model_module = model.module if hasattr(model, 'module') else model
    
    if hasattr(model_module, 'clip_text_model'):
        has_clip_language_model = True
        clip_text_model = model_module.clip_text_model
        if verbose:
            print("Detected CLIP language transformer in model")
    
    # Get config settings for the language transformer
    lang_transformer_lr = None
    lang_transformer_weight_decay = None
    lang_transformer_warmup_ratio = None
    lang_transformer_peak_ratio = None
    
    # Get the model type from the model class name
    model_cls_name = model_module.__class__.__name__.lower()
    
    # Check if we're using a semantic model with semantic features enabled
    semantic_feature_p = 0.0
    if 'tabflex' in model_cls_name or 'ssm' in model_cls_name:
        semantic_feature_p = getattr(model_module, 'semantic_feature_p', 0.0)
    elif 'tabpfn' in model_cls_name or 'transformer' in model_cls_name:
        semantic_feature_p = getattr(model_module, 'semantic_feature_p', 0.0)
    
    # Look for language transformer parameters in config
    if has_clip_language_model and semantic_feature_p > 0.0:
        # Look in prior.classification section for the parameters
        if hasattr(model_module, '_config') and 'prior' in model_module._config:
            config = model_module._config
            if 'classification' in config['prior']:
                classification_config = config['prior']['classification']
                lang_transformer_lr = classification_config.get('language_transformer_lr', 4e-6)
                lang_transformer_weight_decay = classification_config.get('language_transformer_weight_decay', 0.1)
                lang_transformer_warmup_ratio = classification_config.get('language_transformer_warmup_ratio', 0.1)
                lang_transformer_peak_ratio = classification_config.get('language_transformer_peak_ratio', 0.7)
        
        # If not found in model config, use default values
        if lang_transformer_lr is None:
            lang_transformer_lr = 4e-6
            lang_transformer_weight_decay = 0.1
            lang_transformer_warmup_ratio = 0.1
            lang_transformer_peak_ratio = 0.7
        
        if verbose:
            print(f"Using separate optimizer for language transformer with lr={lang_transformer_lr},"
                  f" weight_decay={lang_transformer_weight_decay}")
            print(f"Language transformer LR schedule: warmup_ratio={lang_transformer_warmup_ratio},"
                  f" peak_ratio={lang_transformer_peak_ratio}")
    
    # Create separate optimizers for model and language transformer if needed
    if has_clip_language_model and semantic_feature_p > 0.0:
        # Separate the parameters into two groups
        language_params = []
        model_params = []
        
        # Get all language transformer parameters
        for name, param in clip_text_model.named_parameters():
            if param.requires_grad:
                language_params.append(param)
        
        # Get all other model parameters
        for name, param in model.named_parameters():
            if param.requires_grad and not any(p is param for p in language_params):
                model_params.append(param)
        
        # Create separate optimizers
        model_optimizer = torch.optim.AdamW(
            model_params, 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=(adam_beta1, 0.999)
        )
        
        language_optimizer = torch.optim.AdamW(
            language_params, 
            lr=lang_transformer_lr, 
            weight_decay=lang_transformer_weight_decay, 
            betas=(0.9, 0.999)  # Standard betas for language models
        )
        
        # Create a combined optimizer that updates both parameter groups
        # For compatibility with existing code
        from torch.optim import Optimizer
        
        class CombinedOptimizer(Optimizer):
            def __init__(self, model_optimizer, language_optimizer):
                self.model_optimizer = model_optimizer
                self.language_optimizer = language_optimizer
                # For compatibility with torch.optim.Optimizer
                self.param_groups = model_optimizer.param_groups + language_optimizer.param_groups
                self.state = {}  # Not used directly
                
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                self.model_optimizer.step()
                self.language_optimizer.step()
                return loss
                
            def zero_grad(self):
                self.model_optimizer.zero_grad()
                self.language_optimizer.zero_grad()
                
            def state_dict(self):
                return {
                    'model_optimizer': self.model_optimizer.state_dict(),
                    'language_optimizer': self.language_optimizer.state_dict()
                }
                
            def load_state_dict(self, state_dict):
                self.model_optimizer.load_state_dict(state_dict['model_optimizer'])
                self.language_optimizer.load_state_dict(state_dict['language_optimizer'])
        
        # Create the combined optimizer
        optimizer = CombinedOptimizer(model_optimizer, language_optimizer)
        
        # Store both optimizers in the model for reference
        model.model_optimizer = model_optimizer
        model.language_optimizer = language_optimizer
        
        if optimizer_state is not None:
            if isinstance(optimizer_state, dict) and 'model_optimizer' in optimizer_state:
                # New format state dict
                optimizer.load_state_dict(optimizer_state)
            else:
                # Old format - just load into model optimizer
                model_optimizer.load_state_dict(optimizer_state)
        
        # Initialize schedulers
        spike_scheduler = None
        language_scheduler = None
        
        if scheduler is None:
            # Ensure the cosine annealing period is at least 1 epoch to avoid division by zero
            cosine_period = max(1, epochs - warmup_epochs)
            
            # Create scheduler for model parameters
            if learning_rate_schedule == 'cosine':
                base_scheduler = CosineAnnealingLR(model_optimizer, T_max=cosine_period, eta_min=min_lr)
            elif learning_rate_schedule == 'exponential':
                base_scheduler = ExponentialLR(model_optimizer, gamma=lr_decay, min_lr=min_lr)
            elif learning_rate_schedule == 'constant':
                base_scheduler = ExponentialLR(model_optimizer, gamma=1, min_lr=min_lr)
            else:
                raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
            
            # Add linear warmup to scheduler
            model_scheduler = SequentialLR(
                model_optimizer, 
                [LinearLR(model_optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs),
                base_scheduler], 
                milestones=[warmup_epochs]
            )
            
            # Create custom scheduler for language transformer parameters
            from ticl.utils import LanguageTransformerScheduler
            language_scheduler = LanguageTransformerScheduler(
                language_optimizer,
                warmup_ratio=lang_transformer_warmup_ratio,
                peak_ratio=lang_transformer_peak_ratio,
                max_epochs=epochs,
                min_lr=1e-8,
                verbose=True
            )
            
            # Create scheduler wrapper that updates both schedulers
            class CombinedScheduler:
                def __init__(self, model_scheduler, language_scheduler):
                    self.model_scheduler = model_scheduler
                    self.language_scheduler = language_scheduler
                    self.last_epoch = model_scheduler.last_epoch
                    
                def step(self):
                    self.model_scheduler.step()
                    self.language_scheduler.step()
                    self.last_epoch = self.model_scheduler.last_epoch
                    
                def state_dict(self):
                    return {
                        'model_scheduler': self.model_scheduler.state_dict(),
                        'language_scheduler': self.language_scheduler.state_dict()
                    }
                    
                def load_state_dict(self, state_dict):
                    self.model_scheduler.load_state_dict(state_dict['model_scheduler'])
                    self.language_scheduler.load_state_dict(state_dict['language_scheduler'])
                    self.last_epoch = self.model_scheduler.last_epoch
                    
                def get_last_lr(self):
                    return self.model_scheduler.get_last_lr() + self.language_scheduler.get_last_lr()
            
            scheduler = CombinedScheduler(model_scheduler, language_scheduler)
            start_epoch = 1
        else:
            # If loading from existing scheduler state
            if hasattr(scheduler, 'model_scheduler') and hasattr(scheduler, 'language_scheduler'):
                # Already a combined scheduler
                start_epoch = scheduler.last_epoch + 1
            else:
                # Convert to combined scheduler
                from ticl.utils import LanguageTransformerScheduler
                language_scheduler = LanguageTransformerScheduler(
                    language_optimizer,
                    warmup_ratio=lang_transformer_warmup_ratio,
                    peak_ratio=lang_transformer_peak_ratio,
                    max_epochs=epochs,
                    min_lr=1e-8,
                    last_epoch=scheduler.last_epoch,
                    verbose=True
                )
                
                # Create combined scheduler
                class CombinedScheduler:
                    def __init__(self, model_scheduler, language_scheduler):
                        self.model_scheduler = model_scheduler
                        self.language_scheduler = language_scheduler
                        self.last_epoch = model_scheduler.last_epoch
                        
                    def step(self):
                        self.model_scheduler.step()
                        self.language_scheduler.step()
                        self.last_epoch = self.model_scheduler.last_epoch
                        
                    def state_dict(self):
                        return {
                            'model_scheduler': self.model_scheduler.state_dict(),
                            'language_scheduler': self.language_scheduler.state_dict()
                        }
                        
                    def load_state_dict(self, state_dict):
                        self.model_scheduler.load_state_dict(state_dict['model_scheduler'])
                        self.language_scheduler.load_state_dict(state_dict['language_scheduler'])
                        self.last_epoch = self.model_scheduler.last_epoch
                        
                    def get_last_lr(self):
                        return self.model_scheduler.get_last_lr() + self.language_scheduler.get_last_lr()
                
                # Create the combined scheduler
                scheduler = CombinedScheduler(scheduler, language_scheduler)
                start_epoch = scheduler.last_epoch + 1
        
        # Set up optional spike-based learning rate reduction for main model
        if reduce_lr_on_spike:
            spike_scheduler = ReduceLROnSpike(model_optimizer, smoothing=10, factor=0.5, min_lr=min_lr, tolerance=spike_tolerance, verbose=True)
    
    else:
        # Standard single optimizer approach for all other models
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
        if is_cuda or is_mps:
            scaler = GradScaler()
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
            
            # Initialize a shared batch monitor for all epochs
            batch_monitor = SemanticBatchMonitor(
                enable_monitoring=semantic_batch_monitoring,
                skip_bad_batches=skip_bad_semantic_batches,
                log_frequency=semantic_batch_log_frequency
            )
            
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
                batch_monitor=batch_monitor,
                semantic_batch_monitoring=semantic_batch_monitoring,
                skip_bad_semantic_batches=skip_bad_semantic_batches,
                semantic_batch_log_frequency=semantic_batch_log_frequency
            )
            
            # Log batch monitoring summary to wandb
            if semantic_batch_monitoring and wandb.run:
                summary = batch_monitor.get_batch_history_summary()
                wandb.log({
                    "semantic_batch/summary/total_batches": summary["total_batches"],
                    "semantic_batch/summary/bad_batches": summary["bad_batches"],
                    "semantic_batch/summary/bad_ratio": summary["bad_ratio"]
                })

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
                
                # Store the current epoch count in the model for use by components like SemanticAwareClassifier
                # This enables epoch-dependent behaviors like warmup schedules
                if hasattr(model, 'module'):  # Handle DistributedDataParallel wrapping
                    if hasattr(model.module, '_epoch_count'):
                        model.module._epoch_count = epoch
                else:
                    if hasattr(model, '_epoch_count'):
                        model._epoch_count = epoch
                    else:
                        # Add the attribute if it doesn't exist
                        setattr(model, '_epoch_count', epoch)
                
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