"""
Batch monitoring utilities for detecting and reporting problematic data during training.
This is especially helpful for semantic feature training where certain batches can cause instability.
"""

import torch
import numpy as np
import logging
import json
import uuid
import hashlib
from collections import defaultdict, deque

# Use the same logger as the rest of the codebase
memory_logger = logging.getLogger("memory_profiling")

class SemanticBatchMonitor:
    """
    Utility for monitoring semantic batch quality during training.
    Logs detailed batch statistics to wandb and helps identify problematic batches.
    """
    
    def __init__(self, 
                 enable_monitoring=True, 
                 skip_bad_batches=False, 
                 log_frequency=10,
                 max_history=100):
        """
        Initialize the batch monitor.
        
        Parameters:
        -----------
        enable_monitoring : bool
            Whether to enable detailed batch monitoring
        skip_bad_batches : bool
            Whether to skip batches deemed problematic
        log_frequency : int
            How often to log batch statistics (every N batches)
        max_history : int
            Maximum number of batch fingerprints to keep in history
        """
        self.enable_monitoring = enable_monitoring
        self.skip_bad_batches = skip_bad_batches
        self.log_frequency = log_frequency
        
        # Storage for batch fingerprints and stats
        self.batch_history = deque(maxlen=max_history)
        self.bad_batch_history = deque(maxlen=max_history)
        self.batch_stats = defaultdict(list)
        
        # Track batch counts
        self.batch_count = 0
        self.bad_batch_count = 0
        
        # Define thresholds for problematic batches
        self.thresholds = {
            'min_valid_semantic_ratio': 0.1,      # Minimum ratio of valid semantic targets
            'max_semantic_class': 10,             # Maximum expected semantic class ID
            'max_nan_ratio': 0.5,                 # Maximum ratio of NaN values allowed
            'min_class_token_patterns': 1,        # Minimum number of class token patterns
            'min_valid_targets': 5,               # Minimum number of valid targets
            'max_token_id': 49500,                # Maximum valid CLIP token ID (vocab size ~49408)
            'max_embedding_norm': 100.0,          # Maximum L2 norm for embeddings
            'min_embedding_norm': 0.01,           # Minimum L2 norm for embeddings
            'max_embedding_std': 10.0,            # Maximum standard deviation within embeddings
            'max_valid_targets': 50000,           # Maximum reasonable number of valid targets
            'token_target_ratio_tolerance': 0.2,  # Maximum allowed difference between token and target valid ratios
        }
        
        memory_logger.info(f"Semantic batch monitoring initialized: monitoring={enable_monitoring}, skip_bad={skip_bad_batches}")
    
    def generate_batch_fingerprint(self, batch_info, semantic_tokens=None, semantic_targets=None):
        """
        Generate a unique fingerprint for a batch to trace back to problematic data.
        
        Parameters:
        -----------
        batch_info : dict
            Dictionary with batch metadata
        semantic_tokens : torch.Tensor, optional
            Tensor of semantic tokens
        semantic_targets : torch.Tensor, optional
            Tensor of semantic class targets
            
        Returns:
        --------
        str
            Unique batch fingerprint
        """
        fingerprint_data = {"uuid": str(uuid.uuid4())}
        
        # Extract key data from batch_info
        if batch_info is not None:
            if 'class_token_patterns' in batch_info:
                pattern_keys = sorted(list(batch_info['class_token_patterns'].keys()))
                fingerprint_data['pattern_keys'] = pattern_keys
                
                # Include column names if available (these help identify the data)
                column_names = []
                for key in pattern_keys:
                    pattern = batch_info['class_token_patterns'][key]
                    if 'column_name' in pattern and pattern['column_name'] is not None:
                        column_names.append(pattern['column_name'])
                
                if column_names:
                    fingerprint_data['column_names'] = column_names
        
        # Create a deterministic hash from semantic data
        if semantic_tokens is not None and hasattr(semantic_tokens, 'shape'):
            fingerprint_data['semantic_tokens_shape'] = list(semantic_tokens.shape)
            
            # Sample some token values for fingerprinting
            if hasattr(semantic_tokens, 'cpu') and hasattr(semantic_tokens, 'numpy'):
                try:
                    # Take a small sample of tokens to include in fingerprint
                    tokens_sample = semantic_tokens.cpu().numpy().flatten()[:20].tolist()
                    fingerprint_data['tokens_sample'] = tokens_sample
                except Exception as e:
                    memory_logger.warning(f"Failed to sample tokens for fingerprint: {str(e)}")
        
        if semantic_targets is not None and hasattr(semantic_targets, 'shape'):
            fingerprint_data['semantic_targets_shape'] = list(semantic_targets.shape)
            
            # Count unique target values for fingerprinting
            if hasattr(semantic_targets, 'cpu') and hasattr(semantic_targets, 'numpy'):
                try:
                    targets_np = semantic_targets.cpu().numpy().flatten()
                    unique_targets, counts = np.unique(targets_np, return_counts=True)
                    target_counts = {str(int(t)): int(c) for t, c in zip(unique_targets, counts)}
                    fingerprint_data['target_counts'] = target_counts
                except Exception as e:
                    memory_logger.warning(f"Failed to count targets for fingerprint: {str(e)}")
        
        # Generate a JSON string and hash it for the final fingerprint
        try:
            fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
            fingerprint_hash = hashlib.md5(fingerprint_json.encode()).hexdigest()
            
            # Store full data with the hash for reference
            fingerprint = {
                'hash': fingerprint_hash,
                'data': fingerprint_data
            }
            
            return fingerprint
            
        except Exception as e:
            memory_logger.error(f"Failed to generate batch fingerprint: {str(e)}")
            # Fallback to a random UUID if JSON serialization fails
            return {'hash': str(uuid.uuid4()), 'data': {'error': str(e)}}
    
    def analyze_batch(self, batch_info, semantic_tokens=None, semantic_targets=None, embeddings=None):
        """
        Analyze a batch for potential issues and generate statistics.
        
        Parameters:
        -----------
        batch_info : dict
            Dictionary with batch metadata
        semantic_tokens : torch.Tensor, optional
            Tensor of semantic tokens
        semantic_targets : torch.Tensor, optional
            Tensor of semantic class targets
        embeddings : torch.Tensor, optional
            CLIP text embeddings if available
            
        Returns:
        --------
        tuple
            (batch_stats, fingerprint, is_bad_batch)
        """
        if not self.enable_monitoring:
            return None, None, False
        
        # Generate unique fingerprint for this batch
        fingerprint = self.generate_batch_fingerprint(batch_info, semantic_tokens, semantic_targets)
        
        # Analyze batch components
        stats = {}
        is_bad_batch = False
        reasons = []
        
        # Check if batch_info indicates we lost ignore indices during slicing
        if batch_info is not None and batch_info.get('lost_ignore_indices', False):
            is_bad_batch = True
            reasons.append("Lost all ignore indices (-100) during data slicing")
            stats['lost_ignore_indices'] = True
        
        # Track count
        self.batch_count += 1
        stats['batch_index'] = self.batch_count
        
        # === Check semantic tokens ===
        if semantic_tokens is not None:
            try:
                if hasattr(semantic_tokens, 'shape'):
                    stats['semantic_tokens_shape'] = list(semantic_tokens.shape)
                    
                if hasattr(semantic_tokens, 'dtype'):
                    stats['semantic_tokens_dtype'] = str(semantic_tokens.dtype)
                    
                if hasattr(semantic_tokens, 'device'):
                    stats['semantic_tokens_device'] = str(semantic_tokens.device)
                
                # Check for NaN or extreme values
                if hasattr(semantic_tokens, 'isnan'):
                    nan_count = semantic_tokens.isnan().sum().item()
                    total_count = semantic_tokens.numel()
                    nan_ratio = nan_count / total_count if total_count > 0 else 0
                    
                    stats['semantic_tokens_nan_count'] = nan_count
                    stats['semantic_tokens_nan_ratio'] = nan_ratio
                    
                    if nan_ratio > self.thresholds['max_nan_ratio']:
                        is_bad_batch = True
                        reasons.append(f"High NaN ratio in tokens: {nan_ratio:.4f}")
                
                # Token distribution statistics
                if hasattr(semantic_tokens, 'min') and hasattr(semantic_tokens, 'max'):
                    stats['semantic_tokens_min'] = semantic_tokens.min().item()
                    stats['semantic_tokens_max'] = semantic_tokens.max().item()
                    
                    # Check for token IDs outside the CLIP vocabulary
                    if stats['semantic_tokens_max'] > self.thresholds['max_token_id']:
                        is_bad_batch = True
                        reasons.append(f"Token ID exceeds vocabulary size: {stats['semantic_tokens_max']}")
                        
                # Analyze token value distribution
                if hasattr(semantic_tokens, 'float') and hasattr(semantic_tokens, 'std'):
                    try:
                        # Get distribution statistics for non-padding tokens
                        # CLIP typically uses 0 or 49407 as padding token IDs
                        valid_tokens = semantic_tokens[semantic_tokens > 0]
                        valid_tokens = valid_tokens[valid_tokens < 49407]
                        
                        if valid_tokens.numel() > 0:
                            # Get standard deviation and entropy of token distribution
                            tokens_std = valid_tokens.float().std().item()
                            stats['semantic_tokens_std'] = tokens_std
                            
                            # Compute histogram for token distribution analysis
                            if hasattr(valid_tokens, 'histc'):
                                bins = 10
                                hist = valid_tokens.float().histc(bins=bins, min=0, max=49407)
                                # Convert to probabilities
                                hist = hist / hist.sum()
                                # Compute entropy of distribution
                                entropy = -(hist * torch.log(hist + 1e-10)).sum().item()
                                stats['semantic_tokens_entropy'] = entropy
                    except Exception as e:
                        memory_logger.warning(f"Failed to analyze token distribution: {str(e)}")
                
            except Exception as e:
                memory_logger.warning(f"Failed to analyze semantic tokens: {str(e)}")
                stats['semantic_tokens_error'] = str(e)
        
        # === Check semantic targets ===
        if semantic_targets is not None:
            try:
                if hasattr(semantic_targets, 'shape'):
                    stats['semantic_targets_shape'] = list(semantic_targets.shape)
                
                if hasattr(semantic_targets, 'dtype'):
                    stats['semantic_targets_dtype'] = str(semantic_targets.dtype)
                
                # Check valid vs. invalid targets (-100)
                if hasattr(semantic_targets, 'numel'):
                    total_targets = semantic_targets.numel()
                    stats['semantic_targets_total'] = total_targets
                    
                    valid_mask = semantic_targets != -100
                    valid_count = valid_mask.sum().item()
                    valid_ratio = valid_count / total_targets if total_targets > 0 else 0
                    
                    stats['semantic_targets_valid_count'] = valid_count
                    stats['semantic_targets_valid_ratio'] = valid_ratio
                    
                    # Check for potentially invalid target distributions
                    if valid_ratio == 1.0 and total_targets > 100:
                        # This is suspicious - we would expect some -100 values
                        memory_logger.warning(f"Suspicious semantic targets with 100% valid ratio and {total_targets} total values")
                        # Check if we also have semantic tokens to compare
                        if semantic_tokens is not None and hasattr(semantic_tokens, 'numel'):
                            token_count = semantic_tokens.numel()
                            token_valid_mask = semantic_tokens != -100
                            token_valid_ratio = token_valid_mask.sum().item() / token_count if token_count > 0 else 0
                            
                            # If tokens have -100 values but targets don't, this is inconsistent
                            if token_valid_ratio < 1.0 and valid_ratio == 1.0:
                                is_bad_batch = True
                                reasons.append(f"Inconsistent token/target masking: token valid ratio {token_valid_ratio:.4f} but target valid ratio {valid_ratio:.4f}")
                                stats['semantic_token_target_inconsistent'] = True
                    
                    # Check if we have enough valid targets
                    if valid_count < self.thresholds['min_valid_targets']:
                        is_bad_batch = True
                        reasons.append(f"Too few valid semantic targets: {valid_count}")
                    elif valid_ratio < self.thresholds['min_valid_semantic_ratio']:
                        is_bad_batch = True
                        reasons.append(f"Valid semantic target ratio too low: {valid_ratio:.4f}")
                    elif valid_ratio == 1.0 and valid_count > 10000:
                        # High number of targets with no -100 values is suspicious
                        is_bad_batch = True
                        reasons.append(f"Suspiciously high number of valid targets with no ignore values: {valid_count}")
                
                # Check target distribution
                if hasattr(semantic_targets, 'min') and hasattr(semantic_targets, 'max'):
                    # Compute min/max of valid targets only (ignoring -100)
                    valid_targets = semantic_targets[valid_mask] if 'valid_mask' in locals() else semantic_targets
                    
                    if valid_targets.numel() > 0:
                        stats['semantic_targets_min'] = valid_targets.min().item()
                        stats['semantic_targets_max'] = valid_targets.max().item()
                        
                        # Check if max target is too large
                        if stats['semantic_targets_max'] > self.thresholds['max_semantic_class']:
                            is_bad_batch = True
                            reasons.append(f"Semantic class ID too large: {stats['semantic_targets_max']}")
                    
                    # Count unique targets for analysis
                    if hasattr(valid_targets, 'cpu') and hasattr(valid_targets, 'numpy'):
                        targets_np = valid_targets.cpu().numpy().flatten()
                        unique_targets, counts = np.unique(targets_np, return_counts=True)
                        
                        target_counts = {str(int(t)): int(c) for t, c in zip(unique_targets, counts)}
                        stats['semantic_target_classes'] = list(sorted([int(t) for t in target_counts.keys()]))
                        stats['semantic_target_counts'] = target_counts
                        
                        # Compute target distribution entropy
                        probs = counts / np.sum(counts)
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        stats['semantic_targets_entropy'] = float(entropy)
                
            except Exception as e:
                memory_logger.warning(f"Failed to analyze semantic targets: {str(e)}")
                stats['semantic_targets_error'] = str(e)
        
        # === Check class token patterns ===
        if batch_info is not None and 'class_token_patterns' in batch_info:
            try:
                patterns = batch_info['class_token_patterns']
                pattern_count = len(patterns)
                stats['class_token_patterns_count'] = pattern_count
                
                # Check for insufficient patterns
                if pattern_count < self.thresholds['min_class_token_patterns']:
                    is_bad_batch = True
                    reasons.append(f"Too few class token patterns: {pattern_count}")
                
                # Extract pattern metadata
                pattern_keys = sorted(list(patterns.keys()))
                column_names = []
                semantic_classes = []
                
                for key in pattern_keys:
                    pattern = patterns[key]
                    if 'column_name' in pattern and pattern['column_name'] is not None:
                        column_names.append(pattern['column_name'])
                    
                    if 'semantic_class' in pattern:
                        semantic_classes.append(pattern['semantic_class'])
                
                stats['class_token_patterns_keys'] = pattern_keys
                
                if column_names:
                    stats['class_token_patterns_columns'] = column_names
                
                if semantic_classes:
                    stats['class_token_patterns_semantic_classes'] = list(sorted(set(semantic_classes)))
                
            except Exception as e:
                memory_logger.warning(f"Failed to analyze class token patterns: {str(e)}")
                stats['class_token_patterns_error'] = str(e)
                
        # === Check CLIP embeddings if available ===
        if embeddings is not None:
            try:
                if hasattr(embeddings, 'shape'):
                    stats['embeddings_shape'] = list(embeddings.shape)
                    
                if hasattr(embeddings, 'dtype'):
                    stats['embeddings_dtype'] = str(embeddings.dtype)
                    
                # Check for NaN or Inf values in embeddings
                if hasattr(embeddings, 'isnan') and hasattr(embeddings, 'isinf'):
                    nan_count = embeddings.isnan().sum().item()
                    inf_count = embeddings.isinf().sum().item()
                    total_count = embeddings.numel()
                    
                    nan_ratio = nan_count / total_count if total_count > 0 else 0
                    inf_ratio = inf_count / total_count if total_count > 0 else 0
                    
                    stats['embeddings_nan_count'] = nan_count
                    stats['embeddings_inf_count'] = inf_count
                    stats['embeddings_nan_ratio'] = nan_ratio
                    stats['embeddings_inf_ratio'] = inf_ratio
                    
                    if nan_ratio > 0 or inf_ratio > 0:
                        is_bad_batch = True
                        reasons.append(f"NaN/Inf in embeddings: {nan_count + inf_count} values")
                
                # Compute embedding norms
                if hasattr(embeddings, 'norm'):
                    try:
                        # Compute L2 norm of each embedding
                        if len(embeddings.shape) >= 2:
                            # For batched embeddings [batch_size, emb_dim]
                            emb_norms = embeddings.norm(dim=1)
                            
                            stats['embeddings_norm_min'] = emb_norms.min().item()
                            stats['embeddings_norm_max'] = emb_norms.max().item()
                            stats['embeddings_norm_mean'] = emb_norms.mean().item()
                            stats['embeddings_norm_std'] = emb_norms.std().item()
                            
                            # Check for abnormal embedding norms
                            if stats['embeddings_norm_max'] > self.thresholds['max_embedding_norm']:
                                is_bad_batch = True
                                reasons.append(f"Embedding norm too large: {stats['embeddings_norm_max']:.4f}")
                                
                            if stats['embeddings_norm_min'] < self.thresholds['min_embedding_norm']:
                                is_bad_batch = True
                                reasons.append(f"Embedding norm too small: {stats['embeddings_norm_min']:.4f}")
                        
                        # Compute the standard deviation within each embedding
                        if len(embeddings.shape) >= 2:
                            # Standard deviation across embedding dimensions
                            emb_std = embeddings.std(dim=1)
                            
                            stats['embeddings_internal_std_min'] = emb_std.min().item()
                            stats['embeddings_internal_std_max'] = emb_std.max().item()
                            stats['embeddings_internal_std_mean'] = emb_std.mean().item()
                            
                            if stats['embeddings_internal_std_max'] > self.thresholds['max_embedding_std']:
                                is_bad_batch = True
                                reasons.append(f"High embedding internal variance: {stats['embeddings_internal_std_max']:.4f}")
                            
                        # Check for unusually sparse or dense embeddings 
                        if len(embeddings.shape) >= 2:
                            # Count non-zero elements
                            active_ratio = (embeddings != 0).float().mean(dim=1)
                            stats['embeddings_active_ratio_min'] = active_ratio.min().item()
                            stats['embeddings_active_ratio_max'] = active_ratio.max().item()
                            stats['embeddings_active_ratio_mean'] = active_ratio.mean().item()
                            
                            # Very sparse or very dense embeddings can indicate issues
                            if stats['embeddings_active_ratio_min'] < 0.01:
                                memory_logger.warning(f"Very sparse embedding detected: {stats['embeddings_active_ratio_min']:.4f} active ratio")
                    except Exception as e:
                        memory_logger.warning(f"Failed to compute embedding norms: {str(e)}")
                        
                # Compute cosine similarity between embeddings in batch
                if len(embeddings.shape) >= 2 and embeddings.shape[0] > 1:
                    try:
                        import torch.nn.functional as F
                        
                        # Normalize embeddings
                        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                        
                        # Compute pairwise similarities
                        similarities = torch.matmul(embeddings_norm, embeddings_norm.t())
                        
                        # Remove self-similarities (diagonal)
                        mask = torch.ones_like(similarities) - torch.eye(similarities.shape[0], device=similarities.device)
                        masked_similarities = similarities * mask
                        
                        # Get statistics
                        stats['embeddings_similarity_min'] = masked_similarities.min().item()
                        stats['embeddings_similarity_max'] = masked_similarities.max().item()
                        stats['embeddings_similarity_mean'] = masked_similarities.sum().item() / (mask.sum().item() + 1e-8)
                        
                        # Check for unusually high similarity between different embeddings
                        if stats['embeddings_similarity_max'] > 0.99:
                            is_bad_batch = True
                            reasons.append(f"Near-duplicate embeddings detected: {stats['embeddings_similarity_max']:.4f} similarity")
                    except Exception as e:
                        memory_logger.warning(f"Failed to compute embedding similarities: {str(e)}")
                        
            except Exception as e:
                memory_logger.warning(f"Failed to analyze embeddings: {str(e)}")
                stats['embeddings_error'] = str(e)
        
        # Finalize bad batch detection
        if is_bad_batch:
            stats['is_bad_batch'] = True
            stats['bad_batch_reasons'] = reasons
            self.bad_batch_count += 1
            self.bad_batch_history.append({
                'fingerprint': fingerprint,
                'stats': stats,
                'count': self.bad_batch_count
            })
            
            memory_logger.warning(f"Bad batch detected ({self.bad_batch_count}): {', '.join(reasons)}")
        
        # Save batch fingerprint to history
        self.batch_history.append({
            'fingerprint': fingerprint,
            'stats': stats,
            'is_bad': is_bad_batch
        })
        
        # Update running statistics
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.batch_stats[key].append(value)
        
        return stats, fingerprint, is_bad_batch
    
    def should_log_batch(self):
        """Check if we should log this batch based on frequency"""
        return self.enable_monitoring and (self.batch_count % self.log_frequency == 0)
    
    def should_skip_batch(self, is_bad_batch):
        """Determine if a batch should be skipped during training"""
        return self.skip_bad_batches and is_bad_batch
    
    def log_batch_stats_to_wandb(self, batch_stats, fingerprint, prefix="semantic_batch"):
        """Log batch statistics to wandb with proper prefixing"""
        if not self.enable_monitoring or batch_stats is None:
            return
            
        try:
            import wandb
            
            # Only log if wandb is running
            if not wandb.run:
                return
                
            # Create a flattened dict for wandb
            log_dict = {}
            
            # Add scalar values to wandb
            for key, value in batch_stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    log_dict[f"{prefix}/{key}"] = value
            
            # Add batch fingerprint information
            if fingerprint and 'hash' in fingerprint:
                log_dict[f"{prefix}/fingerprint"] = fingerprint['hash']
            
            # Add bad batch flag if present
            if 'is_bad_batch' in batch_stats:
                log_dict[f"{prefix}/is_bad_batch"] = int(batch_stats['is_bad_batch'])
            
            # Log to wandb
            wandb.log(log_dict)
            
            # Log histograms of accumulated statistics periodically
            if self.batch_count % (self.log_frequency * 5) == 0:
                histograms = {}
                
                for key, values in self.batch_stats.items():
                    if len(values) >= 10:  # Only log histograms with enough data
                        histograms[f"{prefix}/hist_{key}"] = wandb.Histogram(values)
                
                if histograms:
                    wandb.log(histograms)
            
            # Count of bad batches vs total
            wandb.log({
                f"{prefix}/total_count": self.batch_count,
                f"{prefix}/bad_count": self.bad_batch_count,
                f"{prefix}/bad_ratio": self.bad_batch_count / self.batch_count if self.batch_count > 0 else 0
            })
                
        except ImportError:
            memory_logger.warning("Failed to log batch stats: wandb not available")
        except Exception as e:
            memory_logger.warning(f"Failed to log batch stats to wandb: {str(e)}")
    
    def get_batch_history_summary(self):
        """Get a summary of the batch history for debugging"""
        summary = {
            'total_batches': self.batch_count,
            'bad_batches': self.bad_batch_count,
            'bad_ratio': self.bad_batch_count / self.batch_count if self.batch_count > 0 else 0,
            'thresholds': self.thresholds
        }
        
        # Calculate statistics from batch_stats
        stats_summary = {}
        for key, values in self.batch_stats.items():
            if len(values) > 0:
                stats_summary[key] = {
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
        
        summary['stats'] = stats_summary
        
        # Include recent bad batch fingerprints
        if self.bad_batch_history:
            summary['recent_bad_batches'] = [
                {
                    'hash': item['fingerprint']['hash'],
                    'reasons': item['stats'].get('bad_batch_reasons', ['Unknown'])
                }
                for item in list(self.bad_batch_history)[-5:]  # Last 5 bad batches
            ]
        
        return summary