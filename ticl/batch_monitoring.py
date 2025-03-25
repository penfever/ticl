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
            'max_token_length': 150,              # Maximum reasonable token length
            'max_semantic_class': 10,             # Maximum expected semantic class ID
            'max_nan_ratio': 0.5,                 # Maximum ratio of NaN values allowed
            'min_class_token_patterns': 1,        # Minimum number of class token patterns
            'min_valid_targets': 5                # Minimum number of valid targets
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
    
    def analyze_batch(self, batch_info, semantic_tokens=None, semantic_targets=None):
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
                    
                    # Check for extreme token lengths
                    if stats['semantic_tokens_max'] > self.thresholds['max_token_length']:
                        is_bad_batch = True
                        reasons.append(f"Token value too large: {stats['semantic_tokens_max']}")
                
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
                    
                    # Check if we have enough valid targets
                    if valid_count < self.thresholds['min_valid_targets']:
                        is_bad_batch = True
                        reasons.append(f"Too few valid semantic targets: {valid_count}")
                    elif valid_ratio < self.thresholds['min_valid_semantic_ratio']:
                        is_bad_batch = True
                        reasons.append(f"Valid semantic target ratio too low: {valid_ratio:.4f}")
                
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