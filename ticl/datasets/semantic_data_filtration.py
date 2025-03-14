import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ColumnNameProcessor:
    """
    Process column names to:
    1. Filter out low-quality names without clear semantic meaning
    2. Expand abbreviations to full words where possible
    3. Normalize and standardize formats
    """

    def __init__(self, use_transformers=True, batch_size=256, gpu=True, backend='auto'):
        """
        Initialize the processor with necessary models and resources.

        Args:
            use_transformers: Whether to use transformer models
            batch_size: Size of batches for processing
            gpu: Whether to use GPU if available
            backend: Which backend to use for acceleration - 'auto', 'cuda', 'mps', 'rocm', or 'cpu'
        """
        self.use_transformers = use_transformers
        self.batch_size = batch_size
        self.common_abbreviations = self._load_common_abbreviations()
        self.domain_abbreviations = self._load_domain_specific_abbreviations()
        self.backend = backend

        # Regex patterns for filtering
        self.patterns = {
            'alphanumeric_only': re.compile(r'^[a-zA-Z0-9_]+$'),
            'contains_letters': re.compile(r'[a-zA-Z]'),
            'likely_id': re.compile(r'^(id|identifier|key|code|uuid|num|number)$', re.IGNORECASE),
            'single_letters': re.compile(r'^[a-zA-Z]$'),
            'unnamed': re.compile(r'unnamed|^col_\d+$|^column\d+$', re.IGNORECASE),
            'placeholder': re.compile(r'^(null|none|na|nan|undefined|unknown)$', re.IGNORECASE),
            'sql_artifact': re.compile(r'select|from|where|join|insert|delete|update', re.IGNORECASE),
            'timestamp_parts': re.compile(r'^(year|month|day|hour|minute|second|date|time)$', re.IGNORECASE),
        }

        # Setup hardware acceleration
        self._setup_hardware_acceleration()

        # Load NLP resources
        print("Loading NLP resources...")
        if self.use_transformers:
            try:
                # Use lighter models for better performance
                # Zero-shot classification for semantic filtering
                self.zero_shot = pipeline("zero-shot-classification",
                                         model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                                         device=self.device_id)

                # Fill-mask for abbreviation expansion - use smaller model
                self.fill_mask = pipeline("fill-mask",
                                         model="distilbert-base-uncased",
                                         device=self.device_id)
            except Exception as e:
                print(f"Error loading transformers models: {e}")
                print("Falling back to rule-based processing")
                self.use_transformers = False

        # Load spaCy for part-of-speech tagging and NER
        # Use the smaller model for better performance
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Disable components we don't need for better performance
            self.nlp.disable_pipes("parser", "ner")
        except:
            print("Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.disable_pipes("parser", "ner")

    def _setup_hardware_acceleration(self):
        """
        Setup hardware acceleration based on available backends and user preference.
        """
        import torch
        
        self.device_id = -1  # Default to CPU
        self.device_name = "CPU"
            
        # Auto-detect available backends if set to 'auto'
        if self.backend == 'auto':
            if torch.cuda.is_available():
                self.backend = 'cuda'
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.backend = 'mps'
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.backend = 'xpu'  # Intel XPU
            elif hasattr(torch, 'hip') and torch.hip.is_available():
                self.backend = 'rocm'
            else:
                self.backend = 'cpu'
                
        # Setup the selected backend
        if self.backend == 'cuda' and torch.cuda.is_available():
            self.device_id = 0  # Use first GPU
            self.device_name = torch.cuda.get_device_name(0)
            print(f"CUDA GPU detected: {self.device_name}")
            torch.cuda.empty_cache()
            
        elif self.backend == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.device_id = "mps"
            self.device_name = "Apple Silicon MPS"
            print(f"Apple MPS detected: {self.device_name}")
            
        elif self.backend == 'rocm' and hasattr(torch, 'hip') and torch.hip.is_available():
            self.device_id = 0  # Use first GPU
            self.device_name = "AMD ROCm GPU"
            print(f"ROCm GPU detected: {self.device_name}")
            
        elif self.backend == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device_id = 0
            self.device_name = "Intel XPU"
            print(f"Intel XPU detected: {self.device_name}")
            
        else:
            print(f"No {self.backend} acceleration detected or requested backend not available, using CPU")
            self.backend = 'cpu'
            
        # For HuggingFace's device handling
        if self.backend == 'mps':
            # HuggingFace models need special handling for MPS
            # The actual device will be managed by PyTorch
            self.device_id = 0
            
        print(f"Using {self.backend.upper()} acceleration")

    def _load_common_abbreviations(self):
        """Load a dictionary of common abbreviations."""
        return {
            'addr': 'address',
            'admin': 'administrator',
            'amt': 'amount',
            'approx': 'approximate',
            'avg': 'average',
            'bday': 'birthday',
            'bldg': 'building',
            'calc': 'calculation',
            'cat': 'category',
            'cnt': 'count',
            'co': 'company',
            'corp': 'corporation',
            'cust': 'customer',
            'db': 'database',
            'dept': 'department',
            'desc': 'description',
            'dev': 'development',
            'diff': 'difference',
            'dir': 'directory',
            'dist': 'distance',
            'doc': 'document',
            'dob': 'date of birth',
            'dup': 'duplicate',
            'dur': 'duration',
            'ed': 'editor',
            'emp': 'employee',
            'est': 'estimate',
            'exec': 'executive',
            'exp': 'expiration',
            'ext': 'extension',
            'fax': 'facsimile',
            'feb': 'february',
            'fig': 'figure',
            'fn': 'function',
            'freq': 'frequency',
            'govt': 'government',
            'hr': 'hour',
            'hrs': 'hours',
            'ht': 'height',
            'id': 'identifier',
            'idx': 'index',
            'inc': 'incorporated',
            'info': 'information',
            'init': 'initialize',
            'int': 'integer',
            'jan': 'january',
            'lang': 'language',
            'lat': 'latitude',
            'lib': 'library',
            'lim': 'limit',
            'lng': 'longitude',
            'loc': 'location',
            'log': 'logarithm',
            'max': 'maximum',
            'mgmt': 'management',
            'mgr': 'manager',
            'min': 'minimum',
            'misc': 'miscellaneous',
            'mkt': 'market',
            'mth': 'month',
            'nbr': 'number',
            'neg': 'negative',
            'no': 'number',
            'num': 'number',
            'obj': 'object',
            'org': 'organization',
            'orig': 'original',
            'pct': 'percent',
            'pd': 'paid',
            'ph': 'phone',
            'pic': 'picture',
            'pkg': 'package',
            'pop': 'population',
            'pos': 'position',
            'prev': 'previous',
            'prod': 'product',
            'prog': 'program',
            'proj': 'project',
            'prov': 'province',
            'pt': 'point',
            'qty': 'quantity',
            'qtr': 'quarter',
            'rec': 'record',
            'ref': 'reference',
            'reg': 'region',
            'req': 'request',
            'res': 'resource',
            'rev': 'revenue',
            'rmv': 'remove',
            'rpt': 'report',
            'rslt': 'result',
            'sch': 'schedule',
            'sec': 'section',
            'sel': 'select',
            'seq': 'sequence',
            'sig': 'signature',
            'spec': 'specification',
            'src': 'source',
            'std': 'standard',
            'str': 'string',
            'subj': 'subject',
            'sys': 'system',
            'tbl': 'table',
            'tel': 'telephone',
            'temp': 'temperature',
            'tmp': 'temporary',
            'tot': 'total',
            'tran': 'transaction',
            'tx': 'transaction',
            'txt': 'text',
            'typ': 'type',
            'url': 'web address',
            'usr': 'user',
            'util': 'utility',
            'val': 'value',
            'var': 'variable',
            'ver': 'version',
            'vol': 'volume',
            'vs': 'versus',
            'wt': 'weight',
            'yr': 'year',
        }

    def _load_domain_specific_abbreviations(self):
        """Load domain-specific abbreviations (finance, healthcare, etc.)"""
        return {
            # Finance
            'acct': 'account',
            'bal': 'balance',
            'cap': 'capital',
            'cf': 'cash flow',
            'cogs': 'cost of goods sold',
            'cr': 'credit',
            'depr': 'depreciation',
            'div': 'dividend',
            'ebit': 'earnings before interest and taxes',
            'ebitda': 'earnings before interest taxes depreciation and amortization',
            'eps': 'earnings per share',
            'fy': 'fiscal year',
            'gl': 'general ledger',
            'gp': 'gross profit',
            'ltd': 'long term debt',
            'ni': 'net income',
            'noi': 'net operating income',
            'roe': 'return on equity',
            'roi': 'return on investment',
            'ytd': 'year to date',

            # Healthcare
            'bp': 'blood pressure',
            'bmi': 'body mass index',
            'cpt': 'current procedural terminology',
            'diag': 'diagnosis',
            'dob': 'date of birth',
            'dx': 'diagnosis',
            'er': 'emergency room',
            'hb': 'hemoglobin',
            'hc': 'healthcare',
            'ht': 'height',
            'icd': 'international classification of diseases',
            'los': 'length of stay',
            'meds': 'medications',
            'mr': 'medical record',
            'pt': 'patient',
            'rx': 'prescription',
            'sx': 'symptoms',
            'tx': 'treatment',
            'wt': 'weight',

            # Technology
            'api': 'application programming interface',
            'avg': 'average',
            'cpu': 'central processing unit',
            'css': 'cascading style sheets',
            'db': 'database',
            'dns': 'domain name system',
            'fps': 'frames per second',
            'ftp': 'file transfer protocol',
            'gui': 'graphical user interface',
            'html': 'hypertext markup language',
            'http': 'hypertext transfer protocol',
            'id': 'identifier',
            'ip': 'internet protocol',
            'json': 'javascript object notation',
            'lan': 'local area network',
            'os': 'operating system',
            'pw': 'password',
            'ram': 'random access memory',
            'rdbms': 'relational database management system',
            'sdk': 'software development kit',
            'sla': 'service level agreement',
            'sql': 'structured query language',
            'ssd': 'solid state drive',
            'ssl': 'secure sockets layer',
            'tcp': 'transmission control protocol',
            'ui': 'user interface',
            'url': 'uniform resource locator',
            'ux': 'user experience',
            'vm': 'virtual machine',
            'vpn': 'virtual private network',
            'xml': 'extensible markup language',
        }

    def _has_semantic_meaning(self, column_name):
        """
        Use linguistic features to determine if a column name has semantic meaning.
        """
        if not column_name or not isinstance(column_name, str):
            return False

        # Skip very short or very long column names
        if len(column_name) <= 1 or len(column_name) > 50:
            return False

        # Skip common patterns that indicate non-semantic names
        if (self.patterns['unnamed'].search(column_name) or
            self.patterns['placeholder'].search(column_name) or
            self.patterns['sql_artifact'].search(column_name)):
            return False

        # Exclude single letters
        if self.patterns['single_letters'].match(column_name):
            return False

        # If it's alphanumeric and contains some letters, it might be meaningful
        if (self.patterns['alphanumeric_only'].match(column_name) and
            self.patterns['contains_letters'].search(column_name)):
            # Check if it's just an ID column (less semantic value)
            if self.patterns['likely_id'].match(column_name):
                return False

            # Simple timestamp parts are less interesting
            if self.patterns['timestamp_parts'].match(column_name):
                return False

            # Passed all filters
            return True

        return False

    def _expand_abbreviations(self, column_name):
        """Expand common abbreviations in column names."""
        # Convert to lowercase for processing
        name = column_name.lower()

        # Split by common delimiters
        parts = re.split(r'[_\s\-\.]', name)

        # Process each part
        expanded_parts = []
        for part in parts:
            # Check if this part is a known abbreviation
            if part in self.common_abbreviations:
                expanded_parts.append(self.common_abbreviations[part])
            elif part in self.domain_abbreviations:
                expanded_parts.append(self.domain_abbreviations[part])
            else:
                expanded_parts.append(part)

        # Join back with underscores
        expanded = "_".join(expanded_parts)

        # Return the original capitalization if nothing changed
        if expanded == name:
            return column_name

        return expanded

    def _split_camel_case(self, column_name):
        """Split CamelCase column names into separate words."""
        # If there are any delimiters, assume it's not camel case
        if re.search(r'[_\s\-\.]', column_name):
            return column_name

        # Look for camel case pattern (lowercase followed by uppercase)
        matches = re.finditer(r'([a-z0-9])([A-Z])', column_name)
        positions = [m.start() + 1 for m in matches]

        if not positions:
            return column_name

        # Split at positions
        result = column_name
        offset = 0
        for pos in positions:
            result = result[:pos+offset] + '_' + result[pos+offset:]
            offset += 1

        return result.lower()

    def _assess_quality_with_transformers(self, column_names):
        """
        Use zero-shot classification to assess column name quality with optimized batching.

        Args:
            column_names: List of column names to assess

        Returns:
            List of boolean flags indicating quality
        """
        if not self.use_transformers:
            return [self._has_semantic_meaning(name) for name in column_names]

        # Apply rule-based filtering first to reduce workload for the transformer
        print("Pre-filtering with rule-based approach...")
        rule_results = [self._has_semantic_meaning(name) for name in column_names]

        # Only process names that passed rule-based filtering
        transformer_candidates = []
        transformer_indices = []

        for i, (name, passed_rules) in enumerate(zip(column_names, rule_results)):
            # If it failed rule-based filtering, no need to check with transformer
            if not passed_rules:
                continue

            # If it's a valid string and passed rules, add to candidates
            if isinstance(name, str) and name.strip():
                transformer_candidates.append(name)
                transformer_indices.append(i)

        print(f"After rule-based filtering: {len(transformer_candidates)} names to check with transformer")

        # If no candidates, just return the rule-based results
        if not transformer_candidates:
            return rule_results

        # Process candidates with transformer in optimized batches
        batch_size = min(self.batch_size, 512)  # Cap batch size for memory
        results = rule_results.copy()  # Start with rule-based results

        # Process in batches with progress bar
        for i in tqdm(range(0, len(transformer_candidates), batch_size),
                     desc="Transformer quality assessment"):
            batch = transformer_candidates[i:i+batch_size]
            batch_indices = transformer_indices[i:i+batch_size]

            pos_phrase = "words, phrases, names"
            neg_phrase = "letters, numbers"
            # Use zero-shot classifier
            classifications = self.zero_shot(
                batch,
                candidate_labels=[pos_phrase, neg_phrase],
                multi_label=False
            )

            # Update results
            for idx, classification in zip(batch_indices, classifications):
                labels = classification['labels']
                scores = classification['scores']
                
                if labels[0] == pos_phrase:
                    results[idx] = True
                else:
                    results[idx] = False

        return results

    def _expand_with_transformers(self, abbrev):
        """Use masked language model to expand abbreviations."""
        if not self.use_transformers or len(abbrev) <= 2:
            return None

        try:
            # Create a prompt for abbreviation expansion
            prompt = f"The abbreviation {abbrev} stands for [MASK]."
            results = self.fill_mask(prompt)

            # Get the top prediction
            for result in results:
                # Skip if the prediction is too short
                if len(result['token_str'].strip()) < len(abbrev):
                    continue

                # Return the first reasonable expansion
                if result['score'] > 0.1 and len(result['token_str'].split()) <= 3:
                    return result['token_str'].strip()

            return None
        except:
            return None

    def filter_quality_names(self, column_names, use_transformers=True, rule_only_threshold=300000):
        """
        Filter column names to keep only those with semantic meaning.

        Args:
            column_names: List of column names to filter
            use_transformers: Whether to use transformer models for quality assessment
            rule_only_threshold: If more than this many names, use only rule-based filtering

        Returns:
            List of high-quality column names
        """
        print(f"Filtering {len(column_names)} column names for quality...")

        # For very large datasets, default to rule-based filtering only
        if len(column_names) > rule_only_threshold and use_transformers:
            print(f"Dataset has {len(column_names)} names, which exceeds the transformer threshold.")
            print(f"Falling back to rule-based filtering for performance reasons.")
            use_transformers = False

        # Initial basic cleaning - vectorized when possible
        import time
        start_time = time.time()

        # Clean and filter in one pass
        cleaned_names = []
        for name in column_names:
            if isinstance(name, str):
                name = name.strip()
                if name:
                    cleaned_names.append(name)

        print(f"Basic cleaning completed in {time.time() - start_time:.2f} seconds")

        # Assess quality using the appropriate method
        if use_transformers and self.use_transformers:
            print("Using transformer models for quality assessment...")
            quality_flags = self._assess_quality_with_transformers(cleaned_names)
        else:
            print("Using rule-based quality assessment...")
            quality_flags = [self._has_semantic_meaning(name) for name in tqdm(cleaned_names, desc="Rule-based filtering")]

        # Keep only high-quality names
        quality_names = [name for name, is_quality in zip(cleaned_names, quality_flags) if is_quality]

        print(f"Kept {len(quality_names)} high-quality column names")
        return quality_names

    def enhance_names(self, column_names):
        """
        Enhance column names by:
        1. Expanding abbreviations
        2. Splitting camel case
        3. Normalizing format

        Args:
            column_names: List of column names to enhance

        Returns:
            DataFrame with original and enhanced column names
        """
        print(f"Enhancing {len(column_names)} column names...")

        # Vectorize the initial cleaning and validation
        valid_names = []
        valid_indices = []

        for i, name in enumerate(column_names):
            if isinstance(name, str) and name.strip():
                valid_names.append(name)
                valid_indices.append(i)

        # Process names using process_chunk function
        # Define the processing function for a chunk
        def process_chunk(names_chunk):
            chunk_results = []
            for name in names_chunk:
                # Step 1: Split camel case
                split_name = self._split_camel_case(name)

                # Step 2: Expand abbreviations
                expanded_name = self._expand_abbreviations(split_name)

                # Add to results
                chunk_results.append({
                    'original': name,
                    'enhanced': expanded_name,
                    'changed': name != expanded_name
                })
            return chunk_results

        # Try to use multiprocessing, but with better error handling
        try:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            
            # Determine number of workers
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            
            # Split names into chunks for parallel processing
            chunk_size = max(1, len(valid_names) // num_workers)
            name_chunks = [valid_names[i:i+chunk_size] for i in range(0, len(valid_names), chunk_size)]
            
            print(f"Processing {len(valid_names)} names in {len(name_chunks)} chunks using {num_workers} workers")
            
            all_results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in name_chunks]
                
                # Collect results with progress bar
                for future in tqdm(futures, total=len(futures), desc="Enhancing names"):
                    all_results.extend(future.result())
        
        except Exception as e:
            print(f"Multiprocessing failed with error: {e}")
            print("Falling back to sequential processing")
            
            # Sequential processing as fallback
            all_results = process_chunk(valid_names)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_results)

        print(f"Enhanced {df['changed'].sum()} column names")
        return df

    def process_names(self, column_names, quality_threshold=0.7):
        """
        Combined processing pipeline:
        1. Filter for quality names
        2. Enhance names with expansions
        3. Remove duplicates

        Args:
            column_names: List of column names to process
            quality_threshold: Quality threshold for filtering

        Returns:
            DataFrame with processed column names
        """
        # Performance optimization: Remove duplicates first to reduce workload
        print("Removing duplicates from input...")
        unique_names = list(set(column_names))
        print(f"Reduced from {len(column_names)} to {len(unique_names)} unique names")

        # Remove very short and very long names early
        print("Removing very short and very long names...")
        filtered_names = [name for name in unique_names
                         if isinstance(name, str)
                         and len(name) > 2
                         and len(name) < 50]
        print(f"Removed {len(unique_names) - len(filtered_names)} names based on length")

        # Step 1: Filter for quality
        quality_names = self.filter_quality_names(filtered_names)

        # Checkpoint: save intermediates in case of crash
        import pickle
        with open('quality_names_checkpoint.pkl', 'wb') as f:
            pickle.dump(quality_names, f)

        # Step 2: Enhance names
        enhanced_df = self.enhance_names(quality_names)

        # Step 3: Remove duplicates (keeping the first occurrence)
        enhanced_df = enhanced_df.drop_duplicates(subset=['enhanced'], keep='first')

        print(f"Final result: {len(enhanced_df)} processed column names")
        return enhanced_df

    def find_best_expansion(self, name, quality_names, top_n=5):
        """
        Find the best expansion for a name by semantic similarity to other quality names.

        Args:
            name: The column name to expand
            quality_names: List of high-quality column names for context
            top_n: Number of top candidates to consider

        Returns:
            Best expansion for the name
        """
        # Check if it's a known abbreviation
        if name in self.common_abbreviations:
            return self.common_abbreviations[name]
        elif name in self.domain_abbreviations:
            return self.domain_abbreviations[name]

        if self.use_transformers:
            # Try using transformers
            expansion = self._expand_with_transformers(name)
            if expansion:
                return expansion

        # If all else fails, return the original
        return name
    
def main(input_types, output_file="processed_schema_types.py",
         use_transformers=True, batch_size=256, gpu=True, 
         backend='auto', sample_size=None):
    """
    Process a list of schema types to improve quality and expand abbreviations.

    Args:
        input_types: List of types to process
        output_file: Output file path
        use_transformers: Whether to use transformer models
        batch_size: Batch size for processing
        gpu: Whether to use GPU if available
        backend: Which backend to use for acceleration - 'auto', 'cuda', 'mps', 'rocm', or 'cpu'
        sample_size: If not None, process only a sample of the input
    """
    print(f"Processing {len(input_types)} input types")

    # Sample data if requested (useful for testing with large datasets)
    if sample_size and len(input_types) > sample_size:
        print(f"Sampling {sample_size} types from {len(input_types)} total")
        import random
        sampled_types = random.sample(input_types, sample_size)
    else:
        sampled_types = input_types

    # Create processor with optimized settings
    processor = ColumnNameProcessor(
        use_transformers=use_transformers,
        batch_size=batch_size,
        gpu=gpu,
        backend=backend
    )

    # Process the input types
    result_df = processor.process_names(sampled_types)

    # Prepare output
    enhanced_types = result_df['enhanced'].tolist()

    # Write to output file
    with open(output_file, 'w') as f:
        f.write("# Processed Schema.org types and column names\n\n")
        f.write("SCHEMA_TYPES = [\n")

        # Sort alphabetically for better readability
        sorted_types = sorted(enhanced_types)

        # Write in chunks of 5 for better readability
        for i in range(0, len(sorted_types), 5):
            chunk = sorted_types[i:i+5]
            f.write("    " + ", ".join([f'"{t}"' for t in chunk]) + ",\n")

        f.write("]\n")

    # Also save the DataFrame for analysis
    result_df.to_csv("column_name_transformations.csv", index=False)

    print(f"Processed types saved to {output_file}")
    print(f"Transformation details saved to column_name_transformations.csv")

    return enhanced_types

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Process column names with multi-backend support')
    parser.add_argument('--backend', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'rocm', 'xpu', 'cpu'],
                        help='Hardware acceleration backend to use')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-transformers', dest='use_transformers', action='store_false',
                        help='Disable transformer models, use rule-based only')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for processing')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size (for testing with large datasets)')
    parser.add_argument('--output', type=str, default='processed_schema_types.py',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Import the schema types
    try:
        from schema_types import SCHEMA_TYPES
        print(f"Loaded {len(SCHEMA_TYPES)} schema types from schema_types.py")
    except ImportError:
        print("schema_types.py not found. Using sample data for testing...")
        SCHEMA_TYPES = [
            "ProductID", "CustomerName", "OrderDate", "ShippingAddress",
            "PaymentMethod", "ItemQuantity", "UnitPrice", "TotalAmount",
            "dob", "phNum", "ccNum", "empID", "userPW", "acctBal"
        ]
        
    # Process the types with the specified backend
    enhanced_types = main(
        SCHEMA_TYPES,
        output_file=args.output,
        use_transformers=args.use_transformers,
        batch_size=args.batch_size,
        gpu=args.gpu,
        backend=args.backend,
        sample_size=args.sample
    )
    
    # Print some examples
    print("\nSample of enhanced column names:")
    for i in range(min(10, len(enhanced_types))):
        print(f"- {enhanced_types[i]}")