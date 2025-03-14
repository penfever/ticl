import re

file_path = "ticl/datasets/semantic_column_generation.py"

with open(file_path, 'r') as f:
    content = f.read()

# Find the standard process_columns method
pattern = r'(def process_columns\s*\(.*?\).*?)(# Load the completion log if it exists.*?# Process a sample column to determine expected tensor size.*?# Save this result in the completed columns.*?json\.dump\(completed_columns, f\))'

def replacement(match):
    prefix = match.group(1)
    # Our replacement code
    new_code = """        # Load the completion log if it exists
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)
        completed_columns = {}
        
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as f:
                    completed_columns = json.load(f)
                print(f"Loaded completion log with {len(completed_columns)} entries")
            except json.JSONDecodeError:
                print(f"Error parsing log file {log_file_path}, starting fresh")
                completed_columns = {}
        
        # Determine tensor size based on max_non_zero_tokens parameter
        sample_col = column_names[0]
        max_tensor_size = self._get_expected_tensor_size(sample_col, max_non_zero_tokens)
        print(f"Using tensor size of {max_tensor_size} based on max_non_zero_tokens={max_non_zero_tokens}")
        
        # Process first column if needed
        sample_col = next((col for col in column_names if col not in completed_columns), column_names[0])
        if sample_col not in completed_columns:
            sample_tokens = self.process_column(sample_col, max_non_zero_tokens)
            # Save this result in the completed columns
            completed_columns[sample_col] = sample_tokens.tolist()
            with open(log_file_path, 'w') as f:
                json.dump(completed_columns, f)"""
    
    return prefix + new_code

# Use regex to replace with flags to match across lines
modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open(file_path, 'w') as f:
    f.write(modified_content)

print("Process columns method updated successfully")
