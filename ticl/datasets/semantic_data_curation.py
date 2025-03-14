import os
import requests
import json
import os
import pandas as pd
import random
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import openml
from bs4 import BeautifulSoup

def fetch_schemaorg_types():
    """Fetch all types from the Schema.org JSON-LD API endpoint"""
    print("Fetching Schema.org types...")

    response = requests.get("https://schema.org/version/latest/schemaorg-all-https.jsonld")
    if response.status_code != 200:
        print(f"Failed to fetch Schema.org data: {response.status_code}")
        return []

    schema_data = response.json()
    types = []

    # The structure is different than expected - let's properly extract the types
    for item in schema_data.get("@graph", []):
        # Get all rdfs:Class entries (these are the types/classes)
        if isinstance(item.get("@type"), list) and "rdfs:Class" in item.get("@type"):
            type_id = item.get("@id")
            if type_id:
                # Handle both format patterns
                if type_id.startswith("schema:"):
                    clean_type = type_id.replace("schema:", "")
                    types.append(clean_type)
                elif type_id.startswith("http://schema.org/"):
                    clean_type = type_id.replace("http://schema.org/", "")
                    types.append(clean_type)
        # Some items have @type as string instead of list
        elif item.get("@type") == "rdfs:Class":
            type_id = item.get("@id")
            if type_id:
                if type_id.startswith("schema:"):
                    clean_type = type_id.replace("schema:", "")
                    types.append(clean_type)
                elif type_id.startswith("http://schema.org/"):
                    clean_type = type_id.replace("http://schema.org/", "")
                    types.append(clean_type)

    # If we still have no types, let's try an alternative approach
    if not types:
        print("Trying alternative Schema.org extraction method...")

        # Another common pattern in the Schema.org JSON-LD
        for item in schema_data.get("@graph", []):
            if "@id" in item and "@type" in item:
                # Get the type ID
                type_id = item.get("@id", "")

                # Check if it's a Schema.org type
                if "schema.org" in type_id or "schema:" in type_id:
                    # Extract just the type name (after the last / or :)
                    if "/" in type_id:
                        clean_type = type_id.split("/")[-1]
                    elif ":" in type_id:
                        clean_type = type_id.split(":")[-1]
                    else:
                        clean_type = type_id

                    if clean_type and len(clean_type) > 1:
                        types.append(clean_type)

    # Remove duplicates
    types = list(set(types))

    print(f"Found {len(types)} Schema.org types")
    return types

def fetch_wikitables_columns(sample_size=1000, url="http://websail-fe.cs.northwestern.edu/TabEL/tables.json.gz"):
    """
    Fetch column names from WikiTables dataset.
    Handles the large (37+ GiB) dataset by streaming it and processing tables one by one.
    """
    print("Fetching WikiTables column names...")

    try:
        import gzip
        import json
        import tempfile
        import shutil
        from tqdm import tqdm

        # Create a temporary directory for downloading
        temp_dir = tempfile.mkdtemp()
        gz_file_path = os.path.join(temp_dir, "tables.json.gz")

        # Check if we should download or use an existing file
        if os.path.exists("tables.json.gz"):
            print(f"Using existing tables.json.gz file")
            gz_file_path = "tables.json.gz"
        else:
            print(f"Downloading WikiTables from {url} (this might take a while)...")
            # Download the file with progress indicator
            with requests.get(url, stream=True) as response:
                total_size = int(response.headers.get('content-length', 0))
                with open(gz_file_path, 'wb') as f, tqdm(
                    desc="Downloading WikiTables",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)

        print("Processing WikiTables data...")

        # Stream-process the gzipped JSON file
        columns = []
        table_count = 0
        sample_indices = set(random.sample(range(500000), min(sample_size, 500000)))  # Assume up to 500,000 tables

        with gzip.open(gz_file_path, 'rt') as f:
            # Handle the file format - it might be one JSON object per line or a single large JSON array
            try:
                # First try to parse as a JSON array
                tables_data = []
                try:
                    # Try to read up to 5 lines to check format
                    buffer = ""
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            break
                        buffer += line

                    # Reset file pointer
                    f.seek(0)

                    # If buffer starts with '[', it's likely a JSON array
                    if buffer.strip().startswith('['):
                        print("Detected JSON array format")
                        # Process as a JSON array with an iterator
                        decoder = json.JSONDecoder()
                        buffer = ""

                        # Process the opening bracket
                        line = f.readline().strip()
                        if line.startswith('['):
                            line = line[1:]

                        while line:
                            buffer += line.strip()
                            try:
                                obj, pos = decoder.raw_decode(buffer)
                                tables_data.append(obj)
                                buffer = buffer[pos:].strip()

                                # Process comma
                                if buffer.startswith(','):
                                    buffer = buffer[1:].strip()

                                # Sample table
                                if table_count in sample_indices:
                                    process_table(obj, columns)

                                table_count += 1
                                if table_count % 1000 == 0:
                                    print(f"Processed {table_count} tables, found {len(columns)} column names")

                                if table_count >= max(sample_indices) + 1:
                                    break

                            except json.JSONDecodeError:
                                # Need more data
                                line = f.readline()
                                continue

                            line = f.readline()
                    else:
                        # Process as one JSON object per line
                        print("Detected JSON Lines format")
                        for line_num, line in enumerate(f):
                            if line_num in sample_indices:
                                try:
                                    table_data = json.loads(line)
                                    process_table(table_data, columns)
                                except json.JSONDecodeError:
                                    continue

                            # Print progress
                            if line_num % sample_size == 0:
                                print(f"Processed {line_num} tables, found {len(columns)} column names")

                            if line_num >= max(sample_indices) + 1:
                                break

                            table_count = line_num + 1

                except json.JSONDecodeError:
                    # If JSON array parsing fails, try parsing the sample
                    print("Processing from provided sample")
                    sample_data = json.loads(open('paste.txt').read())
                    for table in sample_data:
                        process_table(table, columns)

            except Exception as e:
                print(f"Error processing WikiTables JSON: {str(e)}")
                # Parse the provided sample as fallback
                try:
                    print("Trying to parse the provided sample as fallback")
                    sample_data = json.loads(open('paste.txt').read())
                    for table in sample_data:
                        process_table(table, columns)
                except Exception as e2:
                    print(f"Error processing sample: {str(e2)}")

        # # Clean up temporary directory if we downloaded the file
        # if temp_dir and os.path.exists(temp_dir) and gz_file_path.startswith(temp_dir):
        #     shutil.rmtree(temp_dir)

        # Remove duplicates
        unique_columns = list(set(columns))

        print(f"Found {len(unique_columns)} unique column names from {table_count} WikiTables")
        return unique_columns

    except Exception as e:
        print(f"Error fetching WikiTables data: {str(e)}")
        # Try to process the uploaded sample as a last resort
        try:
            print("Processing uploaded sample as fallback")
            sample_data = json.loads(open('paste.txt').read())
            columns = []
            for table in sample_data:
                if isinstance(table, list):
                    # Handle nested table structure
                    for row in table:
                        for cell in row:
                            if isinstance(cell, dict) and "text" in cell:
                                if cell["text"] and not cell.get("isNumeric", False):
                                    columns.append(cell["text"])
                elif isinstance(table, dict):
                    # Try to extract from tableHeaders
                    if "tableHeaders" in table:
                        for header_row in table["tableHeaders"]:
                            for cell in header_row:
                                if isinstance(cell, dict) and "text" in cell:
                                    if cell["text"] and not cell.get("isNumeric", False):
                                        columns.append(cell["text"])

            unique_columns = list(set(columns))
            print(f"Found {len(unique_columns)} column names from sample")
            return unique_columns
        except Exception as e2:
            print(f"Error processing sample: {str(e2)}")
            return []

def process_table(table_data, columns_list):
    """Helper function to extract column names from a WikiTables table."""
    try:
        # Handle the wikitables specific format
        if isinstance(table_data, dict) and "tableHeaders" in table_data:
            # Extract from tableHeaders
            for header_row in table_data["tableHeaders"]:
                for cell in header_row:
                    if isinstance(cell, dict) and "text" in cell:
                        text = cell["text"]
                        if text and isinstance(text, str) and len(text) > 0:
                            columns_list.append(text)
        elif isinstance(table_data, list):
            # First row might be header
            for cell in table_data[0]:
                if isinstance(cell, dict) and "text" in cell:
                    text = cell["text"]
                    if text and isinstance(text, str) and len(text) > 0:
                        columns_list.append(text)
    except Exception as e:
        # Silently continue for individual table errors
        pass

def fetch_schemaorg_types_html():
    """
    Alternative method to fetch Schema.org types by scraping the HTML.
    This is a fallback in case the JSON-LD API fails.
    """
    print("Fetching Schema.org types using HTML method...")

    try:
        # Method 1: Fetch from the full.html page
        response = requests.get("https://schema.org/docs/full.html")
        if response.status_code != 200:
            print(f"Failed to fetch Schema.org HTML data: {response.status_code}")

            # Method 2: Try the tree view instead
            response = requests.get("https://schema.org/docs/tree.html")
            if response.status_code != 200:
                print(f"Failed to fetch Schema.org tree HTML data: {response.status_code}")
                return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all types - they are typically in links with specific classes
        types = []

        # Method 1: Look for links with class 'class'
        class_links = soup.select("a.class")
        for link in class_links:
            type_name = link.text.strip()
            if type_name:
                types.append(type_name)

        # Method 2: Look for links to schema.org URLs
        if not types:
            for link in soup.find_all("a"):
                href = link.get("href", "")
                if "schema.org" in href and "/" in href:
                    # Extract the type name from the URL
                    type_name = href.split("/")[-1]
                    if type_name and type_name not in ["docs", "schema.org", ""]:
                        types.append(type_name)

        # Remove duplicates
        types = list(set(types))

        print(f"Found {len(types)} Schema.org types using HTML method")
        return types

    except Exception as e:
        print(f"Error fetching Schema.org HTML: {str(e)}")
        return []

def fetch_openml_columns(sample_size=100):
    """
    Fetch column names from random OpenML datasets.
    """
    print("Fetching OpenML column names...")

    try:
        # Get list of all datasets
        datasets = openml.datasets.list_datasets(output_format="dataframe")

        # Sample a subset of datasets
        if len(datasets) > sample_size:
            sampled_indices = random.sample(range(len(datasets)), sample_size)
            sampled_datasets = datasets.iloc[sampled_indices]
        else:
            sampled_datasets = datasets

        all_columns = []
        for _, dataset_info in tqdm(sampled_datasets.iterrows(), total=len(sampled_datasets)):
            try:
                # Get dataset ID
                dataset_id = dataset_info['did']

                # Get dataset
                dataset = openml.datasets.get_dataset(dataset_id)

                # Get features (columns)
                features = dataset.features
                if features:
                    column_names = [feature.name for feature in features.values() if hasattr(feature, 'name')]
                    all_columns.extend([col.strip() for col in column_names if isinstance(col, str) and col.strip()])
            except Exception as e:
                print(f"Error processing OpenML dataset {dataset_id}: {str(e)}")

        print(f"Found {len(all_columns)} column names from OpenML")
        return all_columns

    except Exception as e:
        print(f"Error with OpenML API: {str(e)}")
        return []

def fetch_kaggle_columns(sample_size=100, max_size_bytes=100000000):
    """
    Fetch column names from random Kaggle datasets.
    Requires Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    print("Fetching Kaggle column names...")

    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Get a list of public datasets (over multiple pages)
        all_datasets = []
        page = 1
        max_pages = 40  # Limit how many pages to check

        while len(all_datasets) < sample_size * 3 and page <= max_pages:
            # Get datasets from current page
            page_datasets = list(api.dataset_list(sort_by="updated", page=page))
            if not page_datasets:
                break

            # Add to our collection
            all_datasets.extend(page_datasets)
            page += 1

        print(f"Retrieved {len(all_datasets)} Kaggle datasets for filtering")

        # Filter datasets by size
        filtered_datasets = []
        for dataset in all_datasets:
            # Some datasets don't have totalBytes attribute
            if hasattr(dataset, 'totalBytes') and dataset.totalBytes is not None:
                if dataset.totalBytes <= max_size_bytes:
                    filtered_datasets.append(dataset)

        print(f"Found {len(filtered_datasets)} datasets under {max_size_bytes/1000000:.1f}MB")

        # Sample a subset of datasets
        if len(filtered_datasets) > sample_size:
            sampled_datasets = random.sample(filtered_datasets, sample_size)
        else:
            sampled_datasets = filtered_datasets

        print(f"Sampling {len(sampled_datasets)} datasets for column extraction")

        all_columns = []
        for dataset in tqdm(sampled_datasets):
            try:
                # Get dataset information
                owner_slug = dataset.ref.split('/')[0]
                dataset_slug = dataset.ref.split('/')[1]

                # Create a directory to download the dataset
                temp_dir = f"temp_kaggle_{dataset_slug}"
                os.makedirs(temp_dir, exist_ok=True)

                # Download the dataset files
                api.dataset_download_files(f"{owner_slug}/{dataset_slug}", path=temp_dir, unzip=True)

                # Process all CSV files in the dataset
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            try:
                                # Read only the header to get column names
                                df = pd.read_csv(file_path, nrows=0)
                                columns = list(df.columns)
                                all_columns.extend([col.strip() for col in columns if isinstance(col, str) and col.strip()])
                            except Exception as e:
                                print(f"Error reading {file_path}: {str(e)}")

                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir)

            except Exception as e:
                print(f"Error processing dataset {dataset.ref}: {str(e)}")

        print(f"Found {len(all_columns)} column names from Kaggle")
        return all_columns

    except Exception as e:
        print(f"Error with Kaggle API: {str(e)}")
        print("Make sure you have set up Kaggle API credentials in ~/.kaggle/kaggle.json")
        return []

def clean_and_normalize_types(types_list):
    """Clean and normalize the types to make them usable as Python identifiers"""
    cleaned_types = []

    for type_name in types_list:
        # Convert to string in case we have non-string items
        type_name = str(type_name)

        # Remove special characters and replace spaces with underscores
        clean_name = re.sub(r'[^\w\s]', '', type_name)
        clean_name = clean_name.strip().replace(' ', '_')

        # Make sure it's a valid Python identifier
        if clean_name and clean_name[0].isalpha() and len(clean_name) > 1:
            cleaned_types.append(clean_name)

    return cleaned_types

def save_types_to_file(types_list, common_types=None, filename="schema_types.py"):
    """Save the list of types to a Python file"""
    with open(filename, 'w') as f:
        f.write("# Generated list of Schema.org types and common data column names\n\n")

        # Write the main list of all types
        f.write("SCHEMA_TYPES = [\n")

        # Sort alphabetically for better readability
        sorted_types = sorted(types_list)

        # Write in chunks of 5 for better readability
        for i in range(0, len(sorted_types), 5):
            chunk = sorted_types[i:i+5]
            f.write("    " + ", ".join([f'"{t}"' for t in chunk]) + ",\n")

        f.write("]\n\n")

        # Write the list of common types if provided
        if common_types:
            f.write("# Most common types/column names\n")
            f.write("COMMON_TYPES = [\n")

            # Write in chunks of 5 for better readability
            for i in range(0, len(common_types), 5):
                chunk = common_types[i:i+5]
                f.write("    " + ", ".join([f'"{t}"' for t in chunk]) + ",\n")

            f.write("]\n")

# Fetch types from Schema.org
schema_types = fetch_schemaorg_types()

# If Schema.org extraction failed, try an alternative method
if not schema_types:
    print("Trying direct HTML scraping for Schema.org types...")
    schema_types = fetch_schemaorg_types_html()

# Fetch column names from Kaggle
kaggle_columns = fetch_kaggle_columns(sample_size=500)

# Fetch column names from OpenML
openml_columns = fetch_openml_columns(sample_size=500)

wikitables_columns = fetch_wikitables_columns()

# Combine all sources and remove duplicates
all_types = schema_types + kaggle_columns + openml_columns + wikitables_columns

# Clean and normalize
cleaned_types = clean_and_normalize_types(all_types)

# Remove duplicates
unique_types = list(set(cleaned_types))

# Find most common types (optional)
counter = Counter(cleaned_types)
common_types = [type_name for type_name, count in counter.most_common(100)]

# Print stats about sources
print(f"\nStats by source:")
print(f"Schema.org types: {len(schema_types)}")
print(f"WikiTables columns: {len(wikitables_columns)}")
print(f"Kaggle columns: {len(kaggle_columns)}")
print(f"OpenML columns: {len(openml_columns)}")

print(f"Total unique types collected: {len(unique_types)}")
print(f"Top 10 most common types: {common_types[:10]}")

# Save to file
save_types_to_file(unique_types, common_types)
print(f"Types saved to schema_types.py")

# Create a CSV with type frequencies for analysis (optional)
df = pd.DataFrame(counter.most_common(), columns=['Type', 'Frequency'])
df.to_csv('type_frequencies.csv', index=False)
print(f"Type frequencies saved to type_frequencies.csv")