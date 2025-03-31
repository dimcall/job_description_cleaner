import os
import random
from cleaner.html_preprocessing import clean_html_format
from cleaner.utils_html import fix_merged_phrases
from cleaner.create_noisy_data import add_noise
import json
from tqdm import tqdm

# Configuration
INPUT_DIR = './data/htmls'
APPLY_NOISE_PROB = 0.4 # Probability of applying noise to the training set
MERGE_PROB = 0.5 # Probability of merging two words
JUNK_PROB = 0.15 # Probability of adding junk HTML tokens
SPLITS = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}
OUTPUT_DIR = './data/splits'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load HTML files
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.html')]
print(f'Found {len(all_files)} HTML files')

# Read files and extract number of text chunks per filename
file_info = []
for filename in all_files:
    path = os.path.join(INPUT_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    text_chunks = clean_html_format(html)
    file_info.append((filename, len(text_chunks)))

# Shuffle files
random.seed(48)
random.shuffle(file_info)

# Total number of chunks and split to 80% train, 10% val, 10% test
total_chunks = sum([n for _, n in file_info])
size_train = int(total_chunks * SPLITS['train'])
size_val = int(total_chunks * SPLITS['val'])

splits = {
    'train': [],
    'val': [],
    'test': []
}

counts = {
    'train': 0,
    'val': 0,
    'test': 0
}
# Assign files to splits - dataset is split to (80/10/10) based on the number of chunks not the number of files
for filename, n_chunks in file_info:
    if counts['train'] < size_train:
        splits['train'].append(filename)
        counts['train'] += n_chunks
    elif counts['val'] < size_val:
        splits['val'].append(filename)
        counts['val'] += n_chunks
    else:
        splits['test'].append(filename)
        counts['test'] += n_chunks
        

# Generate dataset
for split, files in splits.items():
    dataset = []
    for filename in tqdm(files):
        path = os.path.join(INPUT_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
            
        # Extract text chunks and clean them
        text_chunks = clean_html_format(html)
        # Include only chunks with 2 or more words
        target = "\n".join([fix_merged_phrases(chunk) for chunk in text_chunks if len(chunk.strip().split()) >= 2])
        # Skip empty targets
        if not target.strip():
            continue
        
        # Apply noise to the training set - val and test are clean
        if split == 'train' and random.random() < APPLY_NOISE_PROB:
            noisy = add_noise(target, merge_prob=MERGE_PROB, junk_prob=JUNK_PROB)
        else:
            noisy = target

        entry = {
            'source': filename,
            'input': noisy,
            'target': target
        }
        dataset.append(entry)

    # Save dataset
    output_path = os.path.join(OUTPUT_DIR, f'dataset_{split}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"{split}: {len(dataset)} samples")
