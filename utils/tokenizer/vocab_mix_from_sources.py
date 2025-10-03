# generates a mixed txt from the parts of input sources. tokenizer vocab will be made from the mixed txt.
# inputs: a list of txt file locations, a target size, a output location
# it divides target size by number of txt files inputted, gathers that much size from each file, and adds then into one txt
# stores in tokenizer/data. so it's gitignored. 

import os

def read_bytes_from_file(filepath, num_bytes):
    """Read a specific number of bytes from a file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(num_bytes)

def create_mixed_vocab_file(source_files, target_size_gb, output_file):

    target_size_bytes = int(target_size_gb * 1024 * 1024 * 1024)
    num_sources = len(source_files)
    bytes_per_source = target_size_bytes // num_sources
    total_bytes_written = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for source_file in source_files:
            content = read_bytes_from_file(source_file, bytes_per_source)
            outfile.write(content)
            outfile.write('\n')
            bytes_written = len(content.encode('utf-8'))
            print(f"  Wrote {bytes_written } bytes from file {source_file}")

# ========== HARDCODED CONFIG ==========
source_files = [
    './data/english_common/bookcorpus.txt',
    './data/english_common/gutenberg.txt',
    './data/english_common/mini_c4.txt',
    './data/english_common/wikitext103.txt'
]

target_size_gb = 1.0

output_file = 'models/classic_transformer_1b/tokenizer/data/mixed_vocab_data.txt'
# ======================================

if __name__ == '__main__':
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    create_mixed_vocab_file(source_files, target_size_gb, output_file)
