#downloads the following datasets into txts:
# book corpus
# project gluttenberg
# wiki text
# minic4

import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

def dump_txt(dataset_name, split, out_file, config=None):
    # load dataset with the split you give
    if config:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    path = os.path.join("data", out_file)
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            t = ex.get("text")
            if t:
                f.write(t.strip() + "\n")
    print(f"Wrote {path}  (N={len(ds)})")

# now YOU specify split for each one:
dump_txt("lucadiliello/bookcorpusopen", "train", "bookcorpus.txt")
dump_txt("manu/project_gutenberg", "en", "gutenberg.txt")   # uses the English split
dump_txt("mattdangerw/mini-c4", "train", "mini_c4.txt")
dump_txt("Salesforce/wikitext", "train+validation+test", "wikitext103.txt", config="wikitext-103-raw-v1")
