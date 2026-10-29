#!/usr/bin/env python3
"""Downloads TSD-SR's LoRA weights and prompt embeddings from the authors' Drive folder.

    tools/max/fetch_tsdsr.py OUT_DIR

Writes OUT_DIR/tsdsr/{transformer,vae}.safetensors and OUT_DIR/default/{prompt,pool}_embeds.pt,
where test/test_tsdsr.py expects them (--lora_dir, --embedding_dir). The folder also holds test
datasets and the -mse variant; those are skipped.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ntire'))
from candidates import list_drive  # noqa: E402

FOLDER = '1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI'  # github.com/Microtreei/TSD-SR README, "GoogleDrive"


def leaf(rel):
    return rel.lower().rsplit('/', 1)[-1]


def main():
    import gdown
    out = sys.argv[1]
    files = []
    # Descend only into the weights and embeddings folders (and folders that hold them).
    keep = lambda rel: leaf(rel) in ('tsdsr', 'default', 'checkpoint', 'checkpoints', 'dataset')
    print('Drive folder:')
    list_drive(FOLDER, '', keep, files)
    for _, rel in files:
        print('  file', rel)
    want = {('tsdsr', 'transformer.safetensors'), ('tsdsr', 'vae.safetensors'),
            ('default', 'prompt_embeds.pt'), ('default', 'pool_embeds.pt')}
    got = set()
    for fid, rel in files:
        parts = rel.split('/')
        key = (parts[-2].lower() if len(parts) > 1 else '', parts[-1])
        if key not in want or key in got:
            continue
        dest = os.path.join(out, *key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print('downloading', rel, flush=True)
        gdown.download(id=fid, output=dest, quiet=True)
        print('  ', os.path.getsize(dest) // 1024 // 1024, 'MB', flush=True)
        got.add(key)
    missing = want - got
    if missing:
        sys.exit(f'not found in the Drive folder: {sorted(missing)}')


if __name__ == '__main__':
    main()
