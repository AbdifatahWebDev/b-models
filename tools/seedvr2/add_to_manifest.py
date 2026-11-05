#!/usr/bin/env python3
"""Adds (or replaces) the Ultra model in a models.json, its files pointing at the ultra-model
release, so the app downloads them from there.

    python tools/seedvr2/add_to_manifest.py models.json ultra.json BASE_URL [--id seedvr2]

ultra.json is the piece list convert.py writes (sizes and SHA-256 of each .param and .bin);
BASE_URL is where those files are (https://github.com/<owner>/b-models/releases/download/ultra-model).
The app needs a version that reads the per-file "url" (Brekwol Upscale 0.7.0); older versions
don't know this model id and skip it.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('manifest')
    ap.add_argument('ultra')
    ap.add_argument('base_url')
    ap.add_argument('--id', default='seedvr2')
    a = ap.parse_args()
    with open(a.manifest) as f:
        manifest = json.load(f)
    with open(a.ultra) as f:
        ultra = json.load(f)
    base = a.base_url.rstrip('/')
    files = []
    for p in ultra['pieces']:
        for kind in ('param', 'bin'):
            files.append({'name': p[kind], 'size': p[f'{kind}_bytes'], 'sha256': p[f'{kind}_sha256'],
                          'url': f'{base}/{p[kind]}'})
    models = [m for m in manifest['models'] if m['id'] != a.id]
    models.append({'id': a.id, 'files': files})
    manifest['models'] = models
    with open(a.manifest, 'w', newline='\n') as f:
        f.write(json.dumps(manifest, indent=2) + '\n')
    print(f"{a.manifest}: {a.id} with {len(files)} files, {sum(x['size'] for x in files) / 2**30:.2f} GB")


if __name__ == '__main__':
    main()
