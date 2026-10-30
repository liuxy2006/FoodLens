"""核对官方模型版本、文件及大小；仅显式 --download 才下载。"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
FILES = ['config.json', 'generation_config.json', 'model.safetensors',
         'preprocessor_config.json', 'processor_config.json', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json', 'chat_template.json', 'chat_template.jinja']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='main')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--destination', default='models/SmolVLM-256M-Instruct')
    parser.add_argument('--manifest', default='outputs/stage2/model_manifest.json')
    args = parser.parse_args()
    with urlopen(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{args.revision}?blobs=true', timeout=60) as response:
        info = json.load(response)
    files = [{'name': f['rfilename'], 'bytes': f.get('size'), 'lfs': f.get('lfs')}
             for f in info['siblings'] if f['rfilename'] in FILES]
    result = {'model_id': MODEL_ID, 'revision': info['sha'], 'files': files,
              'download_bytes': sum(f['bytes'] or 0 for f in files), 'destination': args.destination}
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=result['revision'], local_dir=args.destination,
                          allow_patterns=[f['name'] for f in files])
        # 除库自身的下载校验外，再核对 Git LFS 文件 SHA256。
        import hashlib
        for entry in files:
            file = Path(args.destination) / entry['name']
            if file.stat().st_size != entry['bytes']:
                raise ValueError(f'file size mismatch: {file}')
            if entry['lfs']:
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                if digest != entry['lfs']['sha256']:
                    raise ValueError(f'SHA256 mismatch: {file}')
        print('Download and integrity checks complete.', flush=True)


if __name__ == '__main__':
    main()
