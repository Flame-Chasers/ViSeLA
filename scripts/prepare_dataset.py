"""
Dataset preparation script.

Converts the official ICFG-PEDES and RSTPReid annotation files into the
``train_reid_en.json`` / ``test_reid_en.json`` format expected by this project.

Usage:
    python scripts/prepare_dataset.py --dataset icfg --icfg-json /path/to/ICFG-PEDES/ICFG-PEDES.json
    python scripts/prepare_dataset.py --dataset rstp --rstp-json /path/to/RSTPReid/data_captions.json
    python scripts/prepare_dataset.py --dataset both --icfg-json ... --rstp-json ... --output-root data
"""

import json
import os
import argparse
from pathlib import Path


def process_icfg_pedes(src_json: str, output_dir: str):
    """
    Process the ICFG-PEDES dataset.

    The original format is already compatible; only the train/test split is separated.
    """
    print("Processing ICFG-PEDES...")

    with open(src_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by split
    train_data = []
    test_data = []

    for item in data:
        # Keep only the fields used by this project
        processed_item = {
            'id': item['id'],
            'file_path': item['file_path'],
            'captions': item['captions'],
            'split': item['split']
        }

        if item['split'] == 'train':
            train_data.append(processed_item)
        else:  # test or val
            test_data.append(processed_item)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save (English only; ICFG-PEDES is an English dataset)
    train_file = os.path.join(output_dir, 'train_reid_en.json')
    test_file = os.path.join(output_dir, 'test_reid_en.json')

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("ICFG-PEDES done:")
    print(f"  - train: {len(train_data)} records -> {train_file}")
    print(f"  - test:  {len(test_data)} records -> {test_file}")

    # Count unique person IDs
    train_ids = set(item['id'] for item in train_data)
    test_ids = set(item['id'] for item in test_data)
    print(f"  - train identities: {len(train_ids)}")
    print(f"  - test identities:  {len(test_ids)}")


def process_rstp_reid(src_json: str, output_dir: str):
    """
    Process the RSTPReid dataset.

    ``img_path`` is renamed to ``file_path``.
    """
    print("Processing RSTPReid...")

    with open(src_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by split
    train_data = []
    test_data = []

    for item in data:
        # Field conversion: img_path -> file_path
        processed_item = {
            'id': item['id'],
            'file_path': item['img_path'],
            'captions': item['captions'],
            'split': item['split']
        }

        if item['split'] == 'train':
            train_data.append(processed_item)
        elif item['split'] == 'test':
            test_data.append(processed_item)
        # The 'val' split is not exported; merge it into test or handle it separately if needed.

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save (English only)
    train_file = os.path.join(output_dir, 'train_reid_en.json')
    test_file = os.path.join(output_dir, 'test_reid_en.json')

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("RSTPReid done:")
    print(f"  - train: {len(train_data)} records -> {train_file}")
    print(f"  - test:  {len(test_data)} records -> {test_file}")

    # Count unique person IDs
    train_ids = set(item['id'] for item in train_data)
    test_ids = set(item['id'] for item in test_data)
    print(f"  - train identities: {len(train_ids)}")
    print(f"  - test identities:  {len(test_ids)}")


def main():
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description='Convert official ICFG-PEDES / RSTPReid annotations')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['icfg', 'rstp', 'both'],
                        help='Dataset to process: icfg, rstp, or both')
    parser.add_argument('--icfg-json', type=str, default=None,
                        help='Path to the official ICFG-PEDES.json (required for icfg/both)')
    parser.add_argument('--rstp-json', type=str, default=None,
                        help='Path to the official RSTPReid data_captions.json (required for rstp/both)')
    parser.add_argument('--output-root', type=str, default=str(project_root / 'data'),
                        help='Root directory for the converted annotations (default: <repo>/data)')
    args = parser.parse_args()

    if args.dataset in ['icfg', 'both'] and not args.icfg_json:
        parser.error('--icfg-json is required when --dataset is icfg or both')
    if args.dataset in ['rstp', 'both'] and not args.rstp_json:
        parser.error('--rstp-json is required when --dataset is rstp or both')

    output_root = Path(args.output_root)
    icfg_output = output_root / 'ICFG-PEDES'
    rstp_output = output_root / 'RSTPReid'

    if args.dataset in ['icfg', 'both']:
        process_icfg_pedes(args.icfg_json, str(icfg_output))
        print()

    if args.dataset in ['rstp', 'both']:
        process_rstp_reid(args.rstp_json, str(rstp_output))
        print()

    print("=" * 50)
    print("Next, set the following keys in your training config (see config/examples/):")
    print()

    if args.dataset in ['icfg', 'both']:
        print("# ICFG-PEDES")
        print(f"anno_dir: '{icfg_output}'")
        print("image_dir: '/path/to/ICFG-PEDES/imgs'")
        print("data:")
        print("  languages: ['en']  # ICFG-PEDES ships English captions only")
        print("  use_code_switch: false  # no multilingual code-switch files")
        print()
    if args.dataset in ['rstp', 'both']:
        print("# RSTPReid")
        print(f"anno_dir: '{rstp_output}'")
        print("image_dir: '/path/to/RSTPReid/imgs'")
        print("data:")
        print("  languages: ['en']  # RSTPReid ships English captions only")
        print("  use_code_switch: false  # no multilingual code-switch files")


if __name__ == '__main__':
    main()
