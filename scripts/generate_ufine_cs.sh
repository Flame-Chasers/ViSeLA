#!/bin/bash
#
# Generate multilingual code-switch augmentation files for UFineBench.
#
# Usage:
#   ./scripts/generate_ufine_cs.sh
#
# Edit INPUT_DIR / OUTPUT_DIR below (or export them before running) so that
# INPUT_DIR contains train_reid_{en,ch,de,fr}.json.

# Settings
INPUT_DIR="${INPUT_DIR:-data/UFineBench/annos}"
OUTPUT_DIR="${OUTPUT_DIR:-data/UFineBench/code_switch}"
DEVICE="${DEVICE:-cuda:0}"
NUM_VARIANTS="${NUM_VARIANTS:-2}"
SEED="${SEED:-42}"

# Languages to process
LANGUAGES=("en" "ch" "de" "fr")

# Create the output directory
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "Generating UFineBench code-switch data"
echo "========================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device:           $DEVICE"
echo "Languages:        ${LANGUAGES[*]}"
echo "Variants/caption: $NUM_VARIANTS"
echo "========================================="
echo ""

# Generate one file per language
for lan in "${LANGUAGES[@]}"; do
    echo "===================================="
    echo "Generating code-switch data for: $lan"
    echo "===================================="

    python tools/offline_code_switch.py \
        --lang $lan \
        --input $INPUT_DIR/train_reid_$lan.json \
        --output $OUTPUT_DIR/train_reid_${lan}_cs.json \
        --cache $INPUT_DIR/.cs_cache_$lan.json \
        --progress $INPUT_DIR/.cs_progress_$lan.jsonl \
        --device $DEVICE \
        --num-variants $NUM_VARIANTS \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "[OK] code-switch data generated for $lan"
    else
        echo "[FAILED] code-switch generation failed for $lan"
        exit 1
    fi
    echo ""
done

echo "========================================="
echo "Code-switch data generated for all languages."
echo "========================================="
echo ""
echo "Generated files:"
ls -lh $OUTPUT_DIR/train_reid_*_cs.json

echo ""
echo "Next steps:"
echo "1. Enable code-switch in your UFineBench config (see config/examples/visela_ufinebench.yaml):"
echo "   data:"
echo "     use_code_switch: true"
echo "     code_switch_dir: '$OUTPUT_DIR/'"
echo ""
echo "2. Start training:"
echo "   torchrun --nproc_per_node=4 main.py --config local_visela_ufinebench.yaml"
