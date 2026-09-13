#!/bin/bash
# Runs ON the pod, launched by lifecycle.sh. Stage 0 tokenise the audio, stage 1 finetune.
# Writes /workspace/ft/STATUS, which the local driver polls. Pins torch to the image's build so
# installing omnivoice cannot drag in a different CUDA wheel.
set -uo pipefail
cd /workspace/ft
DATA_DIR=${DATA_DIR:-clips}
st(){ echo "$1" > STATUS; echo "$(date -u +%FT%TZ) STATUS $1"; }
st INSTALLING
python -m pip freeze 2>/dev/null | grep -iE "^(torch|torchaudio|torchvision|triton)==" > constraints.txt
cat constraints.txt
python -m pip install -q --break-system-packages -c constraints.txt "omnivoice==0.2.1" accelerate tensorboard 2>&1 | tail -5
python -c "import omnivoice, torch; print('omnivoice', omnivoice.__version__ if hasattr(omnivoice,'__version__') else 'ok', 'torch', torch.__version__, torch.cuda.is_available())" || { st INSTALL_FAILED; exit 1; }
st TOKENIZING
for split in train dev; do
  minsh=8; [ $split = dev ] && minsh=1
  python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl $DATA_DIR/$split.jsonl \
    --tar_output_pattern "tokens/$split/audios/shard-%06d.tar" \
    --jsonl_output_pattern "tokens/$split/txts/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 --shuffle True --min_num_shards $minsh --skip_errors > tokenize_$split.log 2>&1 \
    || { tail -20 tokenize_$split.log; st TOKENIZE_FAILED; exit 1; }
  ls tokens/$split/; wc -l tokens/$split/data.lst
done
st TRAINING
accelerate launch --num_processes 1 --gpu_ids 0 --mixed_precision bf16 \
  -m omnivoice.cli.train --train_config train_config.json --data_config data_config.json --output_dir exp > train.log 2>&1
rc=$?
tail -5 train.log
if [ $rc -ne 0 ]; then st TRAIN_FAILED; exit 1; fi
ls exp
st COMPLETE
