#!/bin/bash
#SBATCH -J dynamic-tk-eval-xnli-whitespace
#SBATCH --time=10:0:0
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ltl-gpu04

JOBID=$SLURM_JOB_ID
EXP_NAME="dynamic_tokenization_xnli_EN_original_tokenization"

source /mnt/nas_home/dmf45/miniconda3/bin/activate base

exp_types=("plain" "fvt" "original_tk_hypernet" "word_tk_hypernet")

# plain - using original embeddings and tokenization
# fvt - using fvt embeddings and word-level tokenization
# original_tk_hypernet - using hypernetwork embeddings and original tokenization
# word_tk_hypernet - using hypernetwork embeddings and word-level tokenization

languages=('ar' 'bg' 'de' 'el' 'en' 'es' 'fr' 'hi' 'ru' 'sw' 'tr' 'ur' 'vi')

for lng in "${languages[@]}"; do
    for exp_type in "${exp_types[@]}"; do
        python3 encoders/evaluation/xnli/xnli_evaluation.py --exp_type $exp_type --lng $lng --batch_size 32 --adapter_path output_lora_peft/xnli_en_lora_alpha_64_drop_0.3_rank_32_seed_42 --peft --exp_prefix "BEST_LORA_ALL_LNG" > logs/xnli_$today_date/out.xnli_${today_date}.${lng}.${exp_type}.${JOBID}
    done
done
