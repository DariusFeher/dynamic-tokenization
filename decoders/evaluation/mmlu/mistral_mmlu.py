"""
MMLU Evaluation Script for Mistral

This script evaluates Mistral or Hypernetwork-augmented Mistral models on the MMLU benchmark.
It supports different tokenization and embedding experiments, including dynamic, original, 
longest-prefix tokenization with original or hypernetwork embeddings as well as the original
or the expanded 1M vocabulary.

See argparse section for all argument options.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import argparse
import sys
from torch.utils.data import DataLoader
import wandb
from evaluation_utils import MMLUDataset, collate_fn, setup_seed, evaluate_model
from tokenizers.models import BPE, WordPiece

HOME_PATH = "/mnt/nas_home/dmf45/dynamic_tokenization"
sys.path.insert(0, HOME_PATH)
from tokenizations.static_tokenizations import tokenize
from tokenizations.tokenization_utils import DatasetEncoder
from tokenizations.hypernet_cache import LRU_Cache

def main():
    parser = argparse.ArgumentParser(description="Running MMLU Evaluation")
    parser.add_argument("--ds_subject", type=str, default="all", help="The MMLU subject subset to use for evaluation.")
    parser.add_argument("--exp_type", type=str, default="plain", help="Choose which type of experiment to use: plain (original tokenization), original_tk_hypernet (HN embeddings), lp_tk_hypernet (longest prefix tokenization), dynamic_bpe (HN embeddings with different number of merges)")
    parser.add_argument("--verbose", type=bool, default=False, help="Add extra loggings.")
    parser.add_argument("--eval_type", type=str, default="origianl", help="Original (compare with probs of A, B, C or D) or Harness (compare with probs of each choice text)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use during evaluation")
    parser.add_argument("--five_shot", action="store_true", help="Perform 5-shots evaluation")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log data to wandb.")
    parser.add_argument("--exp_prefix", type=str, default="", help="Prefix to be added for the exp name in wandb")
    parser.add_argument("--same_domain_shot", action="store_true", help="Choose 5 shots from the same domain (but different split)")
    parser.add_argument("--lng", type=str, default="en", help="Language to use for hypernetwork")
    parser.add_argument("--max_len", type=int, default=4096, help="Max length to be used during tokenization")
    parser.add_argument("--vocab_1M", action="store_true", help="Use tokenizer with 1M vocab and HN embeddings")
    parser.add_argument("--multiple_merges_exp", action="store_true", help="Use HN embeddings with different perecentages of sequence reduction")
    parser.add_argument("--use_original_emb_for_choices", action="store_true", help="Use original embeddings for A, B, C, D choices")

    args = parser.parse_args()
    setup_seed(1234)

    if not args.no_wandb:
        exp_prefix = args.exp_prefix + "_" if args.exp_prefix else ""
        wandb.init(
            project="dynamic-tokenization",
            config={"dataset": "MMLU", "exp_type": args.exp_type, "eval_type": args.eval_type},
            name=f"{exp_prefix}MMLU_Mistral_{args.exp_type}_{args.eval_type}_five_shot_{args.five_shot}_subject_{args.ds_subject}_batch_size_{args.batch_size}_1M_Vocab_{args.vocab_1M}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    # Setup for hypernet/other exp types
    hypernet = None
    base_model = None
    lang_index = None
    source_embeddings = None
    datasetEncoder = None
    inout_1M_embeddings = None
    if args.exp_type in ["original_tk_hypernet", "lp_tk_hypernet", "dynamic_bpe", "word_tk_hypere"]:
        hypernet = AutoModel.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1")
        if args.exp_type == "lp_tk_hypernet":
            if args.vocab_1M:
                inout_1M_embeddings = torch.load("decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt")
                vocab_new_path = "decoders/data/large_tokenizer/vocab.json"
                merges_path = "decoders/data/large_tokenizer/merges.txt"
                vocab, merges = BPE.read_file(vocab_new_path, merges_path)
            else:
                vocab = tokenizer.get_vocab()
            unk_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
            tokenizer._tokenizer.model = WordPiece(vocab, unk_token=unk_token)
            tokenizer._tokenizer.model.continuing_subword_prefix = ""
        langs = [x.strip() for x in open("artifacts/26l.txt")]
        lang_index = torch.tensor(langs.index(args.lng), dtype=torch.int32).to(device)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        source_embeddings = torch.concatenate([
            base_model.get_input_embeddings().weight.data,
            base_model.get_output_embeddings().weight.data,
        ], axis=1).to(device)
        embeddings_cache = LRU_Cache(
            cache_size=5000,
            emb_size=base_model.get_input_embeddings().weight.data.shape[1],
            device=device,
        )
        datasetEncoder = DatasetEncoder(
            hypernet=hypernet,
            tokenizer=tokenizer,
            device=device,
            lang_index=lang_index,
            surface_form_maxlen=7,
            source_embeddings=source_embeddings,
            embeddings_cache=embeddings_cache,
            exp_type=args.exp_type,
            collect_extra_data=True,
            bpe_tokenizer_boundary="pretokens",
        )

    subjects = [
        "abstract_algebra", "high_school_government_and_politics", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
    ]

    dataset = load_dataset("hails/mmlu_no_train", args.ds_subject)["test"]
    validation_dataset = load_dataset("hails/mmlu_no_train", args.ds_subject)["validation"]
    per_subject_validation_datasets = {}
    if args.same_domain_shot:
        max_length = max(8192, args.max_len)
        for subject_name in subjects:
            try:
                print(f"Downloading {subject_name} data")
                per_subject_validation_datasets[subject_name] = load_dataset(
                    "cais/mmlu", subject_name, cache_dir="~/.cache/huggingface/datasets"
                )["validation"]
            except:
                raise Exception(f"Error when downloading dataset for subject {subject_name}")

    mmlu_dataset = MMLUDataset(
        dataset,
        validation_dataset=validation_dataset,
        validation_datasets=per_subject_validation_datasets,
    )
    mmlu_dataset.five_shot = args.five_shot
    mmlu_dataset.same_domain_shot = args.same_domain_shot
    dataloader = DataLoader(mmlu_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    evaluate_model(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        args=args,
        base_model=base_model,
        hypernet=hypernet,
        lang_index=lang_index,
        source_embeddings=source_embeddings,
        datasetEncoder=datasetEncoder,
        inout_1M_embeddings=inout_1M_embeddings,
        subjects=subjects,
    )

if __name__ == "__main__":
    main()
