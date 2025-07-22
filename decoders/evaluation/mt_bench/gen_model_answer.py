"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0

Note:
- The `get_model_answers` function has been modified to support answer generation using dynamic 
tokenization (dynamic BPE merges) with hypernetwork embeddings.
"""

from tokenizations.hypernet_cache import LRU_Cache
from tokenizations.tokenization_utils import DatasetEncoder
import faiss
import argparse
import json
import os
import random
import time
import scann
from tokenizers.models import BPE
import tokenizers
import copy
import pickle
import numpy as np
import torch.nn.functional as F
import sys
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from itertools import islice
from tqdm import tqdm

HOME_PATH = "/mnt/nas_home/dmf45/dynamic_tokenization"
sys.path.insert(0, HOME_PATH)


# Global - for tracking purposes
global total_old_tokens_used
global total_new_tokens_used
total_old_tokens_used = 0
total_new_tokens_used = 0


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grouper(iterable, n):
    iterable = iter(iterable)
    while True:
        batch = list(islice(iterable, n))
        if not batch:
            break
        yield batch


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    use_hn_emb,
    pre_reorder_num_neighbors,
    leaves_to_search,
    repetition_penalty,
    index_method,
    use_original_vocab_hn,
    use_lp_tokenizer,
    exhaustive_search,
    max_temperature,
    use_top_k,
    min_p,
    eos_bias,
    dynamic_bpe_merges,
):
    setup_seed(1234)
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                use_hn_emb=use_hn_emb,
                pre_reorder_num_neighbors=pre_reorder_num_neighbors,
                leaves_to_search=leaves_to_search,
                repetition_penalty=repetition_penalty,
                index_method=index_method,
                use_original_vocab_hn=use_original_vocab_hn,
                use_lp_tokenizer=use_lp_tokenizer,
                exhaustive_search=exhaustive_search,
                max_temperature=max_temperature,
                use_top_k=use_top_k,
                min_p=min_p,
                eos_bias=eos_bias,
                dynamic_bpe_merges=dynamic_bpe_merges,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def to_longest_prefix_tokenizer(tokenizer, vocab, token2id):
    print("Converting tokenizer to LP tokenizer...")
    # assumes tokenizer is byte converted
    lp_tokenizer = copy.deepcopy(tokenizer)
    unk_token = (
        lp_tokenizer.unk_token
        if lp_tokenizer.unk_token is not None
        else lp_tokenizer.eos_token
    )

    # use WordPiece without prefix to achieve longest-prefix tokenization
    lp_tokenizer._tokenizer.model = tokenizers.models.WordPiece(
        vocab, unk_token=unk_token
    )
    lp_tokenizer._tokenizer.model.continuing_subword_prefix = ""

    lp_tokenizer.eos_token_id = token2id["</s>"]
    lp_tokenizer.bos_token_id = token2id["<s>"]
    lp_tokenizer.pad_token_id = token2id["</s>"]
    lp_tokenizer.unk_token_id = token2id["<unk>"]
    lp_tokenizer.eos_token = "</s>"
    lp_tokenizer.bos_token = "<s>"
    lp_tokenizer.pad_token = tokenizer.eos_token
    lp_tokenizer.unk_token = "<unk>"

    return lp_tokenizer


def apply_ngram_penalty(
    logits,
    filtered_indices,
    current_context: list,
    id2token,
    ngram_set: set,
    n: int = 3,
    penalty: float = 1.5,
):
    if len(current_context) < n - 1:
        return logits  # Not enough tokens to form an n-gram yet

    for token_id in range(logits.size(-1)):
        token = id2token[filtered_indices[token_id]]
        potential_ngram = tuple(current_context[-(n - 1):]) + (token,)
        if potential_ngram in ngram_set:
            logits[0][token_id] = -float("inf")

    return logits


def generate_with_temperature(
    model,
    tokenizer,
    index,
    device,
    input_embeds: torch.tensor,
    embeddings: torch.tensor,
    new_vocab_tokens: set,
    temperature: float = 0.0,
    max_length: int = 1024,
    do_sample: bool = False,
    id2token={},
    tokens_history: set = set([]),
    repetition_penalty: float = 1.0,
    pre_reorder_num_neighbors: int = 1000,
    leaves_to_search: int = 2000,
    index_method: str = "scann",
    tokens: list = [],
    ngram: int = 10,
    penalty_method: str = "history",
    use_original_vocab: bool = False,
    use_top_k: bool = False,
    min_p: float = None,
    eos_bias: float = None,
    token2id: list = [],
):
    global total_old_tokens_used
    global total_new_tokens_used
    vocab_init_path = "decoders/data/tokenizer_hn_mistral/vocab.json"
    with open(vocab_init_path, "r") as file:
        vocab_init = json.load(file)
        vocab_init = set(vocab_init)
    model.eval()
    generated_tokens = []
    if temperature == 0.0:
        assert not do_sample  # We only sample if the temperature is != 0.0

    buffer_input_embeds = torch.zeros(
        (1, max_length + input_embeds.shape[1], input_embeds.shape[2]), device=device
    ).to(torch.bfloat16)
    buffer_input_embeds[:, : input_embeds.shape[1]] = input_embeds.to(torch.bfloat16)
    current_length = input_embeds.shape[1]

    if penalty_method == "ngram":
        import nltk

        ngram_set = set(list(nltk.ngrams(tokens, ngram)))

    assert len(tokens) > 0

    for token in tokens:
        if token in new_vocab_tokens:
            total_new_tokens_used += 1
        else:
            total_old_tokens_used += 1

    for _ in range(max_length):
        attention_mask = torch.ones(current_length)
        with torch.no_grad():
            outputs = model(
                inputs_embeds=buffer_input_embeds[:, :current_length],
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        if not use_original_vocab:
            last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(
                torch.float32
            )  # Shape: (batch_size, hidden_size)

            if index_method == "":
                last_hidden_state = last_hidden_state.clone().detach().cpu()
                logits = (
                    last_hidden_state
                    @ embeddings["output_embeddings"].to(torch.float32).T
                )
                filtered_indices = [
                    i for i in range(embeddings["output_embeddings"].shape[0])
                ]
                assert len(filtered_indices) == 1_000_000
            else:
                query = last_hidden_state.clone().detach().cpu()

                if index_method == "scann":
                    k = 20
                    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
                    normalised_query = query / query_norm
                    approx_indices, distances = index.search(
                        normalised_query[0],
                        leaves_to_search=leaves_to_search,
                        pre_reorder_num_neighbors=pre_reorder_num_neighbors,
                    )
                    filtered_indices = list(approx_indices)[:k]
                elif index_method == "faiss":
                    k = 20
                    _, approx_indices = index.search(query, k)
                    approx_indices = approx_indices[0]
                    filtered_indices = [idx for idx in approx_indices if idx > 0]

                logits = (
                    last_hidden_state
                    @ embeddings["output_embeddings"][
                        torch.tensor(
                            (filtered_indices),
                            device=embeddings["output_embeddings"].device,
                        ).to(torch.long)
                    ]
                    .to(torch.float32)
                    .to(device)
                    .T
                )
            eos_token_idx = token2id["</s>"]
            if (
                eos_bias != 0.0
                and eos_bias is not None
                and eos_token_idx in filtered_indices
            ):
                eos_position = filtered_indices.index(eos_token_idx)
                logits[:, eos_position] += eos_bias
                print("Applying EOS token bias")
        else:
            logits = outputs.logits[0][
                -1
            ]  # IN/OUT Embeddings have already been set if using a small 32k vocab
            logits[tokenizer.eos_token_id] += eos_bias

        # Apply repetition penalty
        if penalty_method == "ngram":
            logits = apply_ngram_penalty(
                logits=logits,
                filtered_indices=filtered_indices,
                id2token=id2token,
                ngram_set=ngram_set,
                current_context=tokens,
                n=ngram,
            )
        elif penalty_method == "history":
            logits = torch.where(
                logits < 0, logits * repetition_penalty, logits / repetition_penalty
            )

        # Sample from categorical if temperature != 0, else choose the token with the highest probability
        if temperature != 0.0:
            logits = logits / temperature
        probabilities = F.softmax(logits, dim=-1)

        # Sample next token id
        if do_sample:
            if use_top_k and min_p is None:  # and not use_min_p:
                try:
                    topk_probs, topk_indices = torch.topk(probabilities, 10, dim=-1)
                    next_token_id = topk_indices[0][
                        torch.multinomial(topk_probs, num_samples=1).item()
                    ].item()
                except:
                    next_token_id = topk_indices[
                        torch.multinomial(topk_probs, num_samples=1).item()
                    ].item()
            elif use_top_k and min_p is not None:
                topk_probs, topk_indices = torch.topk(probabilities, 10, dim=-1)
                top_probs = topk_probs.max(dim=-1, keepdim=True).values
                scaled_min_p = min_p * top_probs
                valid_mask = topk_probs >= scaled_min_p
                valid_probs = topk_probs[valid_mask]
                valid_indices = topk_indices[valid_mask][:10]
                if len(valid_indices) > 0:
                    next_token_id = valid_indices[
                        torch.multinomial(valid_probs, num_samples=1).item()
                    ].item()
                else:
                    print("No valid index with p > scaled_min_p")
                    next_token_id = topk_indices[0][0].item()
            else:
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()
        else:
            next_token_id = torch.argmax(probabilities, dim=-1).item()

        tokens_history.add(next_token_id)

        # Convert ID to token
        if not use_original_vocab:
            next_token = id2token[filtered_indices[next_token_id]]
        else:
            next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]

        if next_token in new_vocab_tokens:
            total_new_tokens_used += 1
        else:
            total_old_tokens_used += 1

        if (
            next_token == "</s>"
            or "".join(generated_tokens[-2:]) == "</s>"
            or "".join(generated_tokens[-3:]) == "</s>"
            or "".join(generated_tokens[-4:]) == "</s>"
        ):
            break

        generated_tokens.append(next_token)
        tokens.append(next_token)

        if penalty_method == "ngram":
            ngram_set.add(tuple(tokens[-(ngram):]))

        # Prepare the next input embeddings
        if not use_original_vocab:
            next_input_embeds = (
                embeddings["input_embeddings"][filtered_indices[next_token_id]]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
                .to(torch.bfloat16)
            )
        else:
            next_input_embeds = (
                model.get_input_embeddings()
                .weight.data[next_token_id]
                .to(torch.bfloat16)
            )

        buffer_input_embeds[:, current_length] = next_input_embeds
        current_length += 1

    return tokenizer.decoder.decode(generated_tokens)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    use_hn_emb,
    pre_reorder_num_neighbors,
    leaves_to_search,
    repetition_penalty,
    index_method,
    use_original_vocab_hn,
    use_lp_tokenizer,
    exhaustive_search,
    max_temperature,
    use_top_k,
    min_p,
    eos_bias,
    dynamic_bpe_merges,
):
    global total_old_tokens_used
    global total_new_tokens_used

    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    vocab_init_path = "decoders/data/tokenizer_hn_mistral/vocab.json"
    vocab_new_path = "decoders/data/large_tokenizer/vocab.json"
    with open(vocab_init_path, "r") as file:
        vocab_init = json.load(file)
    with open(vocab_new_path, "r") as file:
        vocab_new = json.load(file)
    new_tokens = set(vocab_new) - set(vocab_init)
    assert len(new_tokens) == len(vocab_new) - len(vocab_init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token2id = {}  # not defined yet
    if use_hn_emb:  # Use HN embeddings
        print("Load id2token map...")
        with open(
            "decoders/data/1M_vocab_embeddings/token2id.pkl", "rb"
        ) as pickle_file:
            token2id = pickle.load(pickle_file)
            id2token = {idx: token for token, idx in token2id.items()}

        print("Load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
        )

        print("Load 1M IN/OUT embeddings...")
        embeddings = torch.load(
            "decoders/data/1M_vocab_embeddings/large_HN_embeddings.pt"
        )
        assert len(embeddings["input_embeddings"]) == 1_000_000
        assert len(embeddings["output_embeddings"]) == 1_000_000

        if not use_original_vocab_hn:
            if index_method == "scann":
                print("Loading Scann index...")
                INDEX_DIR = "decoders/data/scann_index/scann_index_6_neighbours_200"
                index = scann.scann_ops_pybind.load_searcher(INDEX_DIR)
            else:
                print("Loading Faiss index...")
                index = faiss.read_index(
                    "decoders/data/index/approximate_index_ivfflat_50_clusters.faiss"
                )
                index.nprobe = 20

            print("Updating tokenizer vocab to 1M...")
            vocab_new_path = "decoders/data/large_tokenizer/vocab.json"
            merges_path = "decoders/data/large_tokenizer/merges.txt"
            vocab, _ = BPE.read_file(vocab_new_path, merges_path)
            tokenizer = to_longest_prefix_tokenizer(
                tokenizer=tokenizer, vocab=vocab, token2id=token2id
            )
            assert len(tokenizer.get_vocab()) == 1_000_000
        else:
            if use_lp_tokenizer:
                vocab = tokenizer.get_vocab()
                tokenizer = to_longest_prefix_tokenizer(
                    tokenizer=tokenizer, vocab=vocab, token2id=token2id
                )
            print("Setting input/output embeddings")
            vocab_tokens = [-1] * tokenizer.vocab_size
            vocab = tokenizer.get_vocab()
            for token in vocab:
                vocab_tokens[vocab[token]] = token
            indices = [token2id[token] for token in vocab_tokens]

            input_embeddings_data = (
                embeddings["input_embeddings"][indices].to(torch.bfloat16).to(device)
            )
            input_embeddings = torch.nn.Embedding(
                num_embeddings=input_embeddings_data.shape[0],
                embedding_dim=input_embeddings_data.shape[1],
                _weight=input_embeddings_data,
            )
            input_embeddings.weight.requires_grad = False
            model.set_input_embeddings(input_embeddings)

            output_embeddings_data = (
                embeddings["output_embeddings"][indices].to(torch.bfloat16).to(device)
            )
            output_embeddings = torch.nn.Linear(
                in_features=output_embeddings_data.shape[1],
                out_features=output_embeddings_data.shape[0],
                bias=False,
            )
            output_embeddings.weight = torch.nn.parameter.Parameter(
                data=output_embeddings_data
            )
            output_embeddings.weight.requires_grad = False
            output_embeddings.bias = None
            model.set_output_embeddings(output_embeddings)

    if dynamic_bpe_merges is not None:
        print("Using dynamic bpe!")
        hypernet = AutoModel.from_pretrained(
            "benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
        )
        langs = [x.strip() for x in open("artifacts/26l.txt")]
        lang_index = torch.tensor(langs.index("en"), dtype=torch.int32).to(device)
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        source_embeddings = torch.concatenate(
            [
                base_model.get_input_embeddings().weight.data,
                base_model.get_output_embeddings().weight.data,
            ],
            axis=1,
        ).to(device)
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
            exp_type="dynamic_bpe",
            collect_extra_data=True,
            bpe_tokenizer_boundary="pretokens",
        )

    seq_lens = []

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        temperature = min(temperature, max_temperature)

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                if dynamic_bpe_merges is not None:
                    print(
                        f"Using Dynamic Tokenization with {dynamic_bpe_merges} merges."
                    )
                    inputs = datasetEncoder.encode_examples_unique_tokens_lru(
                        examples=[prompt],
                        task="mmlu",
                        max_length=40000,
                        merges=dynamic_bpe_merges,
                    )  # uses same tokenization as for MMLU
                    input_embeds = inputs["inputs_embeds"].to(torch.bfloat16)
                    for ids_sample in inputs["inputs_embeds"]:
                        seq_lens.append(len(ids_sample))
                    tokens = inputs["batch_tokens"][
                        0
                    ]  # Currently only support batch of size 1
                    tokens_history = set(tokens)
                elif not use_hn_emb or (use_hn_emb and use_original_vocab_hn):
                    print("Using 1M Vocab")
                    input_ids = tokenizer([prompt]).input_ids
                    input_embeds = (
                        model.get_input_embeddings()
                        .weight.data[input_ids[0]]
                        .unsqueeze(0)
                        .to(device)
                    )
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                    tokens_history = set(input_ids[0])
                    seq_lens.append(len(tokens))
                else:  # Using 1M Vocab
                    tokens = ["<s>"] + tokenizer.tokenize(prompt)
                    input_ids = [token2id[token] for token in tokens]
                    tokens_history = set(input_ids)
                    input_embeds = (
                        embeddings["input_embeddings"][input_ids]
                        .unsqueeze(0)
                        .to(device)
                    )
                    seq_lens.append(len(tokens))

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                if use_hn_emb and not use_original_vocab_hn:
                    if exhaustive_search:
                        index_method = ""
                        index = None
                    output = generate_with_temperature(
                        model=model,
                        tokenizer=tokenizer,
                        index=index,
                        device=device,
                        embeddings=embeddings,
                        input_embeds=input_embeds,
                        temperature=temperature,
                        max_length=max_new_token,
                        do_sample=do_sample,
                        id2token=id2token,
                        tokens_history=tokens_history,
                        pre_reorder_num_neighbors=pre_reorder_num_neighbors,
                        leaves_to_search=leaves_to_search,
                        repetition_penalty=repetition_penalty,
                        index_method=index_method,
                        tokens=tokens,
                        use_original_vocab=False,
                        new_vocab_tokens=new_tokens,
                        use_top_k=use_top_k,
                        min_p=min_p,
                        eos_bias=eos_bias,
                        token2id=token2id,
                    )
                else:
                    output = generate_with_temperature(
                        model=model,
                        tokenizer=tokenizer,
                        index=None,
                        device=device,
                        embeddings=None,
                        input_embeds=input_embeds,
                        temperature=temperature,
                        max_length=max_new_token,
                        do_sample=do_sample,
                        id2token=None,
                        tokens_history=tokens_history,
                        pre_reorder_num_neighbors=None,
                        leaves_to_search=None,
                        repetition_penalty=repetition_penalty,
                        index_method="",
                        tokens=tokens,
                        use_original_vocab=True,
                        new_vocab_tokens=new_tokens,
                        use_top_k=use_top_k,
                        min_p=min_p,
                        eos_bias=eos_bias,
                        token2id=token2id,
                    )

                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                conv.update_last_message(output)
                turns.append(output)
                turns.append("")
                break

            choices.append({"index": i, "turns": turns})
            print(
                f"Old_tokens: {total_old_tokens_used} | New_tokens: {total_new_tokens_used}"
            )

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "avg_len": sum(seq_lens) / len(seq_lens),
                "merges": dynamic_bpe_merges,
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="bfloat16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--use_hn_emb",
        action="store_true",
        help="Use custom generation.",
    )
    parser.add_argument("--pre_reorder_num_neighbors", type=int, default=2000)
    parser.add_argument("--leaves_to_search", type=int, default=1000)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--index_method", type=str, default="scann")
    parser.add_argument(
        "--use_original_vocab_hn",
        action="store_true",
        help="Use original vocab (32k) with HN embeddings",
    )
    parser.add_argument("--use_lp_tokenizer", action="store_true")
    parser.add_argument("--exhaustive_search", action="store_true")
    parser.add_argument("--max_temperature", type=float, default=1.0)
    parser.add_argument("--use_top_k", action="store_true")
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--eos_bias", type=float, default=0.0)
    parser.add_argument(
        "--dynamic_bpe_merges",
        type=int,
        default=None,
        help="Use dynamic tokenization with HN embeddings. Check results with different % seq. reduction",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = (
        f"FastChat/fastchat/llm_judge/data/{args.bench_name}/mt_bench_inf.jsonl"
    )
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"FastChat/fastchat/llm_judge/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        use_hn_emb=args.use_hn_emb,
        pre_reorder_num_neighbors=args.pre_reorder_num_neighbors,
        leaves_to_search=args.leaves_to_search,
        repetition_penalty=args.repetition_penalty,
        index_method=args.index_method,
        use_original_vocab_hn=args.use_original_vocab_hn,
        use_lp_tokenizer=args.use_lp_tokenizer,
        exhaustive_search=args.exhaustive_search,
        max_temperature=args.max_temperature,
        use_top_k=args.use_top_k,
        min_p=args.min_p,
        eos_bias=args.eos_bias,
        dynamic_bpe_merges=args.dynamic_bpe_merges,
    )

    reorg_answer_file(answer_file)
