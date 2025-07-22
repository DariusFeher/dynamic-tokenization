---

# Retrofitting Large Language Models with Dynamic Tokenization

This repository contains the code for the paper **[Retrofitting Language Models with Dynamic Tokenization](https://arxiv.org/pdf/2411.18553)**. Dynamic tokenization adaptively adjusts token boundaries based on the input data, via a subword-merging algorithm based on byte-pair encoding, and generates embeddings on-the-fly, improving inference efficiency and multilingual fairness with minimal performance loss. This approach is especially useful for multilingual settings, where a fixed tokenization scheme results in over-segmentation.

## Functionality

- **Dynamic BPE**: Implements dynamic tokenization using a byte-pair encoding (BPE) inspired algorithm that operates on subwords and can adjust the number of merges per batch or per sample.
- **Hypernetwork Embeddings**: Supports dynamic generation of token embeddings using a hypernetwork, allowing for out-of-vocabulary and on-the-fly token embedding generation.
- **LRU Caching**: Uses an LRU cache to store and reuse token embeddings for efficiency (i.e., this is especially useful for common tokens such as "the", "and", "or" etc. - see Appendix D)
- **Task Support**: Ready-to-use for NLI, NER, and MMLU tasks with Mistral-7B, with easy extension to others.

*Requirements:* a pre-trained hypernetwork. See [bminixhofer/zett](https://github.com/bminixhofer/zett#egg=zett&subdirectory=../../zett) for a list of available hypernetworks.


## Minimal Working Example: Encode a Batch with Dynamic Tokenization (Mistral-7B)

Below is a minimal example for encoding a batch of NLI data using dynamic tokenization and hypernetwork embeddings.

```python
from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from zett.utils import get_surface_form_matrix

# LOAD MODELS AND TOKENIZERS
# Note: this requires access to Mistral-7B model. You must be logged in to huggingface via cli.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
)
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Mistral-7B-v0.1",
    trust_remote_code=True
).to(device)

hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Mistral-7B-v0.1"
)

# DATA
examples = [
    {"text": "Ribonucleoprotein particles regulate mRNA stability."},
    {"text": "mRNA stability is modulated by ribonucleoprotein factors."},
]

# TOKENIZE: From 36 tokens to 20 tokens with 10 merges

# INITIAL SUBWORD TOKENIZATION
base_tokenizer.pad_token = base_tokenizer.eos_token
encoding = base_tokenizer(
    [example['text'] for example in examples], add_special_tokens=True, return_tensors="pt", padding=True
)
initial_tokens = [
    base_tokenizer.convert_ids_to_tokens(ids.tolist())
    for ids in encoding.input_ids
]
nr_tokens = sum([len(tokens) for tokens in initial_tokens])
print("Initial tokens:",  initial_tokens)
# Initial tokens: [['</s>', '<s>', '▁R', 'ib', 'on', 'uc', 'le', 'op', 'rote', 'in',
#  '▁particles', '▁reg', 'ulate', '▁m', 'R', 'NA', '▁stability', '.'], ['<s>', '▁m', 
#  'R', 'NA', '▁stability', '▁is', '▁mod', 'ulated', '▁by', '▁rib', 'on', 'uc', 'le',
#  'op', 'rote', 'in', '▁factors', '.']]
print(f"Number of tokens: {nr_tokens}")
# Number of tokens: 36

# DYNAMIC TOKENIZATION with 10 merges (i.e., merge top-10 most frequent adjacent subword pairs)
dynamic_bpe = Dynamic_BPE(tokenizer=hypernet_tokenizer, tokenizer_boundary='pretokens')
dynamic_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
    batch_examples=examples,
    max_nr_merges=10,
    mlm=True
)
nr_dynamic_tokens = sum([len(tokens) for tokens in dynamic_tokens])
print("Dynamic tokens:", dynamic_tokens)
# Dynamic tokens: [['<s>', 'ĠRibonucleoprotein', 'Ġparticles', 'Ġregulate', 'ĠmRNA',
#  'Ġstability', '.', '</s>'], ['<s>', 'ĠmRNA', 'Ġstability', 'Ġis', 'Ġmod', 'ulated',
#  'Ġby', 'Ġrib', 'onucleoprotein', 'Ġfactors', '.', '</s>']]
print(f"Number of tokens: {nr_dynamic_tokens}")
# Number of tokens: 20


# OBTAINING HYPERNETWORK EMBEDDINGS
src_emb = torch.cat([
    base_model.get_input_embeddings().weight.data,
    base_model.get_output_embeddings().weight.data,
], dim=1).to(device)

surfaces = get_surface_form_matrix(
    dynamic_tokens,
    maxlen=hypernet.config.hn_surface_maxlen,
    tokenizer_to_use=hypernet_tokenizer
)[0]

# Predict embeddings
pred_in, pred_out, _ = hypernet(
    torch.from_numpy(surfaces).to(device),
    source_embeddings=src_emb
)

print("Predicted in-emb shape:", pred_in.shape)
print("Predicted out-emb shape:", pred_out.shape)
```

## Reference

If you use this code, please cite:

```
@article{feher2024retrofitting,
  title={Retrofitting Large Language Models with Dynamic Tokenization},
  author={Feher, Darius and Vuli{\'c}, Ivan and Minixhofer, Benjamin},
  journal={arXiv preprint arXiv:2411.18553},
  year={2024}
}
```


