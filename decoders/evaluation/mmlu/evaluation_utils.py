"""
Utility functions and classes for MMLU evaluation.

Includes:
- MMLUDataset: PyTorch Dataset for MMLU with 0-shot and 5-shot prompt formatting
- collate_fn: DataLoader collate function
- setup_seed: Seed setup for reproducibility
- Generate token embeddings on-the-fly. This is used to embed tokens during evaluation (i.e., inference time).
"""

import torch
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from zett.utils import get_surface_form_matrix


def get_hn_embeddings_for_tokens(
    tokens: List[str],
    tokenizer,
    lang_index: int,
    hypernet,
    source_embeddings: torch.Tensor,
    device: torch.device,
    base_input_embeddings: torch.Tensor,
    base_output_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hypernetwork embeddings for a list of tokens.
    
    This function takes a list of tokens and generates their corresponding
    hypernetwork embeddings. Special tokens are handled by using the base
    model's embeddings directly, while other tokens use the hypernetwork
    predictions. The resulting embeddings are converted to bfloat16 for
    memory efficiency.
    
    Args:
        tokens: List of token strings to generate embeddings for
        tokenizer: Hypernetwork tokenizer for surface form generation
        lang_index: Language index tensor for the hypernetwork
        hypernet: Hypernetwork model for embedding prediction
        source_embeddings: Source embeddings for the hypernetwork
        device: Target device for tensor operations
        base_input_embeddings: Base model input embeddings for special tokens
        base_output_embeddings: Base model output embeddings for special tokens
        
    Returns:
        Tuple containing:
        - predicted_input_embeddings: Generated input embeddings (bfloat16)
        - predicted_output_embeddings: Generated output embeddings (bfloat16)
    """
    with torch.no_grad():
        target_surface_forms = get_surface_form_matrix(
            tokens,  # byte representation of the tokens to predict
            maxlen=hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=tokenizer,
        )[0]
        target_surface_forms = torch.from_numpy(target_surface_forms).to(device)
        
        special_tokens_mask = torch.isin(
            target_surface_forms[:, 0],
            torch.tensor(tokenizer.all_special_ids, device=device),
        )

        predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
            target_surface_forms,
            lang_index=lang_index,
            source_embeddings=source_embeddings,
        )

        # Replace special token embeddings with base model embeddings
        predicted_input_embeddings[special_tokens_mask] = base_input_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]
        predicted_output_embeddings[special_tokens_mask] = base_output_embeddings[
            target_surface_forms[special_tokens_mask, 0]
        ]

        return (
            predicted_input_embeddings.to(torch.bfloat16),
            predicted_output_embeddings.to(torch.bfloat16)
        )

class MMLUDataset(Dataset):
    def __init__(self, dataset, validation_dataset, validation_datasets, num_shots=5):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.num_shots = num_shots
        self.validation_datasets = validation_datasets

    def __len__(self):
        return len(self.dataset)

    def format_prompt(
        self,
        question,
        choices,
        subject: str = "",
        is_context_question: bool = False,
        same_domain_shot: bool = True,
        answer: str = "",
        five_shot: bool = False,
    ):
        subject = subject.replace("_", " ")
        if is_context_question:
            assert answer != ""
            if same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
            else:  # random domain shots
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}\n\n"
        else:  # if main prompt question
            if five_shot and same_domain_shot:
                return f"This question refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            elif five_shot and not same_domain_shot:
                return f"This question is about {subject} and refers to the following information.\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        choices = item["choices"]
        correct_answer_index = item["answer"]
        subject = item["subject"]
        context = ""
        five_shot = getattr(self, 'five_shot', False)
        same_domain_shot = getattr(self, 'same_domain_shot', True)
        if five_shot:
            for _ in range(self.num_shots):
                if not same_domain_shot:
                    example = random.choice(self.validation_dataset)
                else:
                    example = random.choice(self.validation_datasets[subject])
                while example["question"] == question and set(
                    example["choices"]
                ) == set(choices):
                    if not same_domain_shot:
                        example = random.choice(self.validation_dataset)
                    else:
                        example = random.choice(self.validation_datasets[subject])

                if example["question"] == question and set(example["choices"]) == set(
                    choices
                ):
                    raise Exception(
                        "Context question should be different than prompt question. Please check!"
                    )

                example_question = example["question"]
                example_choices = example["choices"]
                example_answer_index = example["answer"]
                example_answer = chr(65 + example_answer_index)
                if same_domain_shot:
                    assert example["subject"] == subject
                example_prompt = self.format_prompt(
                    question=example_question,
                    choices=example_choices,
                    is_context_question=True,
                    answer=example_answer,
                    same_domain_shot=same_domain_shot,
                    subject=example["subject"],
                )

                context += example_prompt

        prompt = context + self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        if (five_shot and same_domain_shot) or (not five_shot):
            subject = subject.replace("_", " ")
            prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{prompt}"
        elif (five_shot and not same_domain_shot):
            prompt = f"The following are multiple choice questions (with answers).\n\n{prompt}"
        init_prompt = self.format_prompt(
            question=question,
            choices=choices,
            subject=subject,
            five_shot=five_shot,
            same_domain_shot=same_domain_shot,
        )
        return prompt, choices, correct_answer_index, context, init_prompt, subject

def collate_fn(batch):
    prompts = [item[0] for item in batch]
    choices = [item[1] for item in batch]
    correct_answer_indices = [item[2] for item in batch]
    contexts = [item[3] for item in batch]
    init_prompts = [item[4] for item in batch]
    subjects = [item[5] for item in batch]
    return prompts, choices, correct_answer_indices, contexts, init_prompts, subjects

def setup_seed(seed):
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 