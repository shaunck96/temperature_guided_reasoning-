"""
TEMPERATURE-GUIDED REASONING (TGR) + 4-BIT QUANTIZED LLAMA + WIKIQA EXAMPLE

Key highlights:
  - Loads WikiQA from Hugging Face Datasets.
  - Converts question + correct answer to a simple "Q: ... A: ..." format.
  - Uses a 4-bit Llama model with bitsandbytes.
  - Implements TGR: a custom TemperatureMechanism + GSoT.
  - Adds a ClampLogits processor to avoid device-side asserts from extreme logits.
  - Allows do_sample=True in generation with minimal numeric issues.

Install requirements:
  - pip install transformers>=4.31 bitsandbytes>=0.38.1 datasets
  - (Optional) pip install accelerate wandb
"""

import math
import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    LogitsProcessor,
    LogitsProcessorList,
)
from torch.utils.data import Dataset

# Additional: load_dataset for WikiQA
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TGR")

# -------------------------------------------------------------------------
# 1. TGRConfig
# -------------------------------------------------------------------------
@dataclass
class TGRConfig(PretrainedConfig):
    """
    Configuration for Temperature-Guided Reasoning (TGR).
    """
    model_type: str = "temperature_guided_llama"
    base_model_name_or_path: str = "meta-llama/Llama-2-7b-hf"

    # Typical Llama-based hyperparameters
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32

    # TTM hyperparams
    temperature_heads: int = 8
    initial_temperature: float = 0.5
    temperature_regularization_weight: float = 0.01
    temperature_clamp_min: float = 0.01
    temperature_clamp_max: float = 0.99

    # GSoT
    gsoT_enabled: bool = True
    temperature_conf_threshold: float = 0.4

    max_position_embeddings: int = 2048


# -------------------------------------------------------------------------
# 2. Temperature Mechanism
# -------------------------------------------------------------------------
class TemperatureMechanism(nn.Module):
    """
    Projects hidden states -> token-level temperatures in [clamp_min, clamp_max].
    """
    def __init__(
        self,
        hidden_size: int,
        temperature_heads: int,
        clamp_min: float = 0.01,
        clamp_max: float = 0.99
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature_heads = temperature_heads
        self.proj = nn.Linear(hidden_size, temperature_heads)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        temp_logits = self.proj(hidden_states)
        temp_vals = torch.sigmoid(temp_logits)
        temp_vals = torch.clamp(temp_vals, self.clamp_min, self.clamp_max)
        return temp_vals


def temperature_guided_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    temperature: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Basic scaled dot-product attention with an average temperature factor:
      attn_probs = softmax((QK^T / sqrt(d_k)) * T_mean)
    """
    B, num_heads, seq_len, d_k = query.shape
    if temperature.size(-1) > 1:
        temp_mean = temperature.mean(dim=-1)
    else:
        temp_mean = temperature.squeeze(-1)

    temp_mean = temp_mean.unsqueeze(1).unsqueeze(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    scores = scores * temp_mean

    if attn_mask is not None:
        scores = scores + attn_mask

    attn_probs = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_probs, value)
    return context, attn_probs


# -------------------------------------------------------------------------
# 3. A custom LogitsProcessor to clamp extremes
# -------------------------------------------------------------------------
class ClampLogits(LogitsProcessor):
    """
    Clamps logits between [min_value, max_value] to avoid extremely large
    or small values that can cause NaNs or device asserts.
    """
    def __init__(self, min_value: float = -50.0, max_value: float = 50.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp(scores, self.min_value, self.max_value)


# -------------------------------------------------------------------------
# 4. TGRLlamaModel (4-bit)
# -------------------------------------------------------------------------
class TGRLlamaModel(PreTrainedModel):
    config_class = TGRConfig

    def __init__(self, config: TGRConfig):
        super().__init__(config)

        # bitsandbytes 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Force output_hidden_states
        base_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        base_config.output_hidden_states = True

        # Load Llama (4-bit)
        self.base_llama_model = LlamaForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            config=base_config,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Temperature mechanism
        self.temperature_mechanisms = nn.ModuleList([
            TemperatureMechanism(
                hidden_size=base_config.hidden_size,
                temperature_heads=config.temperature_heads,
                clamp_min=config.temperature_clamp_min,
                clamp_max=config.temperature_clamp_max
            )
        ])
        # Convert to half so it matches the final hidden states (usually half)
        self.temperature_mechanisms.half()

        # GSoT
        self.gsoT_enabled = config.gsoT_enabled
        self.temperature_conf_threshold = config.temperature_conf_threshold

        # Temperature regularization
        self.initial_temp = config.initial_temperature
        self.temp_reg_weight = config.temperature_regularization_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:

        # Forward through Llama
        outputs = self.base_llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        hidden_states = outputs.hidden_states
        final_hidden = hidden_states[-1]
        final_hidden = final_hidden.to(torch.float16)  # ensure FP16

        # Temperature
        temperature_vals = self.temperature_mechanisms[0](final_hidden)

        # Single-head attention example
        B, seq_len, hidden_size = final_hidden.shape
        q = final_hidden.unsqueeze(1)
        k = final_hidden.unsqueeze(1)
        v = final_hidden.unsqueeze(1)

        if attention_mask is not None:
            extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e4
            extended_mask = extended_mask.to(final_hidden.dtype)
        else:
            extended_mask = None

        context_layer, attn_probs = temperature_guided_attention(
            query=q,
            key=k,
            value=v,
            temperature=temperature_vals,
            attn_mask=extended_mask
        )
        new_hidden = context_layer.squeeze(1)

        # GSoT
        if self.gsoT_enabled:
            token_temp_avg = temperature_vals.mean(dim=-1)
            confident_mask = (token_temp_avg >= self.temperature_conf_threshold).float()
            confident_mask = confident_mask.unsqueeze(-1).to(new_hidden.dtype)
            new_hidden = new_hidden * confident_mask

        # Final LM head
        lm_head = self.base_llama_model.lm_head
        logits = lm_head(new_hidden)

        # Optional cross-entropy
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            main_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Temperature reg
            t_diff = (temperature_vals - self.initial_temp) ** 2
            temp_reg_loss = t_diff.mean() * self.temp_reg_weight
            loss = main_loss + temp_reg_loss

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "temperatures": temperature_vals,
            "attention_probs": attn_probs
        }

    @property
    def base_model(self):
        return self.base_llama_model


# -------------------------------------------------------------------------
# 5. WikiQA Data Preparation
# -------------------------------------------------------------------------
class WikiQADataset(Dataset):
    """
    Minimal demonstration for WikiQA: For each question, we take
    the correct answer (if multiple are correct, pick first).
    In real usage, you'd handle negative answers or do pairwise ranking.
    """
    def __init__(self, tokenizer, split="train", max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = load_dataset("wiki_qa", split=split)
        self.max_length = max_length

        self.samples = []
        for ex in self.dataset:
            # wiki_qa has: question, answer, label, ...
            # label=1 means correct, label=0 means incorrect
            # We'll only keep label=1 here for simplicity.
            if ex["label"] == 1:
                question = ex["question"]
                answer = ex["answer"]
                # Build a "Q: ... A: ..." prompt
                prompt = f"Q: {question}\nA: {answer}"
                self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_prompt = self.samples[idx]
        encoded = self.tokenizer(
            text_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        # labels = input_ids for causal LM
        encoded["labels"] = encoded["input_ids"].copy()
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
        }


# -------------------------------------------------------------------------
# 6. Main Demo with WikiQA + TGR
# -------------------------------------------------------------------------
def run_temperature_guided_reasoning_demo():
    """
    Train TGR (4-bit) on the WikiQA dataset (only correct Q-A pairs).
    Then run generation with a clamp to avoid device-side asserts.
    """
    # A. Config + Model
    config = TGRConfig(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        temperature_heads=4,
        gsoT_enabled=True,
        temperature_conf_threshold=0.4
    )

    logger.info("Loading LlamaTokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Initializing TGR model (4-bit) + wikiqa dataset...")
    model = TGRLlamaModel(config)

    # B. Prepare WikiQA dataset
    train_dataset = WikiQADataset(tokenizer, split="train", max_length=192)
    # Optionally: valid_dataset = WikiQADataset(tokenizer, split="validation")

    # C. Training Arguments
    training_args = TrainingArguments(
        output_dir="./tgr_outputs_4bit_wikiqa",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=50,
        save_steps=200,
        evaluation_strategy="no",
        fp16=False,  # avoid extra half-precision scaling
    )

    # D. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting TGR training on WikiQA (only label=1 pairs)...")
    trainer.train()
    logger.info("Training completed.")

    # E. Inference with clamp
    # We define a custom logits_processor to clamp extremes:
    clamp_processor = ClampLogits(min_value=-50, max_value=50)
    logits_processors = LogitsProcessorList([clamp_processor])

    # Build a sample question
    test_question = "When was Abraham Lincoln born?"
    prompt = f"Q: {test_question}\nA:"
    logger.info(f"Prompt: {prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Quick range check for prompt tokens
    max_id = input_ids.max().item()
    if max_id >= tokenizer.vocab_size:
        raise ValueError(
            f"Out-of-range token ID={max_id} vs. vocab_size={tokenizer.vocab_size}.\n"
            "Ensure your tokenizer matches the model."
        )

    # Generate with sampling + clamp
    logger.info("Generating with do_sample=True + clamp to avoid device asserts.")
    model.eval()
    with torch.no_grad():
        generation = model.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            logits_processor=logits_processors,  # clamp extremes
        )

    decoded = tokenizer.decode(generation[0], skip_special_tokens=True)
    logger.info(f"Inference result: {decoded}")

    print("\n===== Inference Result (4-bit, WikiQA, clamp) =====")
    print(decoded)
    print("==================================================\n")


if __name__ == "__main__":
    run_temperature_guided_reasoning_demo()
