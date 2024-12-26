---

# **Temperature-Guided Reasoning (TGR) + 4-bit Quantized Llama on WikiQA**

This document provides a comprehensive explanation of the **Temperature-Guided Reasoning** (TGR) script that integrates:

1. **4-bit quantized Llama** (using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)),  
2. **WikiQA** dataset from Hugging Face Datasets,  
3. **Temperature Mechanism** and **Guided Sequence of Thought** (GSoT),  
4. A custom **ClampLogits** processor to clamp extreme logits and mitigate potential CUDA device-side asserts during generation.

> **Reference**:  
> *GUIDANCE IS ALL YOU NEED: TEMPERATURE-GUIDED REASONING IN LARGE LANGUAGE MODELS*  
> *Eyad Gomaa, PhD. Gomaa Salah, AI Researcher, SILX AI, eyad@sicopilot.cloud (2024)*  
> This paper introduces *Quasar-1*, a novel temperature-guided reasoning architecture with Token Temperature Mechanism (TTM) and Guided Sequence of Thought (GSoT). The approach demonstrates superior logical reasoning capabilities and convergence guarantees, providing a more efficient alternative to standard chain-of-thought methods.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Script Structure](#script-structure)  
   1. [Imports & Logging](#imports--logging)  
   2. [TGRConfig](#tgrconfig)  
   3. [Temperature Mechanism & Attention](#temperature-mechanism--attention)  
   4. [ClampLogits Processor](#clamplogits-processor)  
   5. [TGRLlamaModel](#tgrllamamodel)  
   6. [WikiQADataset](#wikiqadataset)  
   7. [Main Training & Inference Demo](#main-training--inference-demo)  
3. [How to Run](#how-to-run)  
4. [Potential Caveats & Next Steps](#potential-caveats--next-steps)  

---

## **Overview**

- **Objective**  
  Demonstrate how to train a Llama-based model in a **4-bit quantized** setting with **Temperature-Guided Reasoning** (TGR) and the **WikiQA** dataset.

- **Core Concepts**  
  1. **Token Temperature Mechanism (TTM)**: Projects hidden states into a per-token temperature \([0, 1]\).  
  2. **Guided Sequence of Thought (GSoT)**: Zero-out or prune tokens below a certain temperature threshold, focusing the model’s attention on critical tokens.  
  3. **Temperature Regularization**: Penalizes extreme temperature values that deviate from the desired initial average.

- **WikiQA**  
  This script uses the WikiQA dataset (a question-answer corpus). We filter for label=1 (correct answers) and create a minimal prompt:  
  \[
  \text{"Q: <question>\nA: <correct_answer>"}
  \]  
  Then we train in a causal language modeling style.

- **ClampLogits**  
  A custom generation-time processor that **clamps** logits to a safe range (e.g., \([-50, 50]\)) to avoid numerical instabilities (NaNs, infinite exponents) which can trigger CUDA device-side asserts when sampling tokens.

---

## **Script Structure**

Below is the **complete** script with inline commentary.

### **Imports & Logging**

```python
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

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TGR")
```

**Key Points**  
- `transformers` provides Llama model, trainer, etc.  
- `bitsandbytes` config for 4-bit quantization.  
- `datasets` for the WikiQA dataset.  
- `torch` for building custom layers and attention modules.

---

### **TGRConfig**

```python
@dataclass
class TGRConfig(PretrainedConfig):
    """
    Configuration for Temperature-Guided Reasoning (TGR).
    """
    model_type: str = "temperature_guided_llama"
    base_model_name_or_path: str = "meta-llama/Llama-2-7b-hf"

    # Llama-based hyperparameters
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
```

**Purpose**  
Stores hyperparameters related to TGR:

- **Temperature Mechanism** (number of heads, clamp min/max)  
- **GSoT** (threshold)  
- **Llama** defaults (hidden size, intermediate, etc.)  

---

### **Temperature Mechanism & Attention**

```python
class TemperatureMechanism(nn.Module):
    """
    Projects hidden states -> per-token temperatures in [clamp_min, clamp_max].
    """
    def __init__(self, hidden_size: int, temperature_heads: int,
                 clamp_min: float = 0.01, clamp_max: float = 0.99):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature_heads = temperature_heads
        self.proj = nn.Linear(hidden_size, temperature_heads)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Linear -> Sigmoid -> clamp
        temp_logits = self.proj(hidden_states)  # [B, seq_len, T_heads]
        temp_vals = torch.sigmoid(temp_logits)  # [0,1]
        temp_vals = torch.clamp(temp_vals, self.clamp_min, self.clamp_max)
        return temp_vals
```

- **Logic**  
  1. A linear projection from hidden_size -> temperature_heads.  
  2. Apply `sigmoid` to keep values in \([0,1]\).  
  3. `clamp` to avoid exact 0 or 1 extremes.

```python
def temperature_guided_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    temperature: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simplified multi-head attention with average temperature factor.
    """
    B, num_heads, seq_len, d_k = query.shape
    if temperature.size(-1) > 1:
        temp_mean = temperature.mean(dim=-1)  # [B, seq_len]
    else:
        temp_mean = temperature.squeeze(-1)   # [B, seq_len]

    temp_mean = temp_mean.unsqueeze(1).unsqueeze(-1)  # [B, 1, seq_len, 1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores * temp_mean

    if attn_mask is not None:
        scores += attn_mask

    attn_probs = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_probs, value)
    return context, attn_probs
```

- **Key**  
  1. Compute \(QK^T / \sqrt{d_k}\) as usual.  
  2. Multiply by `temp_mean` to modulate how strongly each token contributes.  
  3. Apply `softmax`, then fetch context with `attn_probs @ value`.

---

### **ClampLogits Processor**

```python
class ClampLogits(LogitsProcessor):
    """
    Clamps logits between [min_value, max_value] to avoid numeric blow-ups.
    """
    def __init__(self, min_value: float = -50.0, max_value: float = 50.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, input_ids, scores):
        return torch.clamp(scores, self.min_value, self.max_value)
```

- **Purpose**  
  During generation, the model computes new token logits. This processor ensures logits remain in a safe range, e.g., \([-50, 50]\).  

---

### **TGRLlamaModel**

```python
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

        # Force hidden states
        base_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        base_config.output_hidden_states = True

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
        # Convert to half to match Llama final hidden states (FP16)
        self.temperature_mechanisms.half()

        # GSoT
        self.gsoT_enabled = config.gsoT_enabled
        self.temperature_conf_threshold = config.temperature_conf_threshold

        # Temperature regularization
        self.initial_temp = config.initial_temperature
        self.temp_reg_weight = config.temperature_regularization_weight

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> Dict[str, Any]:
        outputs = self.base_llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        hidden_states = outputs.hidden_states
        final_hidden = hidden_states[-1].to(torch.float16)

        # TTM
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
            q, k, v, temperature_vals, extended_mask
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

        # Optional cross-entropy + temperature reg
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            main_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # Temperature regularization
            t_diff = (temperature_vals - self.initial_temp)**2
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
```

**Key Points**  
- **4-bit** model weights with bitsandbytes.  
- A single temperature mechanism for final hidden states, optionally used inside a toy single-head attention block.  
- GSoT logic zeroes out tokens below a specified temperature threshold.  
- Cross-entropy plus a temperature regularization term if `labels` are provided.

---

### **WikiQADataset**

```python
class WikiQADataset(Dataset):
    """
    Minimal demonstration for WikiQA: use question + correct answer (label=1).
    """
    def __init__(self, tokenizer, split="train", max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = load_dataset("wiki_qa", split=split)
        self.samples = []
        for ex in self.dataset:
            # Keep only correct answers (label=1)
            if ex["label"] == 1:
                question = ex["question"]
                answer = ex["answer"]
                prompt = f"Q: {question}\nA: {answer}"
                self.samples.append(prompt)
        self.max_length = max_length

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
        encoded["labels"] = encoded["input_ids"].copy()
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
        }
```

- **Process**  
  - Loads WikiQA from Hugging Face (`"wiki_qa"`).  
  - Filters for `label=1`.  
  - Builds a prompt: “Q: question\nA: answer”.  
  - Tokenizes and stores in a dict with `labels = input_ids`.

---

### **Main Training & Inference Demo**

```python
def run_temperature_guided_reasoning_demo():
    """
    Train TGR (4-bit) on WikiQA dataset with label=1 only, then run generation
    with a clamp to avoid device asserts.
    """
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

    # Prepare WikiQA dataset (train split only for brevity)
    train_dataset = WikiQADataset(tokenizer, split="train", max_length=192)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./tgr_outputs_4bit_wikiqa",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=50,
        save_steps=200,
        evaluation_strategy="no",
        fp16=False,  # Avoid additional half precision scaling in Trainer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting TGR training on WikiQA (label=1 pairs only)...")
    trainer.train()
    logger.info("Training completed.")

    # Inference with clamp
    clamp_processor = ClampLogits(min_value=-50, max_value=50)
    logits_processors = LogitsProcessorList([clamp_processor])

    test_question = "When was Abraham Lincoln born?"
    prompt = f"Q: {test_question}\nA:"
    logger.info(f"Prompt: {prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Quick range check
    max_id = input_ids.max().item()
    if max_id >= tokenizer.vocab_size:
        raise ValueError(
            f"Out-of-range token ID={max_id} vs. vocab_size={tokenizer.vocab_size}."
        )

    logger.info("Generating with do_sample=True + clamp to avoid device asserts.")
    model.eval()
    with torch.no_grad():
        generation = model.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            logits_processor=logits_processors
        )

    decoded = tokenizer.decode(generation[0], skip_special_tokens=True)
    logger.info(f"Inference result: {decoded}")

    print("\n===== Inference Result (4-bit, WikiQA, clamp) =====")
    print(decoded)
    print("==================================================\n")


if __name__ == "__main__":
    run_temperature_guided_reasoning_demo()
```

**Commentary**  

1. **Train** for 1 epoch on label=1 QA pairs from WikiQA.  
2. **Generation**: clamp logits \([-50, 50]\).  
3. **do_sample=True** with top-p for randomness.  
4. Print the final decoded response.

---

## **How to Run**

1. **Install Required Packages**  
   ```bash
   pip install transformers>=4.31 bitsandbytes>=0.38.1 datasets
   # Optionally: accelerate wandb
   ```
2. **Run the Script**  
   ```bash
   python your_script.py
   ```
   - The script will download the Llama 2 (7B) checkpoint from Hugging Face Hub (requires acceptance of Meta’s Llama 2 license).  
   - Loads WikiQA, trains, and runs an inference example.

3. **Hardware**  
   - Requires a GPU with enough memory to hold the 4-bit quantized Llama model. Typically **12–16 GB VRAM** is enough for small-batch, short-sequence usage.

---

## **Potential Caveats & Next Steps**

1. **WikiQA**  
   - This dataset is minimal and has short QA pairs. Real-world usage may require negative sampling or multi-answer selection.  
   - If the dataset is too small, the model may quickly overfit.

2. **ClampLogits Range**  
   - `[min_value=-50, max_value=50]` is an empirical choice. If you see repeated device-side asserts or undesired generation, try adjusting the clamp bounds.

3. **Extended GSoT**  
   - We only apply GSoT at the final hidden states. Some research (e.g., in [*Guidance Is All You Need*](#) by Gomaa & Salah, 2024) suggests injecting TTM at every layer for more robust reasoning.

4. **Evaluation**  
   - The script uses `evaluation_strategy="no"` for simplicity. For a real application, consider a dev set or custom metrics.

5. **Llama License**  
   - *Llama 2* is distributed under specific license terms from Meta. Ensure you have valid acceptance to use it.

> **Reference**:  
> *Gomaa, E. (2024). Guidance is All You Need: Temperature-Guided Reasoning in Large Language Models. arXiv preprint (arXiv:2412.06822v1)
