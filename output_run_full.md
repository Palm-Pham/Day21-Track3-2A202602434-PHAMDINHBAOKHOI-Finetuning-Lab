#3.core pipeline

Tesla T4 (cuda, sm_75, 14.6 GB) -> precision=fp16
  note: this GPU predates Ampere, so it has NO bfloat16. Using fp16 with gradient scaling instead. Tutorials that hardcode bf16=True fail here. 

tier=T4  mask=assistant-only  eval_limit=full

========================================================================
NB1 — data, chat template & loss mask
========================================================================
tier=T4  model=unsloth/Qwen3.5-4B  max_length=1024
250 mẫu huấn luyện
{
  "instruction": "Phân loại ticket chăm sóc khách hàng sau thành JSON với đúng 4 khóa: intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\nintent thuộc: doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin\nurgency thuộc: cao | trung_binh | thap\nsentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\nproduct: tên sản phẩm xuất hiện trong ticket.",
  "input": "Alo sh
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100% 2.76k/2.76k [00:00<00:00, 8.97MB/s]
tokenizer_config.json: 100% 15.7k/15.7k [00:00<00:00, 39.6MB/s]

tokenizer.json: downloading bytes:  13% 2.69M/20.0M [00:00<00:03, 5.17MB/s]
tokenizer.json: downloading bytes: 100% 6.23M/6.23M [00:00<00:00, 10.8MB/s,  614kB/s  ]
tokenizer.json: reconstructing file: 100% 20.0M/20.0M [00:00<00:00, 34.8MB/s, 1.97MB/s  ]
added_tokens.json: 100% 904/904 [00:00<00:00, 4.24MB/s]
special_tokens_map.json: 100% 876/876 [00:00<00:00, 3.11MB/s]
chat_template.jinja: 100% 7.99k/7.99k [00:00<00:00, 20.3MB/s]
eos_token: <|im_end|>
VERDICT: reasoning preserved — safe to train on traces

--- chuỗi đã render ---
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>

======================================================================
mode = assistant-only   supervised 39/94 (41%)
--- LOSS TÍNH TRÊN ĐOẠN NÀY ---
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

======================================================================
mode = everything   supervised 94/94 (100%)
--- LOSS TÍNH TRÊN ĐOẠN NÀY ---
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>

{
  "mask_mode": "assistant-only",
  "n_supervised": 39,
  "n_total": 94,
  "supervised_fraction": 0.4149,
  "answer_is_supervised": true,
  "question_is_masked": true
}
{
  "n": 250,
  "mean": 93.1,
  "p50": 93,
  "p95": 98,
  "p99": 100,
  "max": 101,
  "suggested_max_length": 256
}

⚠ p95 gợi ý max_length=256 nhưng tier đang đặt 1024. Ghi lại lựa chọn của bạn trong REPORT.md.
train=225  val=25  -> /content/Day21-Track3-Finetuning-Lab/data/split

-- NB1 ok in 21s

========================================================================
NB2 — freeze eval + three baselines
========================================================================
target=50  regression=15  tier=T4
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors.index.json: 100% 76.2k/76.2k [00:00<00:00, 149MB/s]
Downloading bytes:           |  0.00B            
Reconstructing (incomplete total...): |          |  0.00B /  0.00B            

Fetching 2 files:   0% 0/2 [00:00<?, ?it/s]
Reconstructing (incomplete total...):   0% 0.00/3.99G [00:00<?, ?B/s]         
Downloading bytes:   0% 3.27M/9.32G [00:00<29:21, 5.29MB/s]
Downloading bytes:   3% 269M/9.32G [00:01<00:33, 270MB/s,  287MB/s  ]
Downloading bytes:   4% 330M/9.32G [00:01<00:45, 198MB/s,  224MB/s  ]
Downloading bytes:   4% 360M/9.32G [00:02<00:49, 181MB/s,  198MB/s  ]
Downloading bytes:   5% 477M/9.32G [00:02<00:34, 254MB/s,  241MB/s  ]
Downloading bytes:   8% 710M/9.32G [00:03<00:34, 252MB/s,  289MB/s  ]
Downloading bytes:   8% 746M/9.32G [00:03<00:38, 224MB/s,  252MB/s  ]
Downloading bytes:   9% 804M/9.32G [00:04<00:47, 178MB/s,  194MB/s  ]
Downloading bytes:   9% 858M/9.32G [00:04<00:40, 208MB/s,  182MB/s  ]
Downloading bytes:  12% 1.11G/9.32G [00:05<00:42, 192MB/s,  299MB/s  ]
Downloading bytes:  15% 1.43G/9.32G [00:06<00:43, 179MB/s,  173MB/s  ]
Downloading bytes:  20% 1.82G/9.32G [00:09<00:43, 172MB/s,  194MB/s  ]
Downloading bytes:  20% 1.84G/9.32G [00:09<01:03, 118MB/s,  172MB/s  ]
Downloading bytes:  20% 1.90G/9.32G [00:10<01:07, 109MB/s,  105MB/s  ]
Downloading bytes:  24% 2.19G/9.32G [00:11<00:31, 223MB/s,  256MB/s  ]
Downloading bytes:  25% 2.32G/9.32G [00:12<00:40, 172MB/s,  207MB/s  ]
Downloading bytes:  26% 2.39G/9.32G [00:13<00:41, 168MB/s,  162MB/s  ]
Downloading bytes:  36% 3.34G/9.32G [00:17<00:26, 224MB/s,  216MB/s  ]
Reconstructing (incomplete total...):  23% 2.14G/9.32G [00:17<01:38, 72.6MB/s,  213MB/s  ]
Downloading bytes:  37% 3.42G/9.32G [00:18<00:37, 158MB/s,  185MB/s  ]
Downloading bytes:  50% 4.65G/9.32G [00:23<00:22, 210MB/s,  194MB/s  ]
Downloading bytes:  51% 4.77G/9.32G [00:25<00:46, 97.7MB/s, 96.5MB/s  ]
Downloading bytes:  52% 4.86G/9.32G [00:25<00:42, 105MB/s, 99.9MB/s  ] 
Downloading bytes:  53% 4.91G/9.32G [00:26<00:41, 107MB/s,  103MB/s  ]
Downloading bytes:  53% 4.95G/9.32G [00:26<00:32, 134MB/s,  122MB/s  ]
Downloading bytes:  55% 5.12G/9.32G [00:27<00:31, 135MB/s,  188MB/s  ]
Downloading bytes:  56% 5.26G/9.32G [00:28<00:36, 110MB/s,  120MB/s  ]
Downloading bytes:  58% 5.38G/9.32G [00:29<00:24, 161MB/s,  141MB/s  ]
Downloading bytes:  66% 6.11G/9.32G [00:32<00:16, 193MB/s,  185MB/s  ]
Downloading bytes:  67% 6.27G/9.32G [00:33<00:22, 137MB/s,  134MB/s  ]
Downloading bytes:  74% 6.91G/9.32G [00:36<00:15, 156MB/s,  268MB/s  ]
Reconstructing (incomplete total...):  56% 5.19G/9.32G [00:37<00:33, 121MB/s,  189MB/s  ]
Downloading bytes:  75% 6.97G/9.32G [00:38<00:43, 54.1MB/s, 61.8MB/s  ]
Downloading bytes:  76% 7.06G/9.32G [00:39<00:38, 57.9MB/s, 54.8MB/s  ]
Downloading bytes:  81% 7.54G/9.32G [00:42<00:12, 142MB/s,  140MB/s  ]
Downloading bytes:  82% 7.66G/9.32G [00:43<00:13, 125MB/s,  147MB/s  ]
Downloading bytes:  82% 7.69G/9.32G [00:43<00:13, 123MB/s,  125MB/s  ]
Downloading bytes:  86% 8.04G/9.32G [00:45<00:04, 307MB/s,  301MB/s  ]
Reconstructing (incomplete total...):  74% 6.85G/9.32G [00:50<00:28, 87.9MB/s,  196MB/s  ]
Downloading bytes:  86% 8.05G/9.32G [00:59<00:04, 307MB/s,  307MB/s  ]
Reconstructing (incomplete total...):  80% 7.47G/9.32G [01:01<00:28, 64.2MB/s, 94.9MB/s  ]
Reconstructing (incomplete total...):  80% 7.49G/9.32G [01:02<00:33, 55.1MB/s, 64.2MB/s  ]
Reconstructing (incomplete total...):  82% 7.66G/9.32G [01:03<00:22, 73.4MB/s, 55.1MB/s  ]
Reconstructing (incomplete total...):  84% 7.87G/9.32G [01:04<00:15, 91.7MB/s, 73.4MB/s  ]
Reconstructing (incomplete total...):  87% 8.15G/9.32G [01:06<00:12, 97.7MB/s, 91.7MB/s  ]
Reconstructing (incomplete total...):  90% 8.40G/9.32G [01:10<00:10, 84.0MB/s, 97.7MB/s  ]
Reconstructing (incomplete total...):  91% 8.50G/9.32G [01:12<00:10, 75.9MB/s, 84.0MB/s  ]
Reconstructing (incomplete total...):  92% 8.53G/9.32G [01:14<00:13, 59.9MB/s, 75.9MB/s  ]

Fetching 2 files:  50% 1/2 [01:14<01:14, 74.99s/it]
Reconstructing (incomplete total...):  99% 9.24G/9.32G [01:15<00:00, 180MB/s, 59.9MB/s  ] 

Fetching 2 files: 100% 2/2 [01:15<00:00, 37.92s/it]
Download complete: 100% 8.05G/8.05G [01:15<00:00, 307MB/s,  307MB/s  ]
Download complete: 100% 8.05G/8.05G [01:15<00:00, 106MB/s,  307MB/s  ]
Reconstruction complete: 100% 9.32G/9.32G [01:15<00:00, 123MB/s,  180MB/s  ]
Loading weights: 100% 426/426 [00:35<00:00, 12.15it/s]
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
[transformers] `causal_conv1d_update` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `fused_recurrent_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
  [(a) base + naive prompt/target] batch 1/13     16s elapsed  ~  187s left
  [(a) base + naive prompt/target] batch 2/13     28s elapsed  ~  154s left
  [(a) base + naive prompt/target] batch 3/13     41s elapsed  ~  135s left
  [(a) base + naive prompt/target] batch 4/13     53s elapsed  ~  119s left
  [(a) base + naive prompt/target] batch 5/13     66s elapsed  ~  105s left
  [(a) base + naive prompt/target] batch 6/13     78s elapsed  ~   91s left
  [(a) base + naive prompt/target] batch 7/13     91s elapsed  ~   78s left
  [(a) base + naive prompt/target] batch 8/13    103s elapsed  ~   64s left
  [(a) base + naive prompt/target] batch 9/13    116s elapsed  ~   51s left
  [(a) base + naive prompt/target] batch 10/13    128s elapsed  ~   39s left
  [(a) base + naive prompt/target] batch 11/13    141s elapsed  ~   26s left
  [(a) base + naive prompt/target] batch 12/13    154s elapsed  ~   13s left
  [(a) base + naive prompt/target] batch 13/13    166s elapsed  ~    0s left
  [(a) base + naive prompt/target] done: 50 prompts in 166s
  [(a) base + naive prompt/regression] batch 1/4      6s elapsed  ~   17s left
  [(a) base + naive prompt/regression] batch 2/4     13s elapsed  ~   13s left
  [(a) base + naive prompt/regression] batch 3/4     20s elapsed  ~    7s left
  [(a) base + naive prompt/regression] batch 4/4     28s elapsed  ~    0s left
  [(a) base + naive prompt/regression] done: 15 prompts in 28s
(a) base + naive prompt      target=0.000  regression=0.791  format=0.000  3320ms
  [(b) base + optimized prompt/target] batch 1/13      5s elapsed  ~   57s left
  [(b) base + optimized prompt/target] batch 2/13      8s elapsed  ~   46s left
  [(b) base + optimized prompt/target] batch 3/13     12s elapsed  ~   41s left
  [(b) base + optimized prompt/target] batch 4/13     16s elapsed  ~   37s left
  [(b) base + optimized prompt/target] batch 5/13     20s elapsed  ~   32s left
  [(b) base + optimized prompt/target] batch 6/13     24s elapsed  ~   28s left
  [(b) base + optimized prompt/target] batch 7/13     29s elapsed  ~   25s left
  [(b) base + optimized prompt/target] batch 8/13     33s elapsed  ~   20s left
  [(b) base + optimized prompt/target] batch 9/13     36s elapsed  ~   16s left
  [(b) base + optimized prompt/target] batch 10/13     41s elapsed  ~   12s left
  [(b) base + optimized prompt/target] batch 11/13     45s elapsed  ~    8s left
  [(b) base + optimized prompt/target] batch 12/13     49s elapsed  ~    4s left
  [(b) base + optimized prompt/target] batch 13/13     53s elapsed  ~    0s left
  [(b) base + optimized prompt/target] done: 50 prompts in 53s
  [(b) base + optimized prompt/regression] batch 1/4      5s elapsed  ~   15s left
  [(b) base + optimized prompt/regression] batch 2/4     13s elapsed  ~   13s left
  [(b) base + optimized prompt/regression] batch 3/4     20s elapsed  ~    7s left
  [(b) base + optimized prompt/regression] batch 4/4     28s elapsed  ~    0s left
  [(b) base + optimized prompt/regression] done: 15 prompts in 28s
(b) base + optimized prompt  target=0.765  regression=0.791  format=1.000  1053ms
{
  "tier": "T4",
  "model": "unsloth/Qwen3.5-4B",
  "baseline_a": {
    "target": 0.0,
    "regression": 0.7911111111111111,
    "format": 0.0,
    "latency_ms": 3320.0272858199983,
    "n": 50,
    "extra": {}
  },
  "baseline_b": {
    "target": 0.765,
    "regression": 0.7911111111111111,
    "format": 1.0,
    "latency_ms": 1052.7279678399964,
    "n": 50,
    "extra": {}
  },
  "optimized_prompt_sha": "719e74d3b6232053",
  "n_target": 50,
  "n_regression": 15,
  "eval_limit": null,
  "smoke_mode": false
}

-- NB2 ok in 403s

========================================================================
NB3 — train the correct configuration
========================================================================
T4 · unsloth/Qwen3.5-4B · all-linear · r=16 · LR 10x · 16-bit
Tesla T4 (cuda, sm_75, 14.6 GB) -> precision=fp16
  note: this GPU predates Ampere, so it has NO bfloat16. Using fp16 with gradient scaling instead. Tutorials that hardcode bf16=True fail here.
Loading weights: 100% 426/426 [00:32<00:00, 13.06it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
{
  "num_hidden_layers": 32,
  "full_attention_interval": 4,
  "linear_num_key_heads": 16,
  "layer_types": {
    "linear_attention": 24,
    "full_attention": 8
  }
}
placement=text-linear  modules=['down_proj', 'gate_proj', 'in_proj_a', 'in_proj_b', 'in_proj_qkv', 'in_proj_z', 'k_proj', 'o_proj', 'out_proj', 'q_proj', 'up_proj', 'v_proj']
trainable LoRA params ≈ 32.46 M
    {'placement': 'text-linear', 'modules': 12, 'r': 16, 'trainable': 32464896}
    {'placement': 'attn-only(q,v)', 'modules': 2, 'r': 16, 'trainable': 1835008}
    {'placement': 'attn-only(q,v) matched', 'modules': 2, 'r': 283, 'trainable': 32456704}
Dataset({
    features: ['input_ids', 'labels', 'attention_mask'],
    num_rows: 225
})
mask_mode = assistant-only   supervised 9014/20951 tokens (43.0%)
epochs=2.0  ->  30 optimizer steps  (NB4 runs its contrasts at exactly this many)
{
  "output_dir": "/content/Day21-Track3-Finetuning-Lab/adapters/correct",
  "max_length": "1024",
  "per_device_train_batch_size": "1",
  "gradient_accumulation_steps": "16",
  "learning_rate": "0.0001",
  "lr_scheduler_type": "cosine",
  "num_train_epochs": "2.0",
  "logging_steps": "5",
  "save_strategy": "no",
  "report_to": "none",
  "seed": "42",
  "packing": "False",
  "loss_type": "chunked_nll",
  "gradient_checkpointing": "True",
  "warmup_steps": "3",
  "bf16": "False",
  "fp16": "True",
  "padding_free": "False"
}
Truncating train dataset: 100% 225/225 [00:00<00:00, 5811.57 examples/s]
Dropping fully masked examples from train dataset: 100% 225/225 [00:00<00:00, 44643.47 examples/s]
precision fix: {'precision': 'fp16', 'recast': 0, 'trainable_tensors': 496}
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 248046}.
  0% 0/30 [00:00<?, ?it/s][transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
{'loss': '2.163', 'grad_norm': 'nan', 'learning_rate': '9.966e-05', 'entropy': '1.177', 'num_tokens': '7479', 'mean_token_accuracy': '0.6559', 'epoch': '0.3556'}
{'loss': '1.384', 'grad_norm': '2.628', 'learning_rate': '8.83e-05', 'entropy': '1.029', 'num_tokens': '1.49e+04', 'mean_token_accuracy': '0.77', 'epoch': '0.7111'}
{'loss': '0.1401', 'grad_norm': '0.5771', 'learning_rate': '6.434e-05', 'entropy': '0.2705', 'num_tokens': '2.095e+04', 'mean_token_accuracy': '0.9598', 'epoch': '1'}
{'loss': '0.02885', 'grad_norm': '1.022', 'learning_rate': '3.566e-05', 'entropy': '0.04043', 'num_tokens': '2.843e+04', 'mean_token_accuracy': '0.9873', 'epoch': '1.356'}
{'loss': '0.01624', 'grad_norm': '0.2453', 'learning_rate': '1.17e-05', 'entropy': '0.02252', 'num_tokens': '3.59e+04', 'mean_token_accuracy': '0.994', 'epoch': '1.711'}
{'loss': '0.02708', 'grad_norm': 'nan', 'learning_rate': '3.381e-07', 'entropy': '0.01588', 'num_tokens': '4.19e+04', 'mean_token_accuracy': '0.9965', 'epoch': '2'}
{'train_runtime': '426.5', 'train_samples_per_second': '1.055', 'train_steps_per_second': '0.07', 'train_loss': '0.6266', 'epoch': '2'}
100% 30/30 [07:06<00:00, 14.22s/it]
train 427s  final loss 0.6266
saved -> /content/Day21-Track3-Finetuning-Lab/adapters/correct
{
  "run": "correct",
  "label": "all-linear · r=16 · LR 10x · 16-bit",
  "tier": "T4",
  "model": "unsloth/Qwen3.5-4B",
  "precision": "fp16",
  "placement": "text-linear",
  "n_target_modules": 12,
  "r": 16,
  "lora_alpha": 32,
  "learning_rate": 0.0001,
  "load_in_4bit": false,
  "trainable_params": 32464896,
  "train_seconds": 427.1,
  "peak_vram_gb": 8.78,
  "final_loss": 0.6266,
  "mask_mode": "assistant-only",
  "max_steps": 30
}

-- NB3 ok in 488s

========================================================================
NB4 — three misconfiguration contrasts
========================================================================
Loading weights: 100% 426/426 [00:32<00:00, 13.13it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
| placement | modules | r | trainable |
|---|---|---|---|
| text-linear | 12 | 16 | 32464896 |
| attn-only(q,v) | 2 | 16 | 1835008 |
| attn-only(q,v) matched | 2 | 283 | 32456704 |
======================================================================
RUN attn_only: q,v only · r=matched · LR 10x · 16-bit
     Mistake #1 (§11.2): attention-only placement, rank raised to *match parameter count*. If rank were the lever, this would win.
Loading weights: 100% 426/426 [00:32<00:00, 13.11it/s]
  train_ds: Dataset({
    features: ['input_ids', 'labels', 'attention_mask'],
    num_rows: 225
})
  matched rank for attn_only: r=283 (alpha=566)
Truncating train dataset: 100% 225/225 [00:00<00:00, 5963.28 examples/s]
Dropping fully masked examples from train dataset: 100% 225/225 [00:00<00:00, 43994.14 examples/s]
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 248046}.
  0% 0/30 [00:00<?, ?it/s][transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
{'loss': '2.163', 'grad_norm': '5.846', 'learning_rate': '9.966e-05', 'entropy': '1.177', 'num_tokens': '7479', 'mean_token_accuracy': '0.6559', 'epoch': '0.3556'}
{'loss': '0.8275', 'grad_norm': '1.84', 'learning_rate': '8.83e-05', 'entropy': '0.94', 'num_tokens': '1.49e+04', 'mean_token_accuracy': '0.8412', 'epoch': '0.7111'}
{'loss': '0.149', 'grad_norm': '1.475', 'learning_rate': '6.434e-05', 'entropy': '0.2872', 'num_tokens': '2.095e+04', 'mean_token_accuracy': '0.958', 'epoch': '1'}
{'loss': '0.03947', 'grad_norm': '0.8326', 'learning_rate': '3.566e-05', 'entropy': '0.06724', 'num_tokens': '2.843e+04', 'mean_token_accuracy': '0.987', 'epoch': '1.356'}
{'loss': '0.02168', 'grad_norm': '0.2151', 'learning_rate': '1.17e-05', 'entropy': '0.03181', 'num_tokens': '3.59e+04', 'mean_token_accuracy': '0.9912', 'epoch': '1.711'}
{'loss': '0.0257', 'grad_norm': 'nan', 'learning_rate': '3.381e-07', 'entropy': '0.02309', 'num_tokens': '4.19e+04', 'mean_token_accuracy': '0.9945', 'epoch': '2'}
{'train_runtime': '274.3', 'train_samples_per_second': '1.75', 'train_steps_per_second': '0.109', 'train_loss': '0.5378', 'epoch': '2'}
100% 30/30 [04:34<00:00,  9.14s/it]
======================================================================
RUN wrong_lr: all-linear · r=16 · LR 1x (full-FT scale) · 16-bit
     Mistake #2 (§11.3): a full-fine-tune learning rate applied to LoRA.
Loading weights: 100% 426/426 [00:31<00:00, 13.42it/s]
Truncating train dataset: 100% 225/225 [00:00<00:00, 9350.23 examples/s]
Dropping fully masked examples from train dataset: 100% 225/225 [00:00<00:00, 46059.76 examples/s]
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 248046}.
{'loss': '2.163', 'grad_norm': 'nan', 'learning_rate': '9.966e-06', 'entropy': '1.177', 'num_tokens': '7479', 'mean_token_accuracy': '0.6559', 'epoch': '0.3556'}
{'loss': '2.066', 'grad_norm': '5.107', 'learning_rate': '8.83e-06', 'entropy': '1.198', 'num_tokens': '1.49e+04', 'mean_token_accuracy': '0.6802', 'epoch': '0.7111'}
{'loss': '1.606', 'grad_norm': 'nan', 'learning_rate': '6.434e-06', 'entropy': '1.178', 'num_tokens': '2.095e+04', 'mean_token_accuracy': '0.7176', 'epoch': '1'}
{'loss': '1.326', 'grad_norm': '4.628', 'learning_rate': '3.566e-06', 'entropy': '1.103', 'num_tokens': '2.843e+04', 'mean_token_accuracy': '0.7564', 'epoch': '1.356'}
{'loss': '1.141', 'grad_norm': '4.213', 'learning_rate': '1.17e-06', 'entropy': '1.076', 'num_tokens': '3.59e+04', 'mean_token_accuracy': '0.7982', 'epoch': '1.711'}
{'loss': '1.119', 'grad_norm': 'nan', 'learning_rate': '3.381e-08', 'entropy': '1.068', 'num_tokens': '4.19e+04', 'mean_token_accuracy': '0.7909', 'epoch': '2'}
{'train_runtime': '406.5', 'train_samples_per_second': '1.181', 'train_steps_per_second': '0.074', 'train_loss': '1.57', 'epoch': '2'}
100% 30/30 [06:46<00:00, 13.55s/it]
======================================================================
RUN qlora: all-linear · r=16 · LR 10x · 4-bit QLoRA
     The vendor says do NOT use QLoRA on Qwen3.5 (§13). Measure the cost yourself instead of taking either side on faith.
Loading weights: 100% 426/426 [00:32<00:00, 13.08it/s]
Truncating train dataset: 100% 225/225 [00:00<00:00, 10017.18 examples/s]
Dropping fully masked examples from train dataset: 100% 225/225 [00:00<00:00, 46543.62 examples/s]
  precision fix: recast 496/496 trainable tensors bf16 -> fp32 for the fp16 GradScaler
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 248046}.
{'loss': '2.155', 'grad_norm': 'nan', 'learning_rate': '9.966e-05', 'entropy': '1.34', 'num_tokens': '7479', 'mean_token_accuracy': '0.6563', 'epoch': '0.3556'}
{'loss': '1.731', 'grad_norm': '3.358', 'learning_rate': '8.83e-05', 'entropy': '1.287', 'num_tokens': '1.49e+04', 'mean_token_accuracy': '0.7214', 'epoch': '0.7111'}
{'loss': '0.2408', 'grad_norm': '1.031', 'learning_rate': '6.434e-05', 'entropy': '0.3823', 'num_tokens': '2.095e+04', 'mean_token_accuracy': '0.9378', 'epoch': '1'}
{'loss': '0.05115', 'grad_norm': '1.043', 'learning_rate': '3.566e-05', 'entropy': '0.07512', 'num_tokens': '2.843e+04', 'mean_token_accuracy': '0.9829', 'epoch': '1.356'}
{'loss': '0.0308', 'grad_norm': '0.3144', 'learning_rate': '1.17e-05', 'entropy': '0.0356', 'num_tokens': '3.59e+04', 'mean_token_accuracy': '0.9875', 'epoch': '1.711'}
{'loss': '0.02622', 'grad_norm': '3.284', 'learning_rate': '3.381e-07', 'entropy': '0.03075', 'num_tokens': '4.19e+04', 'mean_token_accuracy': '0.9915', 'epoch': '2'}
{'train_runtime': '477.3', 'train_samples_per_second': '1.006', 'train_steps_per_second': '0.063', 'train_loss': '0.7058', 'epoch': '2'}
100% 30/30 [07:57<00:00, 15.91s/it]
| run | label | r | trainable_params | learning_rate | final_loss | train_seconds | peak_vram_gb |
|---|---|---|---|---|---|---|---|
| correct | all-linear · r=16 · LR 10x · 16-bit | 16 | 32464896 | 0.0001 | 0.6266 | 427.1 | 8.78 |
| attn_only | q,v only · r=matched · LR 10x · 16-bit | 283 | 32456704 | 0.0001 | 0.5378 | 274.7 | 8.79 |
| wrong_lr | all-linear · r=16 · LR 1x (full-FT scale) · 16-bit | 16 | 32464896 | 1e-05 | 1.5702 | 407.0 | 8.78 |
| qlora | all-linear · r=16 · LR 10x · 4-bit QLoRA | 16 | 32464896 | 0.0001 | 0.7058 | 477.8 | 3.86 |

-- NB4 ok in 1324s

========================================================================
NB5 — four-group eval + verdict
========================================================================
baseline (b) target = 0.765 — đây là mốc phải vượt
Loading weights: 100% 426/426 [00:31<00:00, 13.32it/s]
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
[transformers] `causal_conv1d_update` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.
[transformers] `fused_recurrent_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. This is correct but much slower; install `flash-linear-attention` for the optimized kernel.
  [ft/target] batch 1/13      7s elapsed  ~   84s left
  [ft/target] batch 2/13     12s elapsed  ~   68s left
  [ft/target] batch 3/13     18s elapsed  ~   60s left
  [ft/target] batch 4/13     23s elapsed  ~   52s left
  [ft/target] batch 5/13     29s elapsed  ~   46s left
  [ft/target] batch 6/13     34s elapsed  ~   40s left
  [ft/target] batch 7/13     39s elapsed  ~   33s left
  [ft/target] batch 8/13     44s elapsed  ~   28s left
  [ft/target] batch 9/13     49s elapsed  ~   22s left
  [ft/target] batch 10/13     55s elapsed  ~   16s left
  [ft/target] batch 11/13     60s elapsed  ~   11s left
  [ft/target] batch 12/13     65s elapsed  ~    5s left
  [ft/target] batch 13/13     71s elapsed  ~    0s left
  [ft/target] done: 50 prompts in 71s
  [ft/regression] batch 1/4     11s elapsed  ~   32s left
  [ft/regression] batch 2/4     22s elapsed  ~   22s left
  [ft/regression] batch 3/4     29s elapsed  ~   10s left
  [ft/regression] batch 4/4     41s elapsed  ~    0s left
  [ft/regression] done: 15 prompts in 41s
fine-tune: {'target': 0.97, 'regression': 0.5888888888888889, 'format': 1.0, 'latency_ms': 1420.3766413599897, 'n': 50, 'extra': {'valid_trace_rate': 0.0}}
| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3320.0 | 50 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1052.7 | 50 |
| (c) LoRA fine-tune | 0.97 | 0.5889 | 1.0 | 1420.4 | 50 |
FAILED
 - general capability regressed by 0.202 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.
Loading weights: 100% 426/426 [00:31<00:00, 13.40it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  [attn_only/target] batch 1/13      3s elapsed  ~   41s left
  [attn_only/target] batch 2/13      7s elapsed  ~   41s left
  [attn_only/target] batch 3/13     11s elapsed  ~   36s left
  [attn_only/target] batch 4/13     14s elapsed  ~   32s left
  [attn_only/target] batch 5/13     18s elapsed  ~   28s left
  [attn_only/target] batch 6/13     22s elapsed  ~   25s left
  [attn_only/target] batch 7/13     25s elapsed  ~   21s left
  [attn_only/target] batch 8/13     28s elapsed  ~   18s left
  [attn_only/target] batch 9/13     31s elapsed  ~   14s left
  [attn_only/target] batch 10/13     35s elapsed  ~   11s left
  [attn_only/target] batch 11/13     39s elapsed  ~    7s left
  [attn_only/target] batch 12/13     42s elapsed  ~    3s left
  [attn_only/target] batch 13/13     46s elapsed  ~    0s left
  [attn_only/target] done: 50 prompts in 46s
attn_only: target=0.970  format=1.000
Loading weights: 100% 426/426 [00:31<00:00, 13.66it/s]
  [wrong_lr/target] batch 1/13     21s elapsed  ~  251s left
  [wrong_lr/target] batch 2/13     42s elapsed  ~  229s left
  [wrong_lr/target] batch 3/13     62s elapsed  ~  206s left
  [wrong_lr/target] batch 4/13     83s elapsed  ~  187s left
  [wrong_lr/target] batch 5/13    104s elapsed  ~  166s left
  [wrong_lr/target] batch 6/13    124s elapsed  ~  145s left
  [wrong_lr/target] batch 7/13    145s elapsed  ~  124s left
  [wrong_lr/target] batch 8/13    166s elapsed  ~  104s left
  [wrong_lr/target] batch 9/13    187s elapsed  ~   83s left
  [wrong_lr/target] batch 10/13    208s elapsed  ~   62s left
  [wrong_lr/target] batch 11/13    229s elapsed  ~   42s left
  [wrong_lr/target] batch 12/13    249s elapsed  ~   21s left
  [wrong_lr/target] batch 13/13    270s elapsed  ~    0s left
  [wrong_lr/target] done: 50 prompts in 270s
wrong_lr: target=0.000  format=0.000
Loading weights: 100% 426/426 [00:33<00:00, 12.57it/s]
  [qlora/target] batch 1/13      7s elapsed  ~   89s left
  [qlora/target] batch 2/13     15s elapsed  ~   80s left
  [qlora/target] batch 3/13     22s elapsed  ~   73s left
  [qlora/target] batch 4/13     29s elapsed  ~   65s left
  [qlora/target] batch 5/13     36s elapsed  ~   58s left
  [qlora/target] batch 6/13     43s elapsed  ~   50s left
  [qlora/target] batch 7/13     50s elapsed  ~   43s left
  [qlora/target] batch 8/13     57s elapsed  ~   35s left
  [qlora/target] batch 9/13     64s elapsed  ~   28s left
  [qlora/target] batch 10/13     71s elapsed  ~   21s left
  [qlora/target] batch 11/13     78s elapsed  ~   14s left
  [qlora/target] batch 12/13     85s elapsed  ~    7s left
  [qlora/target] batch 13/13     92s elapsed  ~    0s left
  [qlora/target] done: 50 prompts in 92s
qlora: target=0.940  format=1.000

| run | target | format | latency_ms | n |
|---|---|---|---|---|
| correct | 0.97 | 1.0 | 1420.4 | 50 |
| attn_only | 0.97 | 1.0 | 920.8 | 50 |
| wrong_lr | 0.0 | 0.0 | 5407.3 | 50 |
| qlora | 0.94 | 1.0 | 1845.4 | 50 |
--- 3 ca TỆ NHẤT (bắt buộc đưa vào report) ---
| i | ticket | ft_score | ft_pred |
|---|---|---|---|
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | 0.75 | {"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | 0.75 | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện.  | 0.75 | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment" |

--- 3 ca TỐT NHẤT ---
| i | ticket | ft_score | ft_pred |
|---|---|---|---|
| 47 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | 1.0 | {"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment":  |
| 48 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu.  | 1.0 | {"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sen |
| 49 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n | 1.0 | {"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sent |

-- NB5 ok in 682s

----------------------------------------
stage timings
  nb1        21s  (0.3 min)
  nb2       403s  (6.7 min)
  nb3       488s  (8.1 min)
  nb4      1324s  (22.1 min)
  nb5       682s  (11.4 min)
  total    2917s (48.6 min)

# 4. gatekeeper

[  ok  ] labkit imports                                   
[  ok  ] tier resolves                                    T4 -> unsloth/Qwen3.5-4B
[  ok  ] all tiers respect the <32 effective-batch rule   
[  ok  ] data/train_seed.jsonl                            250 rows
[  ok  ] data/eval_target.jsonl                           50 rows
[  ok  ] data/eval_regression.jsonl                       15 rows
[  ok  ] unit tests                                       119 passed in 2.79s
[  ok  ] results/template_check.json                      
[  ok  ] results/mask_proof.json                          
[  ok  ] results/token_stats.json                         
[  ok  ] results/baselines_frozen.json                    
[  ok  ] results/runs.csv                                 
[  ok  ] results/verdict.json                             
[  ok  ] results/autopsy.json                             
[  ok  ] submission/REPORT.md                             
[ FAIL ] REPORT.md filled in                              6 placeholders still present (<điền>, <paste>, ...) — this is still the template
[  ok  ] mask proof asserts                               
[  ok  ] full eval set used                               50 target items
[  ok  ] baseline (b) prompt unmodified                   
[  ok  ] baseline (b) beats (a)                           (a)=0.000 -> (b)=0.765
[  ok  ] eval sets unmodified                             
[  ok  ] NB3 run present                                  
[  ok  ] NB4 contrast runs                                
[  ok  ] all runs share ONE step budget                   ['attn_only', 'correct', 'qlora', 'wrong_lr'] at 30 steps
[  ok  ] attn_only is a FAIR contrast                     32,456,704 vs 32,464,896 trainable params
[  ok  ] verdict recorded                                 FAILED (target Δ +0.205, regression Δ -0.202)
[ warn ] note                                             A FAILED verdict is fully gradeable — analyse it honestly in REPORT.md rather than loosening the gate.

25 passed · 1 warnings · 1 failures

Not ready to submit — fix the FAILs above.

================ results/ ================
total 52
drwxr-xr-x  2 root root  4096 Oct  7 10:49 .
drwxr-xr-x 15 root root  4096 Oct  7 10:00 ..
-rw-r--r--  1 root root   433 Oct  7 10:49 autopsy.json
-rw-r--r--  1 root root   516 Oct  7 10:07 baselines_frozen.json
-rw-r--r--  1 root root     0 Oct  7 09:59 .gitkeep
-rw-r--r--  1 root root   592 Oct  7 10:00 mask_proof.json
-rw-r--r--  1 root root 13677 Oct  7 10:49 qualitative.json
-rw-r--r--  1 root root  1096 Oct  7 10:37 runs.csv
-rw-r--r--  1 root root   277 Oct  7 10:00 template_check.json
-rw-r--r--  1 root root   115 Oct  7 10:00 token_stats.json
-rw-r--r--  1 root root   804 Oct  7 10:40 verdict.json

---- runs.csv ----
run,label,tier,model,precision,placement,n_target_modules,r,lora_alpha,learning_rate,load_in_4bit,trainable_params,train_seconds,peak_vram_gb,final_loss,mask_mode,max_steps,teaches
correct,all-linear · r=16 · LR 10x · 16-bit,T4,unsloth/Qwen3.5-4B,fp16,text-linear,12,16,32,0.0001,False,32464896,427.1,8.78,0.6266,assistant-only,30,
attn_only,"q,v only · r=matched · LR 10x · 16-bit",T4,unsloth/Qwen3.5-4B,fp16,attn-only,2,283,566,0.0001,False,32456704,274.7,8.79,0.5378,,30,"Mistake #1 (§11.2): attention-only placement, rank raised to *match parameter count*. If rank were the lever, this would win."
wrong_lr,all-linear · r=16 · LR 1x (full-FT scale) · 16-bit,T4,unsloth/Qwen3.5-4B,fp16,text-linear,12,16,32,1e-05,False,32464896,407.0,8.78,1.5702,,30,Mistake #2 (§11.3): a full-fine-tune learning rate applied to LoRA.
qlora,all-linear · r=16 · LR 10x · 4-bit QLoRA,T4,unsloth/Qwen3.5-4B,fp16,text-linear,12,16,32,0.0001,True,32464896,477.8,3.86,0.7058,,30,The vendor says do NOT use QLoRA on Qwen3.5 (§13). Measure the cost yourself instead of taking either side on faith.

---- verdict.json ----
{
  "comparison": [
    {
      "run": "(a) base + naive prompt",
      "target": 0.0,
      "regression": 0.7911,
      "format": 0.0,
      "latency_ms": 3320.0,
      "n": 50
    },
    {
      "run": "(b) base + optimized prompt",
      "target": 0.765,
      "regression": 0.7911,
      "format": 1.0,
      "latency_ms": 1052.7,
      "n": 50
    },
    {
      "run": "(c) LoRA fine-tune",
      "target": 0.97,
      "regression": 0.5889,
      "format": 1.0,
      "latency_ms": 1420.4,
      "n": 50
    }
  ],
  "verdict": {
    "passed": false,
    "reasons": [
      "general capability regressed by 0.202 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."
    ],
    "target_delta": 0.20499999999999996,
    "regression_delta": -0.2022222222222222
  },
  "valid_trace_rate": 0.0
}
