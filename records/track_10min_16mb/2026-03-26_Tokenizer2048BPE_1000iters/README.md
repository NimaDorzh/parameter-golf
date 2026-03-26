# Session Log: 2026-03-26 — vocab_size=2048 BPE Tokenizer (RTX 4060 Mobile)

## Summary

First local runs on RTX 4060 Mobile. Two experiments:
1. **Baseline smoke test** (200 iters, default tokenizer) — confirms the stack works.
2. **Tokenizer swap** (1000 iters, vocab_size=2048 BPE) — significant BPB improvement from tokenizer alone.

---

## Environment Setup

- OS: Linux, RTX 4060 Mobile
- Python venv + fish shell (`activate.fish`)
- PyTorch 2.11 + CUDA 13.0
- Fixed `CUDA Error 803` with a simple reboot
- HuggingFace login: `nimadorzh`

---

## Experiment 1 — Baseline Smoke Test

**Command:**
```bash
env MAX_WALLCLOCK_SECONDS=0 ITERATIONS=200 VAL_LOSS_EVERY=50 TRAIN_BATCH_TOKENS=32768 python3 train_gpt.py
```

**Results:**
| Metric | Value |
|--------|-------|
| val_bpb | 2.2812 |
| Model size (int8+zlib) | ~7.04 MB |
| Iters | 200 |

**Purpose:** Verify full stack: CUDA → training → quantisation → evaluation.

---

## Experiment 2 — Tokenizer Swap: vocab_size=2048 BPE

**Changes:**
- Added `sp_bpe_2048` entry to `data/tokenizer_specs.json`
- Downloaded and trained new SentencePiece BPE tokenizer (`fineweb_2048_bpe`, vocab=2048)
- Re-tokenized FineWeb dataset (≈1 hour)

**Command:**
```bash
env MAX_WALLCLOCK_SECONDS=0 ITERATIONS=1000 TRAIN_BATCH_TOKENS=32768 python3 train_gpt.py
```

**Results:**
| Metric | Value |
|--------|-------|
| val_loss | 2.6708 |
| val_bpb | **1.2139** |
| Eval time | 146 122 ms |
| Peak memory | 911 MiB alloc / 1760 MiB reserved |
| Serialized model (int8+zlib) | **13 015 918 bytes (13.02 MB)** |
| Payload raw | 17 705 248 bytes |
| raw_torch | 17 750 361 bytes |
| Payload ratio | 3.85× |
| Total submission size | 13 063 604 bytes (13.06 MB) |

**Raw output:**
```
peak memory allocated: 911 MiB reserved: 1760 MiB
Serialized model: 68273559 bytes
Code size: 47686 bytes
Total submission size: 68321245 bytes
Serialized model int8+zlib: 13015918 bytes (payload:17705248 raw_torch:17750361 payload_ratio:3.85x)
Total submission size int8+zlib: 13063604 bytes
final_int8_zlib_roundtrip val_loss:2.6708 val_bpb:1.2139 eval_time:146122ms
final_int8_zlib_roundtrip_exact val_loss:2.67078018 val_bpb:1.21388788
```

---

## Key Insight

Switching from the default tokenizer to vocab_size=2048 BPE dropped BPB by **−0.56** at just ~200 iters (2.28 → 1.72). At 1000 iters it converges further to **1.2139**. The model size grew only marginally (7.04 → 13 MB int8+zlib, expected due to larger embedding table).

---

## Next Steps

- **RunPod** — full training run (full wallclock budget) with the 2048 BPE tokenizer
- Explore: architecture scaling (layers/dim), MLP width, quantisation (int6/QAT), sliding window eval
