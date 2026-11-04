Vendored inference code of MiniCheck (Tang, Laban and Durrett, EMNLP 2024),
https://github.com/Liyan06/MiniCheck at commit b58b9fa, Apache License 2.0 (see LICENSE).
Unmodified. Only the RoBERTa-Large path is used (`lytang/MiniCheck-RoBERTa-Large`, MIT).
`scripts/run_minicheck.py` batches the identical (chunk, claim) inputs; `--verify N`
compares it with the official per-example `Inferencer` path (max difference <= 1e-6).
