# LongMemEval error-memory package

Archive: `docs/research/longmemeval-error-memories-50-20260815.tar.gz`

- Selection: the 50 unique `dataset_index` values with `judge_score=0` in the
  frozen 500-item canonical LongMemEval result.
- Contents: each selected item's completed `checkpoint.json` and complete
  `memory/` directory.
- Layout: `longmemeval-error-memories/items/NNNN/`.
- Included manifest: `longmemeval-error-memories/manifest.json`.
- Uncompressed size: 42.41 MiB across 5,972 archive members.
- Compressed size: 13.35 MiB.
- SHA-256: `cae542a9237c76810f59bb64e9ce50dc21804432455e878a5a6bf43692aaeed4`.

The package excludes questions, generated answers, gold answers, Judge text
and usage, API credentials, stdout/stderr, failure snapshots, and
machine-local launch state. The memories themselves are research artifacts and
must remain in the private repository.

Extract on macOS with:

```bash
tar -xzf docs/research/longmemeval-error-memories-50-20260815.tar.gz
```

The extracted memories are intended for frozen-memory retrieval and Evidence
Ledger experiments. Do not rewrite their checkpoints or use the package as a
Writer recovery target.
