# Baselines

The external memory systems this work is compared against, and the code that
drives them.

| | |
|---|---|
| `third_party/` | One git checkout per system: Mem0, Zep, A-Mem, MemOS, and sixteen more. Not committed; restore with `manifest.json`. |
| `adapters/` | One module per system, each a runnable entry point that builds memory and answers questions through that system's own API. |
| `generate_third_party_manifest.py` | Records every checkout's remote and commit into `third_party/manifest.json`. |

## Running a comparison

The paper's main table cites each system's own reported numbers, which were
produced under different answerers, prompts and judges. To measure them under
one shared condition instead:

```bash
baselines/run_comparison.sh --list          # what can be run
baselines/run_comparison.sh 0 mem0 zep naive
```

Each system retrieves with its own API. Every system's retrieved memories then
go through the same answerer and the same judge
(`scripts/evaluation/evaluate.py`), so the only thing that differs between rows
is retrieval.

Gold answers cannot leak into a prompt, because the answerer's entry point is
`generate_answer(question, memories)` — the dataset's answer is not one of its
arguments and never enters that scope.

Results land in `results/comparison-s<N>/<system>/`, holding `retrieved.json`
and `eval.json`. A system whose run dies partway is reported and skipped; the
others still finish. Success means the evaluation file contains judged records,
not merely that it exists — `evaluate.py` writes partial output as it goes, so
a file being present proves nothing. `read_score.py` makes that distinction and
is what the summary and the skip logic both use.

Each system pins its own dependencies, and most need an API key. A system with
neither installed will fail at the retrieval step and be skipped.

Regenerate the manifest after changing a checkout, so a reported baseline number
stays traceable to the exact commit that produced it:

```bash
python baselines/generate_third_party_manifest.py
```

Each system pins its own dependencies. Their old virtual environments are not
portable and are replaced by frozen package inventories under
`third_party/environments`.
