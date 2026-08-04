# Baselines

The external memory systems this work is compared against, and the code that
drives them.

| | |
|---|---|
| `third_party/` | One git checkout per system: Mem0, Zep, A-Mem, MemOS, and sixteen more. Not committed; restore with `manifest.json`. |
| `adapters/` | One module per system, each a runnable entry point that builds memory and answers questions through that system's own API. |
| `generate_third_party_manifest.py` | Records every checkout's remote and commit into `third_party/manifest.json`. |

## Running a controlled comparison

The paper's main table cites each system's own reported numbers, which were
produced under different answerers, prompts and judges. The packages below exist
to replace those with numbers measured under one shared condition, should a
reviewer ask.

| | |
|---|---|
| `controlled_locomo/` | One shared answerer after retrieval, with the memory-to-prompt boundary enforced so dataset gold, evidence and category cannot reach the final prompt |
| `locomo_baselines/` | Retrieval-only LoCoMo rows: assemble inputs, run, audit |
| `longmemeval_m1/` | The same comparison on LongMemEval |
| `beam_controls/` | BEAM control conditions |
| `token_budget/` | Holds compared systems to an equal visible-token budget |
| `gateways/` | Budget-enforcing HTTP proxies that cap what a comparison run can spend |
| `m4_statistics/` | Paired significance tests over a frozen comparison inventory |
| `human_agreement/` | Blinded annotation packet, and human-versus-judge agreement over it |
| `readonly_control/` | Artifact auditor that re-derives results without producing them |

Within one package the prefix gives the role: `*_contract.py` fixes the shape,
`run_*.py` produces, `audit_*.py` re-derives with separate code and fails on
mismatch, and `freeze_*.py` pins inputs once audits pass. An auditor never
shares code with the runner it checks.

These are held apart from the method itself: none of them import `src/`, and
`scripts/run_experiment.sh` never calls them.

Regenerate the manifest after changing a checkout, so a reported baseline number
stays traceable to the exact commit that produced it:

```bash
python baselines/generate_third_party_manifest.py
```

Each system pins its own dependencies. Their old virtual environments are not
portable and are replaced by frozen package inventories under
`third_party/environments`.
