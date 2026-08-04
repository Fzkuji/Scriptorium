# Baselines

The external memory systems this work is compared against, and the code that
drives them.

| | |
|---|---|
| `third_party/` | One git checkout per system: Mem0, Zep, A-Mem, MemOS, and sixteen more. Not committed; restore with `manifest.json`. |
| `adapters/` | One module per system, each a runnable entry point that builds memory and answers questions through that system's own API. |
| `generate_third_party_manifest.py` | Records every checkout's remote and commit into `third_party/manifest.json`. |

Regenerate the manifest after changing a checkout, so a reported baseline number
stays traceable to the exact commit that produced it:

```bash
python baselines/generate_third_party_manifest.py
```

Each system pins its own dependencies. Their old virtual environments are not
portable and are replaced by frozen package inventories under
`third_party/environments`.
