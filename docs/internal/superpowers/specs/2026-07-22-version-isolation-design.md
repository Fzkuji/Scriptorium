# NativeMem Version Isolation

## Goal

Separate V8, V10, and V11 implementation code so each version can be read,
changed, and tested without entering another version's module. Keep one stable
adapter entry point and shared benchmark infrastructure.

## Structure

```text
src/nativemem_versions/
  common.py
  v8/
    memory.py
    adapter.py
  v10/
    memory.py
    adapter.py
  v11/
    memory.py
    adapter.py

src/adapters/run_nativemem.py
```

`run_nativemem.py` contains only common CLI handling and version dispatch.
Each version adapter owns that version's build and retrieval behavior. Each
version memory module owns its prompts, file representation, and maintenance
logic.

`common.py` contains only code already shared by at least two versions, such as
conversation parsing and source-turn indexing. Version-specific configuration
must not be placed there.

## Dependency Direction

```text
benchmark runner -> run_nativemem -> selected version adapter
selected version adapter -> its memory module + common runtime
```

Version packages must not import another version package. V10 may initially
reuse V8 behavior through a small compatibility import only where V10 is
explicitly defined as the frozen V8.8 method; this dependency must be visible
inside `v10/adapter.py`, not hidden in the dispatcher.

## Compatibility

Existing commands continue importing `src.adapters.run_nativemem`. Historical
public symbols remain as temporary aliases when tests or scripts import them.
No benchmark result directories, checkpoints, or running processes are moved.

Build records use version-specific configuration fields: `v10_config` for V10
and `v11_config` for V11.

## Migration Order

1. Add package-level contract tests for version dispatch and import isolation.
2. Move V11 code and its adapter first.
3. Move V10 code and configuration.
4. Move V8 code last because most historical scripts depend on its symbols.
5. Reduce `run_nativemem.py` to dispatch and compatibility aliases.

Every step must keep existing targeted tests passing. Compatibility aliases are
removed only after repository-wide reference search shows no callers.

## Out of Scope

- Changing memory semantics or prompts.
- Moving experiment results.
- Rewriting LongMemEval checkpoint or retry behavior.
- Creating separate Git branches for each version.
- Removing historical benchmark scripts during this refactor.
