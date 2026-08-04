# Third-party repositories

`manifest.json` records the source URL, exact commit, and whether each local
checkout had uncommitted changes when the manifest was generated. Existing
checkouts are included when the complete project directory is copied.

To reconstruct a missing checkout, clone its `remote`, then check out its
recorded `commit`. A `dirty: true` entry means that the checkout contains local
changes that cannot be reconstructed from the upstream commit alone; preserve
that directory when transferring the project.

The `environments` directory contains package inventories exported from removed
virtual environments. Create new environments on the destination machine;
virtual environments contain absolute paths and are not portable.
