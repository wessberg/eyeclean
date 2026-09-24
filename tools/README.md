# tools/

- `release_tree.py` scans what the public repository would contain for private material.  It looks for:
  - development footage references;
  - local absolute paths;
  - personal names and e-mail addresses;
  - credentials;
  - large files.

  Run it before every commit:

  ```
  python tools/release_tree.py --tracked            # scan the files git would commit
  python tools/release_tree.py ../eyeclean-release  # also copy them to a clean tree
  ```

Training drivers will be added here as a curated, config-driven recipe; see
[docs/RELEASE.md](../docs/RELEASE.md) §4.
