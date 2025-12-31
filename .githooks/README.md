Git hooks

This folder is a suggested place to store Git hook scripts if you prefer to use `core.hooksPath`.

To enable hooks from this folder instead of the default `.git/hooks` use:

```bash
git config core.hooksPath .githooks
```

Note: this repo uses `pre-commit` for hook management. Run `pre-commit install` to set up hooks automatically.
