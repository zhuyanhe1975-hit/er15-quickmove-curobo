from __future__ import annotations

import importlib
import sys


def main() -> int:
    print(f"python={sys.executable}")
    missing = []
    for name in ["torch", "numpy", "yaml", "setuptools_scm", "curobo"]:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            print(f"{name}=MISSING ({exc!r})")
            missing.append(name)
        else:
            print(f"{name}={getattr(mod, '__version__', 'ok')}")
    if missing:
        print("missing=" + ",".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

