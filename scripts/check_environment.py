"""Print the versions required to reproduce this project."""

import importlib
import platform
import sys


PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib", "aeon", "pytest", "yaml"]

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
for name in PACKAGES:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'installed')}")
