"""bovw-forensics: Bag of Visual Words pipeline.

Stages: extract -> cache -> vocab -> encode -> eval.
Each stage is a plain function over numpy arrays so it can be swapped or
tested in isolation.
"""

__version__ = "0.1.0"
