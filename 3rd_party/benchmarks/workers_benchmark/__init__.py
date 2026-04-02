"""MOSAIC Worker Overhead Benchmark.

Two-layer benchmark inspired by rl-tools' paper methodology:

  Layer 1 (horizontal): CleanRL native vs XuanCe native
    -> "Which framework is fastest on its own?"

  Layer 2 (vertical):  For each framework: native vs worker vs fastlane
    -> "What overhead does MOSAIC add?"
"""

__version__ = "2.0.0"
