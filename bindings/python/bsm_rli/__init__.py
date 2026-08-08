"""
BSM-RLI: Bare-Metal Symbolic Micro-Kernel Engine Python Bindings
"""

import ctypes
import os

__version__ = "0.1.0"

class BSMRLEngine:
    def __init__(self, lib_path: str = None):
        if lib_path is None:
            # Search candidate paths
            candidates = [
                os.path.join(os.getcwd(), "build", "libbsm_rli.so"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build", "libbsm_rli.so")
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    lib_path = candidate
                    break

        if lib_path and os.path.exists(lib_path):
            self.lib = ctypes.CDLL(lib_path)
        else:
            self.lib = None

    def is_available(self) -> bool:
        return self.lib is not None
