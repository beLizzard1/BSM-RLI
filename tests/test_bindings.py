import os
import unittest
from bsm_rli import BSMRLEngine

class TestBSMRLIBindings(unittest.TestCase):
    def test_engine_init(self):
        engine = BSMRLEngine()
        self.assertTrue(engine.is_available(), "BSM-RLI C++ dynamic library libbsm_rli.so not found")

if __name__ == "__main__":
    unittest.main()
