import unittest
import tempfile
import json
import os
from engine_core.config.params_loader import ParamsLoader

class TestParamsLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary params file
        self.base_params = {
            "general": {
                "param1": 10,
                "param2": "test",
                "sub": {
                    "a": 1.0
                }
            },
            "list_param": [1, 2, 3],
            "numeric": 5.5
        }
        
        self.test_dir = tempfile.TemporaryDirectory()
        self.params_path = os.path.join(self.test_dir.name, "PARAMS.json")
        with open(self.params_path, 'w') as f:
            json.dump(self.base_params, f)
            
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_load_defaults(self):
        loader = ParamsLoader(self.params_path)
        self.assertEqual(loader.get("general", "param1"), 10)
        self.assertEqual(loader.get("list_param"), [1, 2, 3])
        
    def test_override_replace_scalar(self):
        overrides = {
            "general": {
                "param1": 20
            }
        }
        loader = ParamsLoader(self.params_path, overrides=overrides)
        self.assertEqual(loader.get("general", "param1"), 20)
        self.assertEqual(loader.get("general", "param2"), "test") # Preserved
        
    def test_override_replace_list(self):
        overrides = {
            "list_param": [4, 5]
        }
        loader = ParamsLoader(self.params_path, overrides=overrides)
        self.assertEqual(loader.get("list_param"), [4, 5])
        
    def test_override_deep_merge(self):
        overrides = {
            "general": {
                "sub": {
                    "a": 2.0
                }
            }
        }
        loader = ParamsLoader(self.params_path, overrides=overrides)
        self.assertEqual(loader.get("general", "sub", "a"), 2.0)
        self.assertEqual(loader.get("general", "param1"), 10) # Preserved
        
    def test_strict_unknown_key(self):
        overrides = {
            "unknown": 1
        }
        with self.assertRaises(KeyError):
            ParamsLoader(self.params_path, overrides=overrides, strict=True)
            
    def test_non_strict_unknown_key(self):
        overrides = {
            "unknown": 1
        }
        loader = ParamsLoader(self.params_path, overrides=overrides, strict=False)
        self.assertEqual(loader.get("unknown"), 1)
        
    def test_strict_type_mismatch(self):
        overrides = {
            "general": {
                "param1": "string_instead_of_int"
            }
        }
        with self.assertRaises(TypeError):
            ParamsLoader(self.params_path, overrides=overrides, strict=True)
            
    def test_numeric_type_compatibility(self):
        # int overriding float
        overrides = {
            "numeric": 6 # Base is 5.5 (float)
        }
        loader = ParamsLoader(self.params_path, overrides=overrides, strict=True)
        self.assertEqual(loader.get("numeric"), 6)
        
        # float overriding int
        overrides2 = {
            "general": {
                "param1": 10.5 # Base is 10 (int)
            }
        }
        loader = ParamsLoader(self.params_path, overrides=overrides2, strict=True)
        self.assertEqual(loader.get("general", "param1"), 10.5)

if __name__ == '__main__':
    unittest.main()

