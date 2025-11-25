"""Load and access engine parameters from base_params.json"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import warnings


class ParamsLoader:
    """Single source of truth for engine parameters"""
    
    def __init__(self, params_path: Optional[str] = None, base_path: Optional[Path] = None, overrides_path: Optional[Path] = None, overrides: Dict[str, Any] = None, strict: bool = True):
        # Legacy support: if params_path is provided, use it
        if params_path is not None:
            params_file = Path(params_path)
        elif base_path is not None:
            params_file = Path(base_path)
        else:
            # Default to base_params.json in config directory
            params_file = Path(__file__).parent / "base_params.json"
        
        with open(params_file, 'r') as f:
            self._params = json.load(f)
        
        # Apply overrides from file if provided
        if overrides_path is not None:
            with open(overrides_path, 'r') as f:
                file_overrides = json.load(f)
            self._params = self._deep_merge(self._params, file_overrides, strict=strict)
            
        if overrides:
            self._params = self._deep_merge(self._params, overrides, strict=strict)
    
    def _deep_merge(self, base: Any, override: Any, strict: bool = True, path: str = "") -> Any:
        """
        Recursive merge with strict type checking and validation.
        
        Rules:
        - dict + dict -> recursive merge
        - list in override -> REPLACE base list
        - scalar -> replace
        - unknown keys in strict mode -> raise KeyError
        - type mismatch -> raise TypeError (unless safe numeric cast)
        """
        if isinstance(base, dict) and isinstance(override, dict):
            # Copy base to avoid modifying original structure during recursion
            result = copy.deepcopy(base)
            for k, v in override.items():
                new_path = f"{path}.{k}" if path else k
                
                if k not in base:
                    if strict:
                        raise KeyError(f"Override key '{new_path}' does not exist in base params.")
                    else:
                        warnings.warn(f"Override key '{new_path}' does not exist in base params. Adding it.")
                        result[k] = v
                else:
                    result[k] = self._deep_merge(base[k], v, strict=strict, path=new_path)
            return result
        
        # Handle type mismatch for non-dict overrides (lists and scalars)
        if not isinstance(override, type(base)):
            # Allow int <-> float conversions if both are numeric
            if isinstance(base, (int, float)) and isinstance(override, (int, float)):
                pass # Compatible numeric types
            elif base is None or override is None:
                # Allow None to override or be overridden if logic permits
                pass 
            else:
                 msg = f"Type mismatch at '{path}': expected {type(base).__name__}, got {type(override).__name__}"
                 if strict:
                     raise TypeError(msg)
                 else:
                     warnings.warn(msg)
        
        # For lists and scalars, replace entirely
        return override
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested parameter value using dot notation or tuple of keys"""
        value = self._params
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    def get_default(self, *keys: str) -> Any:
        """Get default value from parameter (handles nested dicts with 'default' key)"""
        value = self.get(*keys)
        if isinstance(value, dict) and 'default' in value:
            return value['default']
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all parameters"""
        return copy.deepcopy(self._params)
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of parameters used (for reporting)"""
        return copy.deepcopy(self._params)
