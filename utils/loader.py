import numpy as np
import pandas as pd
from pathlib import Path
from .common import INPUT_DIR

def load(filepath: str | Path,
         verbose: bool = True):
    """
    Safe load csv file from filepath
    """
    filepath = INPUT_DIR / filepath
    if not filepath.exists():
        raise FileNotFoundError(f"File doesn't exist: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Fail to read file: {e}")
    
    if verbose:
        print(f"Loaded: {df.shape} from {filepath}")
    
    return df
    