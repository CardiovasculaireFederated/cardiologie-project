"""
Serialization utilities for data saving and loading.
"""

import logging
import json
from pathlib import Path
from typing import Union, Any

import pandas as pd
import numpy as np

import io
import torch
from typing import Dict


logger = logging.getLogger(__name__)


def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    compression: str = None
) -> None:
    """
    Save dataframe to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the file
        index: Whether to save index (default: False)
        compression: Compression mode ('infer', 'gzip', 'bz2', 'zip', 'xz', None)
        
    Raises:
        IOError: If save fails
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(
            filepath,
            index=index,
            compression=compression
        )
        logger.info(f"✓ Saved {len(df)} rows to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save dataframe to {filepath}: {str(e)}")
        raise


def load_dataframe(
    filepath: Union[str, Path],
    index_col: int = None
) -> pd.DataFrame:
    """
    Load dataframe from CSV file.
    
    Args:
        filepath: Path to CSV file
        index_col: Column to use as index
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=index_col)
        logger.info(f"✓ Loaded {len(df)} rows from {filepath}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dataframe from {filepath}: {str(e)}")
        raise


def save_json(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path where to save the file
        indent: JSON indentation level
        
    Raises:
        IOError: If save fails
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"✓ Saved JSON to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {str(e)}")
        raise


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded JSON from {filepath}")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {str(e)}")
        raise


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    
    def default(self, obj):
        """Handle numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


def weights_to_bytes(weights: Dict) -> bytes:
    """
    Serialize PyTorch model weights (state_dict) to bytes.
    Used for Kafka transmission in federated learning.
    """
    try:
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        buffer.seek(0)
        return buffer.read()

    except Exception as e:
        logger.error(f"Failed to serialize model weights: {str(e)}")
        raise

def bytes_to_weights(bytes_data: bytes) -> Dict:
    """
    Deserialize bytes back to PyTorch model weights (state_dict).
    Used when receiving weights from Kafka.
    """
    try:
        buffer = io.BytesIO(bytes_data)
        weights = torch.load(buffer, map_location="cpu")
        return weights

    except Exception as e:
        logger.error(f"Failed to deserialize model weights: {str(e)}")
        raise