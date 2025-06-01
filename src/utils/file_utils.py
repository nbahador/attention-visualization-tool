import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_attention_matrix(file_path):
    """Helper function to load attention matrix with error handling"""
    try:
        return pd.read_csv(file_path, index_col=0).values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def save_visualization(fig, save_path, filename, dpi=300):
    """Helper function to save visualizations"""
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    fig.savefig(save_path / filename, dpi=dpi, bbox_inches='tight')