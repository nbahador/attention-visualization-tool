import os
import json
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import entropy

warnings.filterwarnings('ignore')

class BaseVisualizer:
    """Base class with common functionality for all visualizers"""
    def __init__(self, base_path=None):
        """
        Initialize visualizer with optional base path for attention data.
        If no base_path is provided, will look in current directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.save_path = Path("attention_viz_outputs")  # Outputs in current directory
        self._setup_directories()
        self._setup_colormaps()
        self._setup_styles()
        
    def _setup_directories(self):
        """Create all necessary output directories in current working directory"""
        self.save_path.mkdir(exist_ok=True)
        
        dirs = [
            "position_analysis", "head_specialization", "3d_visualizations",
            "interactive_plots", "animations", "attention_visualizations",
            "flow_maps", "network_graphs", "statistical_analysis"
        ]
        
        for d in dirs:
            (self.save_path / d).mkdir(exist_ok=True)
        
        print(f"All visualizations will be saved to: {self.save_path.absolute()}")
    
    def _setup_colormaps(self):
        """Initialize all color maps with better perceptual properties"""
        self.pos_colors = LinearSegmentedColormap.from_list('pos_colors', 
            ['#1a1334', '#26294a', '#01545a', '#017351', '#03c383', 
             '#aad962', '#fbbf45', '#ef6a32', '#ed0345'])
        
        self.head_colors = plt.cm.tab20
        self.focus_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
        self.dynamics_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
        
        self.mol_colors = LinearSegmentedColormap.from_list('mol_colors', 
            ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', 
             '#d8576b', '#ed7953', '#fb9f3a', '#fdca26'])
    
    def _setup_styles(self):
        """Set consistent plotting styles"""
        plt.style.use('ggplot')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _convert_cmap_to_plotly(self, cmap, n_colors=256):
        """Convert matplotlib colormap to Plotly colorscale"""
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
        return [[i/(n_colors-1), f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'] 
                for i, c in enumerate(colors)]
    
    def load_attention_data(self, sample_id=None):
        """
        Robust attention data loader with multiple fallback methods.
        Looks for data in either:
        - The provided base_path (if specified)
        - The current working directory (if base_path=None)
        """
        if sample_id is None:
            files = glob.glob(str(self.base_path / "attention_epoch_*_sample_*_L*H*.csv"))
            if files:
                sample_id = files[0].split('_sample_')[1].split('_')[0]

        attention_data = {}
        metadata = {}

        pattern = self.base_path / f"attention_epoch_*_sample_{sample_id}_L*H*.csv"
        files = glob.glob(str(pattern))

        for file in files:
            try:
                parts = Path(file).stem.split('_')
                layer_head = parts[-1]
                layer = int(layer_head[1:layer_head.index('H')])
                head = int(layer_head[layer_head.index('H')+1:])
                
                att_matrix = self._load_matrix_with_fallbacks(file)
                if att_matrix is None:
                    continue
                
                attention_data[(layer, head)] = att_matrix
                
                meta_file = file.replace('.csv', '_meta.json')
                if os.path.exists(meta_file):
                    metadata[(layer, head)] = self._load_metadata(meta_file)
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        return attention_data, metadata, sample_id
    
    def _load_matrix_with_fallbacks(self, file):
        """Try multiple methods to load attention matrix"""
        methods = [
            self._load_matrix_standard,
            self._load_matrix_alternative_params,
            self._load_matrix_manual_parse,
        ]
        
        for method in methods:
            try:
                matrix = method(file)
                if matrix is not None:
                    return matrix
            except:
                continue
        return None
    
    def _load_matrix_standard(self, file):
        """Standard CSV loading"""
        return pd.read_csv(file, index_col=0).values
    
    def _load_matrix_alternative_params(self, file):
        return pd.read_csv(file, sep=',', encoding='utf-8', index_col=0, on_bad_lines='skip').values
    
    def _load_matrix_manual_parse(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines[1:]:
            line = line.strip().replace('"', '').replace("'", "")
            if line and ',' in line:
                parts = line.split(',')
                try:
                    row_data = [float(x) for x in parts[1:] if x.strip()]
                    if row_data:
                        data_lines.append(row_data)
                except ValueError:
                    continue
        
        if data_lines:
            max_cols = max(len(row) for row in data_lines)
            matrix_data = [row + [0.0] * (max_cols - len(row)) for row in data_lines]
            return np.array(matrix_data)
        return None
    
    def _load_metadata(self, meta_file):
        try:
            with open(meta_file, 'r') as f:
                content = f.read().strip()
                content = (content.replace("'", '"')
                          .replace("True", "true").replace("False", "false")
                          .replace("None", "null"))
                return json.loads(content)
        except json.JSONDecodeError:
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    return json.loads(content[start:end])
            except:
                pass  # Silently ignore JSON recovery failures
        except Exception as e:
            pass  # Silently ignore all other metadata loading errors
        return None