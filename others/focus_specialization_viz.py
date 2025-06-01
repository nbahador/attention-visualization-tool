import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import json
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FocusSpecializationVisualizer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.save_path = Path(r"Enter the path")
        (self.save_path / "head_specialization").mkdir(exist_ok=True)
        
        # Color setup
        self.head_colors = plt.cm.tab20
        self.focus_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

    def load_attention_data(self, sample_id="5388"):
        """Load attention maps for the specified sample"""
        attention_data = {}
        metadata = {}

        pattern = self.base_path / f"attention_epoch_15.0_sample_{sample_id}_L*H*.csv"
        files = glob.glob(str(pattern))

        for file in files:
            try:
                parts = Path(file).stem.split('_')
                layer_head = parts[-1]
                layer = int(layer_head[1:layer_head.index('H')])
                head = int(layer_head[layer_head.index('H')+1:])
                
                # Load attention matrix
                att_matrix = pd.read_csv(file, index_col=0).values
                attention_data[(layer, head)] = att_matrix
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        return attention_data, metadata, sample_id

    def visualize_attention_focus(self, attention_data, sample_id):
        """Create focus specialization visualization"""
        focus_data = []
        
        for (layer, head), matrix in attention_data.items():
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
                
            n = matrix.shape[0]
            if n < 4:  # Skip if matrix is too small
                continue
                
            try:
                # Calculate focus on different regions
                begin_focus = np.mean(matrix[:, :max(1, n//4)])  # First quarter
                end_focus = np.mean(matrix[:, -max(1, n//4):])   # Last quarter
                mid_focus = np.mean(matrix[:, max(1, n//4):-max(1, n//4)])  # Middle half
                self_focus = np.mean(np.diag(matrix))  # Self-attention
                
                focus_data.append({
                    'Layer': layer,
                    'Head': head,
                    'Begin Focus': begin_focus,
                    'End Focus': end_focus,
                    'Mid Focus': mid_focus,
                    'Self Focus': self_focus,
                    'Specialization': np.argmax([begin_focus, mid_focus, end_focus, self_focus])
                })
            except Exception as e:
                print(f"Error processing matrix L{layer}H{head}: {str(e)}")
                continue
        
        if not focus_data:
            print("No valid focus data to visualize")
            return
            
        df = pd.DataFrame(focus_data)
        
        # Create enhanced parallel coordinates plot
        fig = px.parallel_coordinates(
            df,
            dimensions=['Layer', 'Head', 'Begin Focus', 'Mid Focus', 'End Focus', 'Self Focus'],
            color='Specialization',
            color_continuous_scale=px.colors.diverging.Tealrose,
            labels={
                'Layer': 'Layer',
                'Head': 'Head',
                'Begin Focus': 'Begin Focus',
                'Mid Focus': 'Mid Focus',
                'End Focus': 'End Focus',
                'Self Focus': 'Self Focus',
                'Specialization': 'Specialization'
            }
        )
        
        fig.update_layout(
            title=f'Attention Head Focus Specialization - Sample {sample_id}',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html(str(self.save_path / "head_specialization" / f"focus_specialization_{sample_id}.html"))
    
    def run_focus_analysis(self, sample_id="5388"):
        """Run only the focus specialization analysis"""
        try:
            print(f"Loading attention data for sample {sample_id}...")
            attention_data, metadata, sample_id = self.load_attention_data(sample_id)
            
            if not attention_data:
                print("No attention data found!")
                return
                
            print("Creating focus specialization visualization...")
            self.visualize_attention_focus(attention_data, sample_id)
            
            print(f"\nFocus specialization visualization saved to: {self.save_path / 'head_specialization'}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    # Set up paths
    base_path = r"Enter the path"
    attention_path = r"Enter the path"
    
    if not os.path.exists(attention_path):
        print(f"Attention maps path not found: {attention_path}")
        print("Using base path as fallback...")
        attention_path = base_path
    
    # Create visualizer and run focus analysis
    visualizer = FocusSpecializationVisualizer(attention_path)
    
    try:
        visualizer.run_focus_analysis(sample_id="5388")
        print("\n🎉 Focus specialization visualization complete!")
    except Exception as e:
        print(f"\n❌ Error during visualization: {str(e)}")