from .molecular_visualizer import MolecularAttentionVisualizer
from .flow_visualizer import AttentionFlowVisualizer

class CombinedAttentionVisualizer:
    """Combines all visualization approaches into a unified interface"""
    
    def __init__(self, base_path):
        self.molecular_viz = MolecularAttentionVisualizer(base_path)
        self.flow_viz = AttentionFlowVisualizer(base_path)
    
    def run_full_analysis(self, sample_id=None):
        """Run all available visualizations"""
        print("Running molecular attention analysis...")
        self.molecular_viz.run_analysis(sample_id)
        
        print("\nRunning attention flow analysis...")
        self.flow_viz.run_comprehensive_analysis(sample_id)
        
        print("\n🎨 All visualizations completed successfully!")
        print("Check the output directories for your results.")