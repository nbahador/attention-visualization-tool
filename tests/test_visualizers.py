import pytest
import os
import numpy as np
from pathlib import Path
from src.visualizers.base_visualizer import BaseVisualizer
from src.visualizers.molecular_visualizer import MolecularAttentionVisualizer
from src.visualizers.flow_visualizer import AttentionFlowVisualizer

@pytest.fixture
def sample_attention_data():
    # Create a temporary directory with sample attention files
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample attention files
    for layer in range(2):
        for head in range(2):
            matrix = np.random.rand(10, 10)
            np.savetxt(test_dir / f"attention_epoch_1.0_sample_123_L{layer}H{head}.csv", matrix, delimiter=",")
    
    yield test_dir
    
    # Clean up
    for f in test_dir.glob("*.csv"):
        f.unlink()
    test_dir.rmdir()

def test_base_visualizer_load_data(sample_attention_data):
    viz = BaseVisualizer(sample_attention_data)
    attention_data, metadata, sample_id = viz.load_attention_data()
    
    assert len(attention_data) == 4  # 2 layers x 2 heads
    assert sample_id == "123"
    for key in [(0,0), (0,1), (1,0), (1,1)]:
        assert key in attention_data
        assert attention_data[key].shape == (10, 10)

def test_molecular_visualizer(sample_attention_data):
    viz = MolecularAttentionVisualizer(sample_attention_data)
    viz.run_analysis(sample_id="123")
    
    # Check if output files were created
    output_dir = sample_attention_data.parent / "visualizations"
    assert (output_dir / "position_analysis" / "position_attention_123.png").exists()
    assert (output_dir / "head_specialization" / "head_specialization_123.png").exists()
    
    # Clean up
    for f in output_dir.glob("**/*"):
        if f.is_file():
            f.unlink()
    output_dir.rmdir()

def test_flow_visualizer(sample_attention_data):
    viz = AttentionFlowVisualizer(sample_attention_data)
    viz.run_comprehensive_analysis(sample_id="123")
    
    # Check if output files were created
    output_dir = sample_attention_data.parent / "visualizations"
    assert (output_dir / "flow_maps" / "attention_river_sample_123.png").exists()
    assert (output_dir / "3d_visualizations" / "attention_landscape_sample_123.html").exists()
    
    # Clean up
    for f in output_dir.glob("**/*"):
        if f.is_file():
            f.unlink()
    output_dir.rmdir()