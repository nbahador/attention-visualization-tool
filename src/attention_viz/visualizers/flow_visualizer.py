import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from scipy.stats import entropy
from matplotlib.animation import FuncAnimation
from .base_visualizer import BaseVisualizer

class AttentionFlowVisualizer(BaseVisualizer):
    """Visualizes attention flow dynamics and patterns"""
    
    def create_attention_river_plot(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        heads_per_layer = max([k[1] for k in attention_data.keys()]) + 1
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        flow_data = []
        for layer in layers:
            layer_attentions = []
            for head in range(heads_per_layer):
                if (layer, head) in attention_data:
                    att_matrix = attention_data[(layer, head)]
                    max_attention = np.max(att_matrix)
                    entropy_attention = entropy(att_matrix.flatten() + 1e-10)
                    sparsity = np.sum(att_matrix > 0.1) / att_matrix.size
                    layer_attentions.append([max_attention, entropy_attention, sparsity])
                else:
                    layer_attentions.append([0, 0, 0])
            flow_data.append(layer_attentions)
        
        x_positions = np.linspace(0, 10, len(layers))
        colors = plt.cm.viridis(np.linspace(0, 1, heads_per_layer))
        
        for head in range(heads_per_layer):
            y_vals = [flow_data[i][head][0] * 100 if i < len(flow_data) else 0 
                     for i in range(len(layers))]
            width_vals = [flow_data[i][head][2] * 50 if i < len(flow_data) else 0 
                         for i in range(len(layers))]
            
            for i in range(len(layers)-1):
                x_curve = np.linspace(x_positions[i], x_positions[i+1], 100)
                y_curve = np.interp(x_curve, [x_positions[i], x_positions[i+1]], 
                                   [y_vals[i], y_vals[i+1]])
                width_curve = np.interp(x_curve, [x_positions[i], x_positions[i+1]], 
                                       [width_vals[i], width_vals[i+1]])
                
                ax.fill_between(x_curve, y_curve - width_curve/2, y_curve + width_curve/2,
                               alpha=0.6, color=colors[head], 
                               label=f'Head {head}' if i == 0 else "")
        
        ax.set_xlabel('Layers', fontsize=14)
        ax.set_ylabel('Attention Flow Intensity', fontsize=14)
        ax.set_title(f'Attention River Flow - Sample {sample_id}', fontsize=16, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_path / "flow_maps" / f"attention_river_sample_{sample_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_3d_attention_landscape(self, attention_data, sample_id):
        fig = go.Figure()
        
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        heads_per_layer = max([k[1] for k in attention_data.keys()]) + 1
        
        colors = px.colors.qualitative.Set3
        
        for i, layer in enumerate(layers):
            z_data = []
            for head in range(heads_per_layer):
                if (layer, head) in attention_data:
                    att_matrix = attention_data[(layer, head)]
                    if att_matrix.shape[0] > 50:
                        step = att_matrix.shape[0] // 50
                        att_matrix = att_matrix[::step, ::step]
                    z_data.append(att_matrix)
                else:
                    z_data.append(np.zeros((10, 10)))
            
            if z_data:
                combined_matrix = np.mean(z_data, axis=0)
                x = np.arange(combined_matrix.shape[1])
                y = np.arange(combined_matrix.shape[0])
                X, Y = np.meshgrid(x, y)
                
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=combined_matrix + i * 0.5,
                    colorscale='Viridis',
                    opacity=0.8,
                    name=f'Layer {layer}',
                    showscale=i == 0
                ))
        
        fig.update_layout(
            title=f'3D Attention Landscape - Sample {sample_id}',
            scene=dict(
                xaxis_title='Token Position',
                yaxis_title='Token Position', 
                zaxis_title='Attention + Layer Offset',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(str(self.save_path / "3d_visualizations" / f"attention_landscape_sample_{sample_id}.html"))
    
    def create_attention_network_graph(self, attention_data, sample_id, threshold=0.1):
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        
        for idx, layer in enumerate(layers[:6]):
            ax = axes[idx]
            
            layer_matrices = []
            for head in range(12):
                if (layer, head) in attention_data:
                    layer_matrices.append(attention_data[(layer, head)])
            
            if layer_matrices:
                avg_matrix = np.mean(layer_matrices, axis=0)
                G = nx.Graph()
                n_tokens = min(avg_matrix.shape[0], 30)
                
                for i in range(n_tokens):
                    G.add_node(i, pos=(np.cos(2*np.pi*i/n_tokens), np.sin(2*np.pi*i/n_tokens)))
                
                for i in range(n_tokens):
                    for j in range(i+1, n_tokens):
                        weight = (avg_matrix[i, j] + avg_matrix[j, i]) / 2
                        if weight > threshold:
                            G.add_edge(i, j, weight=weight)
                
                pos = nx.get_node_attributes(G, 'pos')
                weights = [G[u][v]['weight'] for u, v in G.edges()]
                
                if weights:
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, 
                                         node_color='lightblue', alpha=0.7)
                    nx.draw_networkx_edges(G, pos, ax=ax, width=np.array(weights)*10,
                                         alpha=0.5, edge_color=weights, 
                                         edge_cmap=plt.cm.viridis)
                
                ax.set_title(f'Layer {layer} Attention Network', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                ax.axis('off')
        
        plt.suptitle(f'Attention Network Graphs - Sample {sample_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_path / "network_graphs" / f"attention_networks_sample_{sample_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_heatmap_grid(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        heads_per_layer = max([k[1] for k in attention_data.keys()]) + 1
        
        fig, axes = plt.subplots(len(layers), heads_per_layer, 
                                figsize=(heads_per_layer*3, len(layers)*2.5))
        
        if len(layers) == 1:
            axes = axes.reshape(1, -1)
        if heads_per_layer == 1:
            axes = axes.reshape(-1, 1)
            
        for i, layer in enumerate(layers):
            for j in range(heads_per_layer):
                ax = axes[i, j]
                
                if (layer, j) in attention_data:
                    att_matrix = attention_data[(layer, j)]
                    if att_matrix.shape[0] > 100:
                        step = att_matrix.shape[0] // 100
                        att_matrix = att_matrix[::step, ::step]
                    
                    im = ax.imshow(att_matrix, cmap='viridis', aspect='auto')
                    ax.set_title(f'L{layer}H{j}', fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle(f'Attention Heatmap Grid - Sample {sample_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_path / "attention_visualizations" / f"heatmap_grid_sample_{sample_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_evolution_animation(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            layer = layers[frame % len(layers)]
            
            layer_matrices = []
            for head in range(12):
                if (layer, head) in attention_data:
                    layer_matrices.append(attention_data[(layer, head)])
            
            if layer_matrices:
                avg_matrix = np.mean(layer_matrices, axis=0)
                if avg_matrix.shape[0] > 50:
                    step = avg_matrix.shape[0] // 50
                    avg_matrix = avg_matrix[::step, ::step]
                
                im = ax.imshow(avg_matrix, cmap='plasma', animated=True, aspect='auto')
                ax.set_title(f'Layer {layer} - Attention Evolution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Token Position')
                ax.set_ylabel('Token Position')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        anim = FuncAnimation(fig, animate, frames=len(layers), interval=1000, repeat=True)
        anim.save(str(self.save_path / "animations" / f"attention_evolution_sample_{sample_id}.gif"), 
                 writer='pillow', fps=1)
        plt.close()
    
    def create_attention_statistics_dashboard(self, attention_data, sample_id):
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=['Max Attention per Layer', 'Attention Entropy', 'Sparsity Pattern',
                           'Head Diversity', 'Token Importance', 'Layer Correlation',
                           'Attention Distribution', 'Pattern Consistency', 'Flow Dynamics'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        
        max_attentions = []
        entropies = []
        sparsities = []
        
        for layer in layers:
            layer_stats = {'max': [], 'entropy': [], 'sparsity': []}
            for head in range(12):
                if (layer, head) in attention_data:
                    matrix = attention_data[(layer, head)]
                    layer_stats['max'].append(np.max(matrix))
                    layer_stats['entropy'].append(entropy(matrix.flatten() + 1e-10))
                    layer_stats['sparsity'].append(np.sum(matrix > 0.1) / matrix.size)
            
            max_attentions.append(np.mean(layer_stats['max']) if layer_stats['max'] else 0)
            entropies.append(np.mean(layer_stats['entropy']) if layer_stats['entropy'] else 0)
            sparsities.append(np.mean(layer_stats['sparsity']) if layer_stats['sparsity'] else 0)
        
        fig.add_trace(go.Scatter(x=layers, y=max_attentions, mode='lines+markers', 
                                name='Max Attention', line=dict(color='red', width=3)), 
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=layers, y=entropies, mode='lines+markers', 
                                name='Entropy', line=dict(color='blue', width=3)), 
                     row=1, col=2)
        
        fig.add_trace(go.Scatter(x=layers, y=sparsities, mode='lines+markers', 
                                name='Sparsity', line=dict(color='green', width=3)), 
                     row=1, col=3)
        
        fig.update_layout(height=1200, showlegend=False, 
                         title_text=f"Attention Statistics Dashboard - Sample {sample_id}")
        
        fig.write_html(str(self.save_path / "statistical_analysis" / f"statistics_dashboard_sample_{sample_id}.html"))
    
    def run_comprehensive_analysis(self, sample_id=None):
        print("Loading attention data...")
        attention_data, metadata, sample_id = self.load_attention_data(sample_id)
        
        if not attention_data:
            print("No attention data found!")
            return
            
        print(f"Analyzing sample {sample_id} with {len(attention_data)} attention matrices...")
        
        print("Creating attention river plot...")
        self.create_attention_river_plot(attention_data, sample_id)
        
        print("Creating 3D attention landscape...")
        self.create_3d_attention_landscape(attention_data, sample_id)
        
        print("Creating attention network graphs...")
        self.create_attention_network_graph(attention_data, sample_id)
        
        print("Creating heatmap grid...")
        self.create_attention_heatmap_grid(attention_data, sample_id)
        
        print("Creating attention evolution animation...")
        self.create_attention_evolution_animation(attention_data, sample_id)
        
        print("Creating statistics dashboard...")
        self.create_attention_statistics_dashboard(attention_data, sample_id)
        
        print(f"All visualizations saved to: {self.save_path}")