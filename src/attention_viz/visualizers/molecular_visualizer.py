import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from scipy.stats import entropy
from pathlib import Path
import os
import json
import glob
import warnings
from .base_visualizer import BaseVisualizer

class MolecularAttentionVisualizer(BaseVisualizer):
    """Visualizes molecular attention patterns with specialized analyses"""
    
    def visualize_position_attention(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        max_len = max(m.shape[0] for m in attention_data.values())
        
        pos_attention, pos_variance, pos_entropy = self._calculate_position_stats(attention_data, layers, max_len)
        
        self._create_position_static_plot(pos_attention, pos_variance, pos_entropy, layers, max_len, sample_id)
        self._create_position_interactive_plot(pos_attention, pos_variance, pos_entropy, layers, max_len, sample_id)
    
    def _calculate_position_stats(self, attention_data, layers, max_len):
        pos_attention = np.zeros((len(layers), max_len))
        pos_variance = np.zeros((len(layers), max_len))
        pos_entropy = np.zeros((len(layers), max_len))
        
        for i, layer in enumerate(layers):
            layer_mats = [m for (l, h), m in attention_data.items() if l == layer]
            if not layer_mats:
                continue
                
            stacked = np.stack([m for m in layer_mats if m.shape[0] == max_len])
            avg_attention = np.mean(stacked, axis=0)
            
            pos_attention[i, :] = np.mean(avg_attention, axis=1)
            pos_variance[i, :] = np.var(avg_attention, axis=1)
            pos_entropy[i, :] = np.array([entropy(row) for row in avg_attention])
        
        return pos_attention, pos_variance, pos_entropy
    
    def _create_position_static_plot(self, pos_attention, pos_variance, pos_entropy, layers, max_len, sample_id):
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[0.95, 0.05],
                            hspace=0.5, wspace=0.1)
        
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(pos_attention, aspect='auto', cmap=self.mol_colors,
                       interpolation='gaussian', vmin=np.percentile(pos_attention, 5),
                       vmax=np.percentile(pos_attention, 95))
        
        ax1.set_xticks(np.arange(-0.5, max_len, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
        ax1.grid(which='minor', color='w', linestyle='-', linewidth=1.5, alpha=0.3)
        
        ax1.set_title(f'Molecular Position-wise Attention - Sample {sample_id}', 
                     fontsize=16, pad=20, fontweight='bold')
        ax1.set_ylabel('Layer', fontsize=14)
        ax1.set_yticks(range(len(layers)))
        ax1.set_yticklabels([f'L{l}' for l in layers], fontsize=12)
        ax1.set_xlabel('Position in Molecular Formula', fontsize=14)
        
        ax2 = fig.add_subplot(gs[1, 0])
        avg_variance = np.mean(pos_variance, axis=0)
        ax2.fill_between(range(max_len), avg_variance, color=self.focus_colors[1], alpha=0.3)
        ax2.plot(range(max_len), avg_variance, color=self.focus_colors[1], linewidth=3, alpha=0.8)
        
        top_positions = np.argsort(avg_variance)[-3:][::-1]
        for pos in top_positions:
            ax2.plot(pos, avg_variance[pos], 'o', markersize=10, 
                    color='#e63946', alpha=0.8)
            ax2.text(pos, avg_variance[pos]*1.1, f'Pos {pos}', 
                    ha='center', fontsize=10, color='#e63946')
        
        ax2.set_title('Attention Variance by Position', fontsize=14, pad=15)
        ax2.set_xlabel('Position in Formula', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor('#f8f9fa')
        
        ax3 = fig.add_subplot(gs[2, 0])
        avg_entropy = np.mean(pos_entropy, axis=0)
        ax3.fill_between(range(max_len), avg_entropy, color='#457b9d', alpha=0.3)
        ax3.plot(range(max_len), avg_entropy, color='#457b9d', linewidth=3, alpha=0.8)
        
        top_entropy = np.argsort(avg_entropy)[-3:][::-1]
        for pos in top_entropy:
            ax3.plot(pos, avg_entropy[pos], 'o', markersize=10, 
                    color='#1d3557', alpha=0.8)
            ax3.text(pos, avg_entropy[pos]*1.1, f'Pos {pos}', 
                    ha='center', fontsize=10, color='#1d3557')
        
        ax3.set_title('Attention Entropy by Position', fontsize=14, pad=15)
        ax3.set_xlabel('Position in Formula', fontsize=12)
        ax3.set_ylabel('Entropy', fontsize=12)
        ax3.grid(True, alpha=0.2)
        ax3.set_facecolor('#f8f9fa')
        
        cax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=25, fontsize=12)
        
        fig.text(0.02, 0.98, 'H', fontsize=24, color='#1a759f', alpha=0.3)
        fig.text(0.96, 0.02, 'O', fontsize=24, color='#ff9e00', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_path / "position_analysis" / f"position_attention_{sample_id}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_position_interactive_plot(self, pos_attention, pos_variance, pos_entropy, layers, max_len, sample_id):
        fig = go.Figure()
        
        plotly_colorscale = self._convert_cmap_to_plotly(self.mol_colors)

        fig.add_trace(go.Heatmap(
            z=pos_attention,
            x=list(range(max_len)),
            y=[f'Layer {l}' for l in layers],
            colorscale=plotly_colorscale,
            colorbar=dict(title='Attention Weight'),
            hoverongaps=False,
            hovertemplate='Layer: %{y}<br>Position: %{x}<br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(max_len)),
            y=np.mean(pos_variance, axis=0),
            mode='lines+markers',
            name='Variance',
            line=dict(color=self.focus_colors[1], width=3),
            marker=dict(size=8, color=self.focus_colors[1]),
            yaxis='y2',
            hovertemplate='Position: %{x}<br>Variance: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(max_len)),
            y=np.mean(pos_entropy, axis=0),
            mode='lines+markers',
            name='Entropy',
            line=dict(color='#457b9d', width=3),
            marker=dict(size=8, color='#457b9d'),
            yaxis='y3',
            hovertemplate='Position: %{x}<br>Entropy: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Molecular Position-wise Attention Analysis - Sample {sample_id}',
            xaxis=dict(title='Position in Molecular Formula'),
            yaxis=dict(title='Layer', domain=[0.6, 1.0]),
            yaxis2=dict(title='Variance', domain=[0.3, 0.55], anchor='x'),
            yaxis3=dict(title='Entropy', domain=[0.0, 0.25], anchor='x'),
            height=800,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(t=100, b=100),
            showlegend=True
        )
        
        top_positions = np.argsort(np.mean(pos_variance, axis=0))[-3:][::-1]
        top_entropy = np.argsort(np.mean(pos_entropy, axis=0))[-3:][::-1]
        
        for pos in top_positions:
            fig.add_annotation(
                x=pos,
                y=np.mean(pos_variance, axis=0)[pos],
                yref='y2',
                text=f'Top Var Pos {pos}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color='#e63946')
            )
            
        for pos in top_entropy:
            fig.add_annotation(
                x=pos,
                y=np.mean(pos_entropy, axis=0)[pos],
                yref='y3',
                text=f'Top Entropy Pos {pos}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color='#1d3557')
            )
        
        fig.write_html(str(self.save_path / "interactive_plots" / f"position_attention_{sample_id}.html"))
    
    def visualize_head_specialization(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        heads = sorted(set([k[1] for k in attention_data.keys()]))
        max_len = max(m.shape[0] for m in attention_data.values())
        
        self._create_head_specialization_grid(attention_data, layers, heads, max_len, sample_id)
        self._create_head_specialization_3d(attention_data, sample_id)
    
    def _create_head_specialization_grid(self, attention_data, layers, heads, max_len, sample_id):
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 3*len(layers) + 2), facecolor='white')
        gs = fig.add_gridspec(len(layers) + 1, len(heads) + 1, 
                            height_ratios=[1]*len(layers) + [0.1],
                            width_ratios=[1]*len(heads) + [0.05],
                            hspace=0.5, wspace=0.3)

        all_vals = np.concatenate([m.flatten() for m in attention_data.values()])
        vmin, vmax = np.percentile(all_vals, [5, 95])

        for (layer, head), matrix in attention_data.items():
            row = layers.index(layer)
            col = heads.index(head)
            ax = fig.add_subplot(gs[row, col])
        
            if matrix.shape[0] < max_len:
                pad_width = ((0, max_len - matrix.shape[0]), (0, max_len - matrix.shape[0]))
                matrix = np.pad(matrix, pad_width, mode='constant')
        
            im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f'L{layer}H{head}', fontsize=10, pad=5)
        
            ax.plot(range(max_len), range(max_len), color='red', linewidth=0.5, alpha=0.5)
        
            if row == len(layers)-1:
                ax.set_xlabel('Position', fontsize=9)
            if col == 0:
                ax.set_ylabel('Layer', fontsize=9)
                
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_facecolor('#f8f9fa')

        cax = fig.add_subplot(gs[:-1, -1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=12)

        fig.suptitle(f'Attention Head Specialization - Sample {sample_id}', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_path / "head_specialization" / f"head_specialization_{sample_id}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_head_specialization_3d(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        heads_per_layer = {layer: sorted([h for (l, h) in attention_data.keys() if l == layer]) 
                          for layer in layers}

        n_layers = len(layers)
        max_heads_per_layer = max(len(heads) for heads in heads_per_layer.values())

        n_cols = min(3, max_heads_per_layer)
        n_rows = min(2, n_layers)
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'is_3d': True} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=[f'Layer {layer} Head {head}' 
                           for layer in layers 
                           for head in heads_per_layer[layer]][:n_rows*n_cols],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )

        plot_idx = 1

        for layer in layers[:n_rows]:
            for head in heads_per_layer[layer][:n_cols]:
                if plot_idx > n_rows*n_cols:
                    break
                
                matrix = attention_data[(layer, head)]
                x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
            
                row = ((plot_idx-1) // n_cols) + 1
                col = ((plot_idx-1) % n_cols) + 1
            
                fig.add_trace(
                    go.Surface(
                        x=x, y=y, z=matrix,
                        name=f'L{layer}H{head}',
                        showscale=plot_idx == 1,
                        colorscale='Viridis',
                        opacity=0.9,
                        hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}'
                    ),
                    row=row, col=col
                )
            
                fig.update_scenes(
                    aspectratio=dict(x=1, y=1, z=0.7),
                    camera_eye=dict(x=1.5, y=1.5, z=0.6),
                    row=row, col=col
                )
            
                plot_idx += 1

        fig.update_layout(
            title=f'3D Head Specialization - Sample {sample_id}',
            height=900 if n_rows == 1 else 1200,
            width=1500,
            margin=dict(l=50, r=50, b=50, t=80),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            template='plotly_white'
        )

        total_heads = sum(len(heads) for heads in heads_per_layer.values())
        if plot_idx <= total_heads:
            fig.add_annotation(
                text=f"Showing first {n_rows*n_cols} of {total_heads} heads",
                xref="paper", yref="paper",
                x=0.5, y=-0.1, showarrow=False
            )

        fig.write_html(str(self.save_path / "3d_visualizations" / f"head_specialization_3d_{sample_id}.html"))
    
    def visualize_attention_focus(self, attention_data, sample_id):
        focus_data = []
        
        for (layer, head), matrix in attention_data.items():
            n = matrix.shape[0]
            if n < 4:
                continue
                
            try:
                begin_focus = np.mean(matrix[:, :max(1, n//4)])
                end_focus = np.mean(matrix[:, -max(1, n//4):])
                mid_focus = np.mean(matrix[:, max(1, n//4):-max(1, n//4)])
                
                focus_data.append({
                    'Layer': layer,
                    'Head': head,
                    'Begin Focus': begin_focus,
                    'End Focus': end_focus,
                    'Mid Focus': mid_focus,
                    'Specialization': np.argmax([begin_focus, mid_focus, end_focus])
                })
            except Exception as e:
                print(f"Error processing matrix L{layer}H{head}: {str(e)}")
                continue
            
        if not focus_data:
            print("No valid focus data to visualize")
            return
            
        df = pd.DataFrame(focus_data)
        
        fig = px.parallel_categories(
            df,
            dimensions=['Layer', 'Head', 'Specialization'],
            color='Specialization',
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={
                'Layer': 'Layer',
                'Head': 'Head',
                'Specialization': 'Focus Region'
            }
        )
        
        fig.update_layout(
            title=f'Attention Head Focus Specialization - Sample {sample_id}',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html(str(self.save_path / "head_specialization" / f"focus_specialization_{sample_id}.html"))
        
        fig = go.Figure()
        
        for i, region in enumerate(['Begin Focus', 'Mid Focus', 'End Focus']):
            fig.add_trace(go.Bar(
                x=[f'L{row["Layer"]}H{row["Head"]}' for _, row in df.iterrows()],
                y=df[region],
                name=region.replace('_', ' '),
                marker_color=self.focus_colors[i],
                opacity=0.8,
                width=0.4
            ))
        
        fig.update_layout(
            barmode='group',
            title=f'Attention Focus Regions - Sample {sample_id}',
            xaxis_title='Attention Head',
            yaxis_title='Average Attention Weight',
            height=600,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        fig.write_html(str(self.save_path / "interactive_plots" / f"focus_regions_{sample_id}.html"))
    
    def create_attention_flow_animation(self, attention_data, sample_id):
        layers = sorted(set([k[0] for k in attention_data.keys()]))
        max_len = max(m.shape[0] for m in attention_data.values())
        
        avg_matrices = []
        for layer in layers:
            layer_mats = [m for (l, h), m in attention_data.items() if l == layer]
            if layer_mats:
                padded = [np.pad(m, ((0, max_len - m.shape[0]), (0, max_len - m.shape[0]))) 
                         for m in layer_mats if m is not None and len(m.shape) == 2]
                if padded:
                    avg_matrices.append(np.mean(padded, axis=0))
        
        if not avg_matrices:
            print("No valid matrices for animation")
            return
            
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 8), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[0.7, 0.3])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            layer = frame % len(layers)
            matrix = avg_matrices[layer]
            
            im = ax1.imshow(matrix, cmap='viridis', aspect='auto',
                          vmin=np.percentile(np.concatenate(avg_matrices), 5),
                          vmax=np.percentile(np.concatenate(avg_matrices), 95))
            ax1.set_title(f'Layer {layers[layer]} Attention Patterns', fontsize=12)
            ax1.set_xlabel('To Position', fontsize=10)
            ax1.set_ylabel('From Position', fontsize=10)
            
            pos_attention = np.mean(matrix, axis=1)
            ax2.bar(range(len(pos_attention)), pos_attention, color=self.focus_colors[2])
            ax2.set_title('Average Attention by Position', fontsize=12)
            ax2.set_xlabel('Position in Formula', fontsize=10)
            ax2.set_ylabel('Attention Weight', fontsize=10)
            ax2.set_ylim(0, np.max([np.max(np.mean(m, axis=1)) for m in avg_matrices]) * 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#f8f9fa')
            
            fig.suptitle(f'Attention Flow - Sample {sample_id} - Layer {layers[layer]}', 
                        fontsize=14, y=0.98)
            
            return im,
        
        anim = FuncAnimation(fig, update, frames=len(layers), interval=1000, blit=False)
        anim.save(str(self.save_path / "animations" / f"attention_flow_{sample_id}.gif"), 
                 writer='pillow', fps=1, dpi=150)
        plt.close()
    
    def create_token_interaction_network(self, attention_data, sample_id, layer=None, head=None):
        if layer is None or head is None:
            layer, head = self._find_most_distinctive_head(attention_data)
        
        if (layer, head) not in attention_data:
            print(f"Layer {layer} Head {head} not found in data")
            return
            
        matrix = attention_data[(layer, head)]
        n_tokens = min(matrix.shape[0], 50)
        
        G = nx.DiGraph()
        threshold = np.percentile(matrix.flatten(), 90)
        
        for i in range(n_tokens):
            G.add_node(i, pos=(i, 0))
        
        for i in range(n_tokens):
            for j in range(n_tokens):
                if matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=matrix[i, j])
        
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 6), facecolor='white')
        ax = fig.add_subplot(111)
        
        pos = nx.get_node_attributes(G, 'pos')
        weights = [G[u][v]['weight']*10 for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='#3498db', alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color='#e74c3c', 
                              alpha=0.6, arrowstyle='->', arrowsize=15)
        
        label_nodes = [0, n_tokens//4, n_tokens//2, 3*n_tokens//4, n_tokens-1]
        labels = {n: f"Token {n}" for n in label_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        ax.set_title(f'Token Interaction Network - Layer {layer} Head {head}\nSample {sample_id}', 
                    fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.save_path / "interactive_plots" / f"token_network_L{layer}H{head}_{sample_id}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self._create_interactive_network(G, layer, head, sample_id)
    
    def _find_most_distinctive_head(self, attention_data):
        max_entropy = -1
        best_head = (0, 0)
        
        for (layer, head), matrix in attention_data.items():
            if matrix.size == 0:
                continue
                
            try:
                current_entropy = entropy(matrix.flatten())
                if current_entropy > max_entropy:
                    max_entropy = current_entropy
                    best_head = (layer, head)
            except:
                continue
                
        return best_head
    
    def _create_interactive_network(self, G, layer, head, sample_id):
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Interactive Token Network - Layer {layer} Head {head}<br>Sample {sample_id}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           template='plotly_white')
                       )
        
        fig.write_html(str(self.save_path / "interactive_plots" / f"interactive_network_L{layer}H{head}_{sample_id}.html"))
    
    def run_analysis(self, sample_id=None):
        try:
            print("Loading attention data...")
            attention_data, metadata, sample_id = self.load_attention_data(sample_id)
            
            if not attention_data:
                print("No attention data found!")
                return
                
            print(f"Analyzing sample {sample_id} with {len(attention_data)} attention matrices...")
            
            print("1. Visualizing position-wise attention...")
            self.visualize_position_attention(attention_data, sample_id)
            
            print("2. Analyzing head specialization...")
            self.visualize_head_specialization(attention_data, sample_id)
            
            print("3. Visualizing attention focus regions...")
            self.visualize_attention_focus(attention_data, sample_id)
            
            print("4. Creating attention flow animation...")
            self.create_attention_flow_animation(attention_data, sample_id)
            
            print("5. Creating token interaction networks...")
            self.create_token_interaction_network(attention_data, sample_id)
            
            print(f"\nAnalysis complete! Results saved to: {self.save_path}")
            print("Visualizations created:")
            print("- Position-wise attention heatmaps")
            print("- Head specialization matrices")
            print("- 3D head specialization plots")
            print("- Attention focus region analysis")
            print("- Attention flow animation")
            print("- Token interaction networks")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise