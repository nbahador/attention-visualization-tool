import os
import tarfile
import json
import numpy as np
import torch
from tqdm import tqdm
import warnings
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase.db import connect
from transformers import AutoModelWithLMHead, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from transformers import DefaultFlowCallback
from PIL import Image
import io
import math

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
custom_temp_dir = r"Enter the path"
dataset_path = r"Enter the path to train_4M.tar.gz"
results_dir = r"Enter the path"
os.makedirs(results_dir, exist_ok=True)

# Initialize visualization directories
vis_dirs = {
    'pre_train': os.path.join(results_dir, "pre_training_visualization"),
    'training': os.path.join(results_dir, "training_metrics"),
    'post_train': os.path.join(results_dir, "post_training_analysis"),
    'molecules': os.path.join(results_dir, "molecule_visualizations"),
    'latent': os.path.join(results_dir, "latent_representations"),
    'attention': os.path.join(results_dir, "attention_maps")
}
for dir_path in vis_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Load pretrained model and tokenizer
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add special tokens for chemical properties
special_tokens = ["[FORMULA]", "[ENERGY]", "[CHARGE]", "[SPIN]", "[HOMO]", "[LUMO]", "[GAP]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model = AutoModelWithLMHead.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

class ChemicalDataset(Dataset):
    def __init__(self, db_path, tokenizer, max_length=256, sample_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_and_process_data(db_path, sample_size)
        self.save_dataset_summary()
        self.visualize_pre_training()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.create_chemical_description(item)
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze(),
            "original_text": text,
            "smiles": item['smiles'],
            "formula": item['formula']
        }
    
    def load_and_process_data(self, db_path, sample_size=None):
        """Load and process all molecules from the ASE database"""
        print("Loading molecules from database...")
        data = []
        
        with connect(db_path) as db:
            total_molecules = len(db)
            print(f"Total molecules in database: {total_molecules}")
            
            # Use all molecules if sample_size is None
            if sample_size is None:
                indices = range(total_molecules)
                print("Processing all molecules...")
            else:
                indices = np.random.choice(total_molecules, size=min(sample_size, total_molecules), replace=False)
                print(f"Processing {len(indices)} molecules...")
            
            for i in tqdm(indices, desc="Processing molecules"):
                try:
                    row = db.get(i + 1)  # ASE database indexing starts from 1
                    processed_mol = self.process_molecule(row)
                    if processed_mol:
                        data.append(processed_mol)
                except Exception as e:
                    print(f"Error processing molecule {i}: {e}")
                    continue
        
        print(f"Successfully processed {len(data)} molecules")
        return data
    
    def process_molecule(self, row):
        """Process a single molecule from the database"""
        try:
            # Get atomic structure
            atoms = row.toatoms()
            positions = atoms.positions
            atomic_numbers = atoms.numbers
            
            # Convert to SMILES
            smiles = self.convert_to_smiles(atoms)
            if not smiles:
                return None
            
            # Get properties from key_value_pairs
            properties = row.key_value_pairs if hasattr(row, 'key_value_pairs') else {}
            
            # Create chemical formula
            formula = atoms.get_chemical_formula()
            
            # Extract energy and other properties
            energy = properties.get('energy', 0.0)
            charge = properties.get('charge', 0)
            spin = properties.get('spin', 0)
            homo = properties.get('homo', 0.0)
            lumo = properties.get('lumo', 0.0)
            gap = lumo - homo if (homo != 0.0 and lumo != 0.0) else 0.0
            
            return {
                'smiles': smiles,
                'formula': formula,
                'energy': energy,
                'charge': charge,
                'spin': spin,
                'homo': homo,
                'lumo': lumo,
                'gap': gap,
                'num_atoms': len(atoms),
                'positions': positions.tolist(),
                'atomic_numbers': atomic_numbers.tolist()
            }
            
        except Exception as e:
            print(f"Error processing molecule: {e}")
            return None
    
    def convert_to_smiles(self, atoms):
        """Convert ASE atoms object to SMILES string"""
        try:
            # Create RDKit molecule from atoms
            mol = Chem.RWMol()
            
            # Add atoms
            for atomic_num in atoms.numbers:
                atom = Chem.Atom(int(atomic_num))
                mol.AddAtom(atom)
            
            # Add bonds based on distance
            positions = atoms.positions
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    
                    # Get covalent radii
                    r1 = self.get_covalent_radius(atoms.numbers[i])
                    r2 = self.get_covalent_radius(atoms.numbers[j])
                    
                    # Bond threshold (sum of covalent radii + tolerance)
                    bond_threshold = (r1 + r2) * 1.2
                    
                    if dist < bond_threshold:
                        bond_order = self.detect_bond_order(dist, r1 + r2)
                        mol.AddBond(i, j, bond_order)
            
            # Sanitize and get SMILES
            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                return smiles
            except:
                # If sanitization fails, try with basic sanitization
                try:
                    mol = Chem.RWMol(mol)
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
                    smiles = Chem.MolToSmiles(mol)
                    return smiles
                except:
                    return None
                    
        except Exception as e:
            return None
    
    def detect_bond_order(self, distance, ideal_distance):
        """Detect bond order based on distance"""
        ratio = distance / ideal_distance
        if ratio < 0.8:
            return Chem.BondType.TRIPLE
        elif ratio < 0.9:
            return Chem.BondType.DOUBLE
        else:
            return Chem.BondType.SINGLE
    
    def get_covalent_radius(self, atomic_number):
        """Get covalent radius for an element"""
        # Covalent radii in Angstroms
        covalent_radii = {
            1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 0.99
        }
        return covalent_radii.get(atomic_number, 1.0)
    
    def create_chemical_description(self, item):
        """Create a text description of the chemical compound"""
        description = f"[FORMULA] {item['formula']} "
        description += f"SMILES: {item['smiles']} "
        description += f"[ENERGY] {item['energy']:.4f} "
        description += f"[CHARGE] {item['charge']} "
        description += f"[SPIN] {item['spin']} "
        
        if item['homo'] != 0.0:
            description += f"[HOMO] {item['homo']:.4f} "
        if item['lumo'] != 0.0:
            description += f"[LUMO] {item['lumo']:.4f} "
        if item['gap'] != 0.0:
            description += f"[GAP] {item['gap']:.4f} "
            
        description += f"Atoms: {item['num_atoms']}"
        
        return description
    
    def save_dataset_summary(self):
        """Save dataset statistics and examples"""
        print("Saving dataset summary...")
        
        summary = {
            "total_molecules": len(self.data),
            "elements_distribution": self.get_elements_distribution(),
            "properties_summary": self.get_properties_summary(),
            "sample_descriptions": [self.create_chemical_description(item) for item in self.data[:10]]
        }
        
        with open(os.path.join(vis_dirs['pre_train'], "dataset_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save to Excel (sample due to size)
        sample_data = self.data[:10000] if len(self.data) > 10000 else self.data
        df = pd.DataFrame(sample_data)
        df.to_excel(os.path.join(vis_dirs['pre_train'], "dataset_sample.xlsx"), index=False)
        
        print(f"Dataset summary saved. Total molecules: {len(self.data)}")
    
    def visualize_pre_training(self):
        """Generate pre-training visualizations"""
        print("Generating pre-training visualizations...")
        
        # 1. Draw sample molecules
        self.draw_sample_molecules()
        
        # 2. Property distributions
        self.plot_property_distributions()
        
        # 3. Dataset statistics
        self.plot_dataset_statistics()
    
    def draw_sample_molecules(self, num_samples=10):
        """Draw and save sample molecules"""
        print("Drawing sample molecules...")
        samples = self.data[:num_samples]
        
        for i, item in enumerate(samples):
            try:
                mol = Chem.MolFromSmiles(item['smiles'])
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    img.save(os.path.join(vis_dirs['molecules'], f"sample_{i+1}.png"))
            except Exception as e:
                print(f"Failed to draw molecule {i}: {e}")
    
    def plot_property_distributions(self):
        """Plot distributions of chemical properties"""
        print("Plotting property distributions...")
        
        properties = ['energy', 'charge', 'spin', 'homo', 'lumo', 'gap', 'num_atoms']
        df = pd.DataFrame(self.data)
        
        # Filter out zero values for HOMO/LUMO/GAP
        for prop in ['homo', 'lumo', 'gap']:
            if prop in df.columns:
                df.loc[df[prop] == 0.0, prop] = np.nan
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, prop in enumerate(properties):
            if prop in df.columns and i < len(axes):
                ax = axes[i]
                # Remove NaN values for plotting
                data = df[prop].dropna()
                if len(data) > 0:
                    sns.histplot(data, kde=True, ax=ax)
                    ax.set_title(f'Distribution of {prop}')
                    ax.set_xlabel(prop)
        
        # Remove empty subplots
        for i in range(len(properties), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dirs['pre_train'], 'property_distributions.png'), dpi=300)
        plt.close()
    
    def plot_dataset_statistics(self):
        """Plot general dataset statistics"""
        print("Plotting dataset statistics...")
        
        # Element distribution
        elements_dist = self.get_elements_distribution()
        top_elements = dict(sorted(elements_dist.items(), key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_elements.keys(), top_elements.values())
        plt.title('Top 10 Most Common Elements')
        plt.xlabel('Element')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dirs['pre_train'], 'element_distribution.png'), dpi=300)
        plt.close()
    
    def get_elements_distribution(self):
        """Calculate element frequency distribution"""
        element_counts = defaultdict(int)
        
        for item in self.data:
            formula = item['formula']
            # Parse chemical formula
            i = 0
            while i < len(formula):
                if formula[i].isupper():
                    element = formula[i]
                    i += 1
                    
                    # Check for lowercase letter
                    if i < len(formula) and formula[i].islower():
                        element += formula[i]
                        i += 1
                    
                    # Check for number
                    num_str = ""
                    while i < len(formula) and formula[i].isdigit():
                        num_str += formula[i]
                        i += 1
                    
                    count = int(num_str) if num_str else 1
                    element_counts[element] += count
                else:
                    i += 1
        
        return dict(element_counts)
    
    def get_properties_summary(self):
        """Calculate summary statistics for properties"""
        properties = {}
        
        for prop in ['energy', 'charge', 'spin', 'homo', 'lumo', 'gap', 'num_atoms']:
            values = [item[prop] for item in self.data if prop in item and item[prop] != 0.0]
            if values:
                properties[prop] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return properties

class MetricSaverCallback(DefaultFlowCallback):
    def __init__(self):
        self.metrics_history = []
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            step_metrics = metrics.copy()
            step_metrics['step'] = state.global_step
            step_metrics['epoch'] = state.epoch
            self.metrics_history.append(step_metrics)
        
            # Create epoch-specific directory
            epoch_dir = os.path.join(vis_dirs['training'], f"epoch_{state.epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
        
            try:
                # Save metrics to Excel with proper file handling
                excel_path = os.path.join(epoch_dir, f"training_metrics_epoch_{state.epoch}.xlsx")
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                    pd.DataFrame(self.metrics_history).to_excel(writer, index=False)
            except Exception as e:
                print(f"Error saving metrics to Excel: {e}")
            
            # Plot training curves
            self.plot_training_curves(state.epoch)
        return control

    def plot_training_curves(self, epoch):
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
    
        plt.figure(figsize=(12, 5))
    
        # Loss curves
        plt.subplot(1, 2, 1)
        if 'train_loss' in df.columns:
            plt.plot(df['step'], df['train_loss'], label='Training Loss', alpha=0.7)
        if 'eval_loss' in df.columns:
            plt.plot(df['step'], df['eval_loss'], label='Validation Loss', alpha=0.7)
    
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # Learning rate
        plt.subplot(1, 2, 2)
        if 'learning_rate' in df.columns:
            plt.plot(df['step'], df['learning_rate'], label='Learning Rate', color='orange')
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
    
        # Save in epoch directory
        epoch_dir = os.path.join(vis_dirs['training'], f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        plt.savefig(os.path.join(epoch_dir, 'training_curves.png'), dpi=300)
        plt.close()

class VisualizationCallback(DefaultFlowCallback):
    def __init__(self, dataset, sample_size=100):
        super().__init__()
        self.dataset = dataset
        self.sample_size = min(sample_size, len(dataset))
        self.sample_indices = np.random.choice(len(dataset), size=self.sample_size, replace=False)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Save visualizations at the end of each epoch"""
        print(f"\nSaving visualizations for epoch {state.epoch}...")
        
        # Get the model from the trainer
        model = kwargs['model']
        device = next(model.parameters()).device
        
        # Create directory for this epoch
        epoch_dir = os.path.join(vis_dirs['attention'], f"epoch_{state.epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Process sample molecules
        for idx in tqdm(self.sample_indices, desc="Generating visualizations"):
            try:
                item = self.dataset[idx]
                inputs = {
                    'input_ids': item['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': item['attention_mask'].unsqueeze(0).to(device),
                    'output_attentions': True  # Request attention maps
                }
                
                # Get model outputs with attention
                outputs = model(**inputs)
                attentions = outputs.attentions  # Tuple of attention tensors for each layer
                
                # Save attention maps for each layer and head
                self.save_attention_maps(
                    attentions, 
                    item['original_text'], 
                    item['smiles'], 
                    item['formula'], 
                    state.epoch, 
                    idx, 
                    epoch_dir
                )
                
                # Save latent representations
                if state.epoch == 1 or state.epoch % 5 == 0 or state.epoch == args.num_train_epochs:
                    self.save_latent_representations(
                        outputs.hidden_states, 
                        item['original_text'], 
                        item['smiles'], 
                        item['formula'], 
                        state.epoch, 
                        idx, 
                        epoch_dir
                    )
                    
            except Exception as e:
                print(f"Error generating visualizations for sample {idx}: {e}")
                continue
        
        # After processing all samples, create cluster visualizations
        if state.epoch == 1 or state.epoch % 5 == 0 or state.epoch == args.num_train_epochs:
            self.visualize_latent_clusters(model, state.epoch)
    
    def save_attention_maps(self, attentions, original_text, smiles, formula, epoch, sample_idx, epoch_dir):
        """Save attention maps for all layers and heads as Excel files"""
        if not attentions:  # Check if attentions is empty
            print(f"No attention maps to save for sample {sample_idx}")
            return

        tokens = tokenizer.tokenize(original_text)

        # Create directory for CSV backups
        csv_dir = os.path.join(epoch_dir, "csv_backup")
        os.makedirs(csv_dir, exist_ok=True)

        # Debugging: Print model config
        print(f"\nDebugging attention maps for sample {sample_idx}")
        print(f"Number of attention layers: {len(attentions)}")
        print(f"Model config number of heads: {model.config.num_attention_heads}")

        # Save each layer and head combination
        for layer_idx, layer_attention in enumerate(attentions):
            try:
                # Debugging: Print attention tensor shape
                print(f"\nLayer {layer_idx} attention tensor shape (before squeeze): {layer_attention.shape}")
            
                layer_attention = layer_attention.squeeze(0).cpu().detach().numpy()
                num_heads = layer_attention.shape[0]
            
                # Debugging: Print number of heads in this layer
                print(f"Layer {layer_idx} has {num_heads} attention heads")
                print(f"Attention tensor shape after squeeze: {layer_attention.shape}")

                for head_idx in range(num_heads):
                    try:
                        head_attention = layer_attention[head_idx]
                    
                        # Debugging: Print head attention shape
                        print(f"  Head {head_idx} attention shape: {head_attention.shape}")
                    
                        # Verify we have square attention matrix
                        if head_attention.shape[0] != head_attention.shape[1]:
                            print(f"Warning: Non-square attention matrix for layer {layer_idx} head {head_idx}")
                            continue
                    
                        # Create DataFrame with proper token labels
                        df = pd.DataFrame(
                            head_attention,
                            index=tokens[:len(head_attention)],
                            columns=tokens[:len(head_attention)]
                        )

                        # Create filename for this head
                        filename = f"attention_epoch_{epoch}_sample_{sample_idx}_L{layer_idx}H{head_idx}"
                        csv_path = os.path.join(csv_dir, f"{filename}.csv")
                
                        # Save to CSV
                        df.to_csv(csv_path)
                
                        # Save metadata
                        metadata = {
                            'Layer': layer_idx,
                            'Head': head_idx,
                            'Formula': formula,
                            'SMILES': smiles,
                            'Sample ID': sample_idx,
                            'Epoch': epoch,
                            'Num_Tokens': len(tokens),
                            'Attention_Shape': head_attention.shape
                        }
                
                        with open(csv_path.replace('.csv', '_meta.json'), 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        print(f"  Successfully saved head {head_idx} attention map")

                    except Exception as e:
                        print(f"Error saving layer {layer_idx} head {head_idx}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

            except Exception as e:
                print(f"Error processing layer {layer_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Completed saving attention maps for sample {sample_idx} in {csv_dir}")
    
    def save_latent_representations(self, hidden_states, original_text, smiles, formula, epoch, sample_idx, epoch_dir):
        """Save latent representations as Excel files"""
        # Create a Pandas Excel writer
        excel_filename = os.path.join(
            epoch_dir, 
            f"latent_representations_sample_{sample_idx}.xlsx"
        )
    
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            for layer_idx, layer_states in enumerate(hidden_states):
                # layer_states shape: (batch_size, seq_len, hidden_size)
                layer_states = layer_states.squeeze(0).cpu().detach().numpy()
            
                # Reduce dimensionality for visualization
                reducer = umap.UMAP(n_components=2, random_state=42)
                reduced = reducer.fit_transform(layer_states)
            
                # Create DataFrame with coordinates
                df = pd.DataFrame(
                    reduced,
                    columns=['UMAP_1', 'UMAP_2']
                )
            
                # Add token information
                tokens = tokenizer.tokenize(original_text)
                df['Token'] = tokens[:len(reduced)] + [''] * (len(reduced) - len(tokens))
            
                # Add metadata
                df['Layer'] = layer_idx
                df['Formula'] = formula
                df['SMILES'] = smiles
                df['Sample_ID'] = sample_idx
            
                # Save to Excel sheet
                sheet_name = f"Layer_{layer_idx}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def visualize_latent_clusters(self, model, epoch):
        """Visualize latent space clusters with molecule thumbnails"""
        print(f"Generating latent space cluster visualization for epoch {epoch}...")

        # Get embeddings for our sample molecules
        embeddings = []
        smiles_list = []
        formulas = []
        clusters = []

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for idx in tqdm(self.sample_indices, desc="Extracting embeddings"):
                try:
                    item = self.dataset[idx]
                    inputs = {
                        'input_ids': item['input_ids'].unsqueeze(0).to(device),
                        'attention_mask': item['attention_mask'].unsqueeze(0).to(device)
                    }
        
                    outputs = model(**inputs, output_hidden_states=True)
                    cls_embedding = outputs.hidden_states[-1][0, 0].cpu().numpy()
                    embeddings.append(cls_embedding)
                    smiles_list.append(item['smiles'])
                    formulas.append(item['formula'])
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue

        if not embeddings:
            print("No embeddings extracted for clustering!")
            return

        embeddings = np.array(embeddings)

        # Cluster the embeddings
        n_clusters = min(6, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Reduce dimensionality for visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        # Create DataFrame for the visualization
        df = pd.DataFrame({
            'smiles': smiles_list,
            'formula': formulas,
            'cluster': clusters,
            'x': reduced[:, 0],
            'y': reduced[:, 1]
        })

        # Create a figure with two subplots
        fig = plt.figure(figsize=(24, 12))
        gs = GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig)

        # Subfigure 1: UMAP plot with clusters
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=30)
        ax1.set_title(f'Molecular Clusters in Latent Space (Epoch {epoch})', fontsize=14)
        ax1.set_xlabel('UMAP 1', fontsize=12)
        ax1.set_ylabel('UMAP 2', fontsize=12)

        # Add cluster legend with proper spacing
        legend = ax1.legend(*scatter.legend_elements(),
                           title="Clusters",
                           loc="upper right",
                           bbox_to_anchor=(1.25, 1),
                           frameon=True)
        ax1.add_artist(legend)

        # Add cluster centers
        cluster_centers = kmeans.cluster_centers_
        centers_reduced = reducer.transform(cluster_centers)
        ax1.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                   marker='x', s=200, c='black', linewidth=3, label='Centroids')

        # Subfigure 2: Molecule thumbnails organized by cluster
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')  # Turn off axes for the thumbnail display

        # Create a grid for molecule images (n_clusters rows, 5 columns)
        thumbnail_grid = GridSpec(n_clusters, 5, figure=fig)
        thumbnail_grid.update(left=0.55, right=0.95, wspace=0.2, hspace=0.4)

        # For each cluster, show representative molecules
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            sample_size = min(5, len(cluster_data))

            if sample_size == 0:
                continue

            # Get samples from different parts of the cluster
            samples = cluster_data.sample(n=sample_size, random_state=42)

            # Plot molecules for this cluster
            for j, (_, sample) in enumerate(samples.iterrows()):
                try:
                    mol = Chem.MolFromSmiles(sample['smiles'])
                    if mol:
                        # Draw molecule
                        img = Draw.MolToImage(mol, size=(200, 200))
        
                        # Create subplot for this molecule
                        img_ax = fig.add_subplot(thumbnail_grid[cluster_id, j])
                        img_ax.imshow(img)
                        img_ax.axis('off')
        
                        # Add formula information
                        img_ax.text(0.5, -0.1, sample['formula'], transform=img_ax.transAxes, 
                                  ha='center', va='top', fontsize=8)
                
                        # Add cluster label to first molecule in row
                        if j == 0:
                            img_ax.text(-0.3, 0.5, f'Cluster {cluster_id}', 
                                      transform=img_ax.transAxes, 
                                      ha='right', va='center', fontsize=12)
                except Exception as e:
                    print(f"Error drawing molecule for cluster {cluster_id}: {e}")
                    continue

        # Save in epoch directory
        cluster_dir = os.path.join(vis_dirs['latent'], f"epoch_{epoch}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_dir, 'molecular_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save cluster assignments
        df.to_excel(os.path.join(cluster_dir, 'cluster_assignments.xlsx'), index=False)

def generate_descriptions_targets_excel(dataset, results_dir):
    """Generate Excel file with chemical descriptions and corresponding targets"""
    print("Building chemical descriptions and targets dataset...")
    
    descriptions_data = []
    
    # Process all molecules to create description-target pairs
    for i, item in enumerate(tqdm(dataset.data, desc="Generating descriptions")):
        try:
            # Create the input description (what the model sees)
            input_description = dataset.create_chemical_description(item)
            
            # Create various target formats for different training objectives
            
            # Target 1: SMILES prediction
            smiles_target = item['smiles']
            
            # Target 2: Property prediction
            property_target = f"Energy: {item['energy']:.4f}, Charge: {item['charge']}, Spin: {item['spin']}"
            
            # Target 3: Formula prediction
            formula_target = item['formula']
            
            # Target 4: Complete description (for masked language modeling)
            complete_target = input_description
            
            # Target 5: Next token prediction format
            next_token_input = f"Molecule with formula {item['formula']} has"
            next_token_target = f"SMILES {item['smiles']} and energy {item['energy']:.4f}"
            
            # Target 6: Property regression targets (numerical)
            energy_target = item['energy']
            homo_target = item['homo'] if item['homo'] != 0.0 else None
            lumo_target = item['lumo'] if item['lumo'] != 0.0 else None
            gap_target = item['gap'] if item['gap'] != 0.0 else None
            
            descriptions_data.append({
                'molecule_id': i + 1,
                'input_description': input_description,
                'smiles': item['smiles'],
                'formula': item['formula'],
                'target_smiles': smiles_target,
                'target_properties': property_target,
                'target_formula': formula_target,
                'target_complete': complete_target,
                'next_token_input': next_token_input,
                'next_token_target': next_token_target,
                'energy': item['energy'],
                'charge': item['charge'],
                'spin': item['spin'],
                'homo': item['homo'] if item['homo'] != 0.0 else None,
                'lumo': item['lumo'] if item['lumo'] != 0.0 else None,
                'gap': item['gap'] if item['gap'] != 0.0 else None,
                'num_atoms': item['num_atoms'],
                'target_energy': energy_target,
                'target_homo': homo_target,
                'target_lumo': lumo_target,
                'target_gap': gap_target,
                'description_length': len(input_description),
                'smiles_length': len(item['smiles'])
            })
            
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(descriptions_data)
    
    # Save main dataset
    main_file = os.path.join(results_dir, "chemical_descriptions_targets.xlsx")
    
    # For very large datasets, save in chunks to avoid memory issues
    if len(df) > 100000:
        print(f"Large dataset detected ({len(df)} molecules). Saving in chunks...")
        
        # Save full dataset as CSV (more memory efficient)
        csv_file = os.path.join(results_dir, "chemical_descriptions_targets.csv")
        df.to_csv(csv_file, index=False)
        print(f"Full dataset saved as CSV: {csv_file}")
        
        # Save sample as Excel for easy viewing
        sample_df = df.sample(n=min(50000, len(df)), random_state=42)
        sample_df.to_excel(main_file, index=False, engine='openpyxl')
        print(f"Sample dataset (50k molecules) saved as Excel: {main_file}")
        
        # Save chunks as separate Excel files
        chunk_size = 50000
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_file = os.path.join(results_dir, f"chemical_descriptions_targets_chunk_{i//chunk_size + 1}.xlsx")
            chunk.to_excel(chunk_file, index=False, engine='openpyxl')
        
        print(f"Dataset saved in {(len(df)-1)//chunk_size + 1} chunks")
        
    else:
        # Save entire dataset as Excel
        df.to_excel(main_file, index=False, engine='openpyxl')
        print(f"Dataset saved as Excel: {main_file}")
    
    # Create summary statistics
    summary_stats = {
        'total_molecules': len(df),
        'average_description_length': df['description_length'].mean(),
        'average_smiles_length': df['smiles_length'].mean(),
        'energy_range': {
            'min': df['energy'].min(),
            'max': df['energy'].max(),
            'mean': df['energy'].mean(),
            'std': df['energy'].std()
        },
        'charge_distribution': df['charge'].value_counts().to_dict(),
        'spin_distribution': df['spin'].value_counts().to_dict(),
        'atoms_range': {
            'min': df['num_atoms'].min(),
            'max': df['num_atoms'].max(),
            'mean': df['num_atoms'].mean()
        }
    }
    
    # Save summary
    summary_file = os.path.join(results_dir, "descriptions_targets_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Create training format examples
    training_examples = []
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        training_examples.append({
            'example_id': i + 1,
            'input_text': row['input_description'],
            'target_smiles': row['target_smiles'],
            'target_properties': row['target_properties'],
            'next_token_format': {
                'input': row['next_token_input'],
                'target': row['next_token_target']
            }
        })
    
    examples_file = os.path.join(results_dir, "training_examples.json")
    with open(examples_file, "w") as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"Generated {len(df)} description-target pairs")
    print(f"Summary statistics saved to: {summary_stats}")
    print(f"Training examples saved to: {examples_file}")
    
    return df

def analyze_latent_space(model, tokenizer, dataset):
    """Analyze and visualize the latent space"""
    print("Analyzing latent space...")
    
    # Get embeddings for a subset of molecules (due to computational constraints)
    max_samples = min(2000, len(dataset))
    sample_indices = np.random.choice(len(dataset), size=max_samples, replace=False)
    embeddings = []
    smiles = []
    properties = []
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Extracting embeddings"):
            try:
                item = dataset[idx]
                inputs = {
                    'input_ids': item['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': item['attention_mask'].unsqueeze(0).to(device)
                }
                
                outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state of [CLS] token as representation
                cls_embedding = outputs.hidden_states[-1][0, 0].cpu().numpy()
                embeddings.append(cls_embedding)
                smiles.append(dataset.data[idx]['smiles'])
                properties.append({
                    'energy': dataset.data[idx]['energy'],
                    'charge': dataset.data[idx]['charge'],
                    'num_atoms': dataset.data[idx]['num_atoms']
                })
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    if not embeddings:
        print("No embeddings extracted!")
        return
    
    embeddings = np.array(embeddings)
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Dimensionality reduction
    visualize_embeddings(embeddings, smiles, properties, "PCA")
    visualize_embeddings(embeddings, smiles, properties, "UMAP")
    
    # Clustering
    cluster_embeddings(embeddings, smiles, properties)

def visualize_embeddings(embeddings, smiles, properties, method="PCA"):
    """Visualize embeddings using dimensionality reduction"""
    print(f"Creating {method} visualization...")
    
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    reduced = reducer.fit_transform(embeddings)
    
    # Color by different properties
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Basic scatter plot
    axes[0, 0].scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)
    axes[0, 0].set_title(f'{method} Projection of Molecular Embeddings')
    axes[0, 0].set_xlabel(f'{method} 1')
    axes[0, 0].set_ylabel(f'{method} 2')
    
    # Color by energy
    energies = [p['energy'] for p in properties]
    scatter = axes[0, 1].scatter(reduced[:, 0], reduced[:, 1], c=energies, cmap='viridis', alpha=0.6, s=10)
    axes[0, 1].set_title(f'{method} Colored by Energy')
    axes[0, 1].set_xlabel(f'{method} 1')
    axes[0, 1].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Color by charge
    charges = [p['charge'] for p in properties]
    scatter = axes[1, 0].scatter(reduced[:, 0], reduced[:, 1], c=charges, cmap='RdBu', alpha=0.6, s=10)
    axes[1, 0].set_title(f'{method} Colored by Charge')
    axes[1, 0].set_xlabel(f'{method} 1')
    axes[1, 0].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Color by number of atoms
    num_atoms = [p['num_atoms'] for p in properties]
    scatter = axes[1, 1].scatter(reduced[:, 0], reduced[:, 1], c=num_atoms, cmap='plasma', alpha=0.6, s=10)
    axes[1, 1].set_title(f'{method} Colored by Number of Atoms')
    axes[1, 1].set_xlabel(f'{method} 1')
    axes[1, 1].set_ylabel(f'{method} 2')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dirs['latent'], f'{method.lower()}_projection.png'), dpi=300)
    plt.close()
    
    # Save coordinates
    df = pd.DataFrame({
        'smiles': smiles,
        f'{method}_1': reduced[:, 0],
        f'{method}_2': reduced[:, 1],
        'energy': energies,
        'charge': charges,
        'num_atoms': num_atoms
    })
    df.to_excel(os.path.join(vis_dirs['latent'], f'{method.lower()}_coordinates.xlsx'), index=False)

def cluster_embeddings(embeddings, smiles, properties, n_clusters=8):
    """Cluster embeddings and visualize results with molecule thumbnails"""
    print("Performing clustering analysis...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Reduce dimensionality for visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    # Create a figure with GridSpec for better layout control
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[:2, :2])
    scatter = ax_main.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=30)
    ax_main.set_title('Molecular Clusters in Latent Space (UMAP)', fontsize=14)
    ax_main.set_xlabel('UMAP 1', fontsize=12)
    ax_main.set_ylabel('UMAP 2', fontsize=12)
    
    # Add cluster centers
    cluster_centers = kmeans.cluster_centers_
    centers_reduced = reducer.transform(cluster_centers)
    ax_main.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                   marker='x', s=200, c='black', linewidth=3, label='Centroids')
    
    # Create subplots for each cluster with molecule thumbnails
    cluster_axes = []
    for i in range(n_clusters):
        row = i // 2
        col = 2 + (i % 2)
        ax = fig.add_subplot(gs[row, col])
        cluster_axes.append(ax)
    
    # For each cluster, show representative molecules
    df = pd.DataFrame({
        'smiles': smiles,
        'cluster': clusters,
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'energy': [p['energy'] for p in properties],
        'charge': [p['charge'] for p in properties],
        'num_atoms': [p['num_atoms'] for p in properties]
    })
    
    # Save cluster assignments
    df.to_excel(os.path.join(vis_dirs['latent'], 'cluster_assignments.xlsx'), index=False)
    
    # Analyze cluster properties
    cluster_analysis = df.groupby('cluster').agg({
        'energy': ['mean', 'std', 'count'],
        'charge': ['mean', 'std'],
        'num_atoms': ['mean', 'std']
    }).round(4)
    
    cluster_analysis.to_excel(os.path.join(vis_dirs['latent'], 'cluster_analysis.xlsx'))
    
    # For each cluster, select representative molecules
    for cluster_id in range(n_clusters):
        ax = cluster_axes[cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Select up to 5 representative molecules from the cluster
        sample_size = min(5, len(cluster_data))
        if sample_size == 0:
            continue
            
        # Get samples from different parts of the cluster
        samples = cluster_data.sample(n=sample_size, random_state=42)
        
        # Clear the axis
        ax.clear()
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_data)})', fontsize=10)
        ax.axis('off')
        
        # Create a grid for molecule images
        img_grid = GridSpec(1, sample_size)
        img_grid.update(wspace=0.05, hspace=0.05)
        
        for j, (_, sample) in enumerate(samples.iterrows()):
            try:
                mol = Chem.MolFromSmiles(sample['smiles'])
                if mol:
                    # Draw molecule
                    img = Draw.MolToImage(mol, size=(200, 200))
                    
                    # Create subplot for this molecule
                    img_ax = fig.add_subplot(img_grid[0, j])
                    img_ax.imshow(img)
                    img_ax.axis('off')
                    
                    # Add property information
                    props = f"E: {sample['energy']:.2f}\nC: {sample['charge']}\nAtoms: {sample['num_atoms']}"
                    img_ax.text(0.5, -0.1, props, transform=img_ax.transAxes, 
                              ha='center', va='top', fontsize=8)
            except Exception as e:
                print(f"Error drawing molecule for cluster {cluster_id}: {e}")
                continue
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dirs['latent'], 'molecular_clusters_with_thumbnails.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering completed. Found {n_clusters} clusters.")

def extract_database(tar_path):
    """Extract the ASE database from tar file"""
    print("Extracting database...")
    os.makedirs(custom_temp_dir, exist_ok=True)
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not os.path.realpath(member_path).startswith(os.path.realpath(path)):
                raise ValueError("Attempted Path Traversal in Tar File")
        tar.extractall(path, members, numeric_owner=numeric_owner) 
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        db_members = [m for m in tar.getmembers() if m.name.endswith('.aselmdb')]
        if not db_members:
            raise ValueError("No .aselmdb file found in archive")
        safe_extract(tar, path=custom_temp_dir, members=db_members)
        extracted_path = os.path.join(custom_temp_dir, db_members[0].name)
        print(f"Database extracted to: {extracted_path}")
        return extracted_path

def visualize_molecular_maps(self, property_name='energy', num_samples=5000):
    """Generate molecular maps with density contours for chemical properties"""
    try:
        # Get a sample of molecules (for performance)
        sample_size = min(num_samples, len(self.data))
        sample_indices = np.random.choice(len(self.data), size=sample_size, replace=False)
        sample_data = [self.data[i] for i in sample_indices]
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Filter out invalid property values
        if property_name in ['homo', 'lumo', 'gap']:
            df = df[df[property_name] != 0.0]
        
        # Reduce dimensionality for visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(np.vstack([item['features'] for item in sample_data if 'features' in item]))
        
        # Update data with coordinates
        for i, idx in enumerate(sample_indices):
            if i < len(coords):
                self.data[idx]['umap_x'] = coords[i, 0]
                self.data[idx]['umap_y'] = coords[i, 1]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Scatter plot with property coloring
        scatter = plt.scatter(
            coords[:, 0], coords[:, 1],
            c=df[property_name],
            cmap='viridis',
            alpha=0.6,
            s=15,
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(property_name.capitalize(), rotation=270, labelpad=15)
        
        # Add density contours
        sns.kdeplot(
            x=coords[:, 0],
            y=coords[:, 1],
            levels=5,
            color='white',
            alpha=0.5,
            linewidths=1
        )
        
        plt.title(f'Molecular Map: {property_name.capitalize()} Distribution', fontsize=14)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.2)
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Error generating molecular map: {e}")
        return None

def main():
    try:
        print("=== Chemical Dataset Training Pipeline ===")
        
        # Extract database
        db_path = extract_database(dataset_path)
        
        # Create dataset with ALL molecules (sample_size=None)
        print("\n=== Creating chemical dataset ===")
        dataset = ChemicalDataset(db_path, tokenizer, sample_size=None)
        
        print(f"\nDataset created with {len(dataset)} molecules")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
               
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(results_dir, "model_checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=15,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            save_steps=400,  # Changed to be a multiple of eval_steps (200)
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=200,
            logging_dir=os.path.join(results_dir, "logs"),
            logging_steps=100,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none"  # Disable wandb/tensorboard reporting
        )
        
        # Initialize visualization callback
        metric_saver = MetricSaverCallback()
        viz_callback = VisualizationCallback(dataset, sample_size=100)
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[metric_saver, viz_callback]
        )
        
        # Train the model
        print("\n=== Starting training ===")
        trainer.train()

        # Manually save metrics after training
        history = trainer.state.log_history
        if history:
            df = pd.DataFrame(history)
            df.to_excel(os.path.join(vis_dirs['training'], "training_metrics.xlsx"), index=False)
            
            # Plot training curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            if 'loss' in df.columns:
                plt.plot(df['step'], df['loss'], label='Training Loss', alpha=0.7)
            if 'eval_loss' in df.columns:
                plt.plot(df['step'], df['eval_loss'], label='Validation Loss', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if 'learning_rate' in df.columns:
                plt.plot(df['step'], df['learning_rate'], color='orange', label='Learning Rate')
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dirs['training'], 'training_curves.png'), dpi=300)
            plt.close()
        
        # Save the final model
        print("\n=== Saving final model ===")
        final_model_dir = os.path.join(results_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Save test data
        print("Saving test data...")
        test_data = [dataset.data[i] for i in test_dataset.indices]
        with open(os.path.join(results_dir, "test_data.json"), "w") as f:
            json.dump(test_data[:1000], f, indent=2)  # Save sample of test data
        
        # Generate and save chemical descriptions with targets
        print("Generating chemical descriptions and targets Excel file...")
        generate_descriptions_targets_excel(dataset, results_dir)
        
        # Analyze latent space
        print("\n=== Analyzing latent space ===")
        analyze_latent_space(model, tokenizer, dataset)
        
        # Generate molecular maps with density contours
        print("\n=== Generating molecular maps with density contours ===")
        if hasattr(dataset, 'visualize_molecular_maps'):
            try:
                # Create directory for molecular maps
                maps_dir = os.path.join(results_dir, "molecular_maps")
                os.makedirs(maps_dir, exist_ok=True)
                
                # Visualize for different properties
                properties_to_visualize = ['energy', 'homo', 'lumo', 'gap', 'num_atoms']
                
                for prop in properties_to_visualize:
                    try:
                        print(f"Generating molecular map for {prop}...")
                        fig = dataset.visualize_molecular_maps(property_name=prop)
                        if fig:
                            fig.savefig(os.path.join(maps_dir, f'molecular_map_{prop}.png'), 
                                       dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            
                            # Save data to Excel
                            df = pd.DataFrame({
                                'smiles': [item['smiles'] for item in dataset.data],
                                'x': [item.get('umap_x', 0) for item in dataset.data],
                                'y': [item.get('umap_y', 0) for item in dataset.data],
                                prop: [item.get(prop, 0) for item in dataset.data]
                            })
                            df.to_excel(os.path.join(maps_dir, f'molecular_map_data_{prop}.xlsx'), 
                                       index=False)
                    except Exception as e:
                        print(f"Error generating map for {prop}: {e}")
            except Exception as e:
                print(f"Error in molecular map generation: {e}")
        
        # Generate final summary
        print("\n=== Generating final summary ===")
        final_summary = {
            "total_molecules_processed": len(dataset),
            "training_molecules": train_size,
            "validation_molecules": val_size,
            "test_molecules": test_size,
            "model_saved_to": final_model_dir,
            "results_directory": results_dir,
            "training_completed": True,
            "molecular_maps_generated": 'energy' in properties_to_visualize  # True if at least one map was generated
        }
        
        with open(os.path.join(results_dir, "training_summary.json"), "w") as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n=== Training completed successfully! ===")
        print(f"Results saved to: {results_dir}")
        print(f"Model saved to: {final_model_dir}")
        print(f"Total molecules processed: {len(dataset):,}")
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        print(f"Dataset size: {len(dataset):,} molecules")
        print(f"Training set: {train_size:,} molecules")
        print(f"Validation set: {val_size:,} molecules") 
        print(f"Test set: {test_size:,} molecules")
        
        if viz_callback.metrics_history:
            final_metrics = viz_callback.metrics_history[-1]
            print(f"Final training loss: {final_metrics.get('train_loss', 'N/A')}")
            print(f"Final validation loss: {final_metrics.get('eval_loss', 'N/A')}")
        
        print("\n=== Output Files Generated ===")
        print("Pre-training visualizations:")
        print(f"  - Dataset summary: {vis_dirs['pre_train']}/dataset_summary.json")
        print(f"  - Sample molecules: {vis_dirs['molecules']}/")
        print(f"  - Property distributions: {vis_dirs['pre_train']}/property_distributions.png")
        print(f"  - Element distribution: {vis_dirs['pre_train']}/element_distribution.png")
        
        print("\nTraining metrics:")
        print(f"  - Training curves: {vis_dirs['training']}/training_curves.png")
        print(f"  - Metrics data: {vis_dirs['training']}/training_metrics.xlsx")
        
        print("\nAttention maps:")
        print(f"  - Saved per epoch in: {vis_dirs['attention']}/")
        
        print("\nLatent space analysis:")
        print(f"  - PCA projection: {vis_dirs['latent']}/pca_projection.png")
        print(f"  - UMAP projection: {vis_dirs['latent']}/umap_projection.png")
        print(f"  - Molecular clusters: {vis_dirs['latent']}/molecular_clusters.png")
        print(f"  - Cluster analysis: {vis_dirs['latent']}/cluster_analysis.xlsx")
        
        print("\nMolecular maps:")
        print(f"  - Energy density map: {os.path.join(results_dir, 'molecular_maps/molecular_map_energy.png')}")
        print(f"  - HOMO density map: {os.path.join(results_dir, 'molecular_maps/molecular_map_homo.png')}")
        print(f"  - LUMO density map: {os.path.join(results_dir, 'molecular_maps/molecular_map_lumo.png')}")
        print(f"  - Gap density map: {os.path.join(results_dir, 'molecular_maps/molecular_map_gap.png')}")
        print(f"  - Size density map: {os.path.join(results_dir, 'molecular_maps/molecular_map_num_atoms.png')}")
        
        print(f"\nFinal model: {final_model_dir}/")
        print(f"Training summary: {results_dir}/training_summary.json")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Partial results may be available in:", results_dir)
        
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        
        with open(os.path.join(results_dir, "error_log.json"), "w") as f:
            json.dump(error_info, f, indent=2)
        
        print(f"Error details saved to: {results_dir}/error_log.json")
        raise

if __name__ == "__main__":
    main()