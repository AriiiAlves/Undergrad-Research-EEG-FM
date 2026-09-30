import os
import json
import torch
import sys
import matplotlib.pyplot as plt
from braindecode.datautil import load_concat_dataset
from einops import rearrange
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Paths
from tqdm import tqdm # Progress bar

base_dir = Path(__file__).resolve().parent  # Parent folder
sys.path.append(str(base_dir / "BrainOmni")) # Where the brainomni files live

from BrainOmni.brainomni.model import BrainOmni
from BrainOmni.braintokenizer.model import BrainTokenizer

class BenchModel():
    BASE_DIR = "bench_model"

    def __init__(self, windows_path, ds_name):
        # Dataset name
        self.DS_NAME = ds_name.strip().replace(" ","")
        # Directories
        self.LATENT_SPACE_DIR = f"{self.BASE_DIR}/latent_space_collection"
        self.PLOTS_DIR = f"{self.BASE_DIR}/umap_plots"
        self.DS_PLOTS_DIR = f"{self.PLOTS_DIR}/{ds_name}"
        self.CHECKPOINTS_DIR = f"{self.BASE_DIR}/ckpt_collection"
        # Ensure that the folder already exists
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.LATENT_SPACE_DIR, exist_ok=True)
        os.makedirs(self.PLOTS_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINTS_DIR, exist_ok=True)
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load windows
        self.loaded_windows = []
        self.load_windows(windows_path, ds_name)
        # Variables
        self.final_features_path: str = ""
        self.z_score = False
    
    def load_windows(self, windows_path, ds_name):
        windows_path = Path(windows_path)
        assert windows_path.is_dir(), f"{windows_path} is not a valid directory"

        self.loaded_windows = load_concat_dataset(
            path=windows_path,
            preload=False  # load by demand (lazy loading), saves RAM
        )

        self.DS_PLOTS_DIR = f"{self.PLOTS_DIR}/{ds_name}"
        os.makedirs(self.DS_PLOTS_DIR, exist_ok=True)
    
    def umap(self):
        # Verify features path
        self.final_features_path = Path(self.final_features_path)
        assert self.final_features_path.exists(), "Final features .pt path isn't valid"

        # Load features and convert to numpy
        final_features = torch.load(self.final_features_path)
        final_features = final_features.cpu().numpy()
        limit = len(final_features)

        # Verify loaded windows (lazy loading)
        assert self.loaded_windows, "Windows are not loaded"
        labels = self.loaded_windows.description.fillna("N/A").to_dict('list')
        if limit != 0:
            labels = {k: v[:limit] for k, v in labels.items()}

        # For each label, run umap
        for label_type, subject_labels in labels.items():
            # For umap: final_features.shape = [N_subjects, latent_dim]
            reducer = umap.UMAP(
                n_neighbors=min(5, len(final_features) - 1), # Adjust n_neighbors if few subjects
                min_dist=0.1, 
                metric='cosine', 
                random_state=42
            )
            embedding = reducer.fit_transform(final_features)

            # Plot 2D umap graphic
            fig = plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=embedding[:, 0], 
                y=embedding[:, 1],
                hue=subject_labels, # List with labels
                palette='viridis',
                s=100,
                alpha=0.8
            )

            plt.title(f"UMAP - {label_type.capitalize()} {"(Z-Score)" if self.z_score else "(Pure Raw)"} ({self.MODEL_NAME.replace("_", " ").capitalize()})", fontsize=14)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            if(self.z_score):
                fig.savefig(f"{self.DS_PLOTS_DIR}/{self.MODEL_NAME}_{self.DS_NAME}_umap_with_z_score_label_{label_type}.png", dpi=300, bbox_inches='tight')
            else:
                fig.savefig(f"{self.DS_PLOTS_DIR}/{self.MODEL_NAME}_{self.DS_NAME}_umap_without_z_score_label_{label_type}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            

class BenchBrainomni(BenchModel):
    BASE_DIR = "brainomni_bench"
    MODEL_NAME = "brainomni_base"

    def __init__(self, windows_path, ds_name):
        super().__init__(windows_path, ds_name)
        # Set pre-trained BrainOmni model
        self.model = self.get_brainomni(f"{self.CHECKPOINTS_DIR}") # Inicialize ou carregue o checkpoint do BrainOmni
        self.model.to(self.device)
        self.model.eval()

    # load brainomni model
    @staticmethod
    def get_brainomni(ckpt_path) -> BrainOmni:
        ckpt_path = Path(ckpt_path)
        model_config_path = ckpt_path / "base/model_cfg.json"
        ckpt_file = ckpt_path / "base/BrainOmni.pt"

        # If files doesn't exist, download
        if not (model_config_path.exists() and ckpt_file.exists()):
            ckpt_path.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoint not found in {ckpt_path}. Downloading files...")
            
            from huggingface_hub import hf_hub_download
            
            # Plz replace by correct repo_id and files
            repo_id = "OpenTSLab/BrainOmni" 
            
            hf_hub_download(repo_id=repo_id, filename="base/model_cfg.json", local_dir=ckpt_path)
            hf_hub_download(repo_id=repo_id, filename="base/BrainOmni.pt", local_dir=ckpt_path)
            print("Download concluded!")

        model_config_path = os.path.join(ckpt_path, "base/model_cfg.json")
        with open(model_config_path) as f:
            model_config = json.load(f)
        model = BrainOmni(**model_config)
        checkpoint = torch.load(os.path.join(ckpt_path, "base/BrainOmni.pt"), map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        # when using omni, freeze tokenizer
        for p in model.tokenizer.parameters():
            p.requires_grad = False
        return model

    def use_avg_latent_space(self, limit=0, z_score=False):
        LATENT_SPACE_FILE = f"brainomni_{self.DS_NAME}_latent_space_avg_with_z_score.pt" if z_score else f"brainomni_{self.DS_NAME}_latent_space_avg_without_z_score.pt"
        self.z_score = z_score

        file_path = Path(f"{self.LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")

        if file_path.exists(): 
            self.final_features_path = file_path
            return

        subject_embeddings = []

        with torch.no_grad():
            total_datasets = len(self.loaded_windows.datasets)
            for dataset_idx, window_ds in enumerate(self.loaded_windows.datasets):
                if limit != 0 and dataset_idx >= limit: 
                    break 

                # Gets 3D position of subject electrodes
                pos_3d = torch.tensor(window_ds.ch_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
                pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1)
                
                subject_encoded_windows = []
                
                # Processing each one window by time in RAM (optimized)
                with tqdm(total=len(window_ds), desc=f"Subject {dataset_idx+1}/{limit if limit != 0 else total_datasets}", unit="window") as pbar:
                    for idx in range(len(window_ds)):
                        x, _, _ = window_ds[idx]
                        # Applies z-score
                        if z_score: 
                            x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
                        
                        inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                        sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(self.device)

                        latent_space = self.model.encode(inputs, pos, sensor_type)
                        subject_encoded_windows.append(latent_space.cpu())
                        pbar.update(1)

                # Calculates subject avg
                subject_tensor = torch.cat(subject_encoded_windows, dim=0).mean(dim=[0, 1])
                subject_embeddings.append(subject_tensor)
                
                # free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        # Final matrix by subject
        final_features = torch.stack(subject_embeddings)
        final_features = final_features.reshape(final_features.size(0), -1)

        print("UMAP final format (by subject):", final_features.shape)
        torch.save(final_features, file_path)
        self.final_features_path = file_path