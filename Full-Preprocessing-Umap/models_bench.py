import os
import json
import torch
import sys
import matplotlib.pyplot as plt
from braindecode.datautil import load_concat_dataset
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Paths
from tqdm import tqdm # Progress bar
import numpy as np

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
        self.final_features_name: str = ""
        self.z_score = False
        self.convert_to_uv = False
    
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

        print("="*50)
        print(final_features.shape)
        print("="*50)

        # Verify loaded windows (lazy loading)
        assert self.loaded_windows, "Windows are not loaded"
        
        labels = None

        if self.final_features_name == "windows":
            subject_ids_path = self.final_features_path.with_name(self.final_features_path.stem + "_subject_ids.pt")
            assert subject_ids_path.exists(), ("Subject IDs file for windows was not found.")

            subject_ids = torch.load(subject_ids_path).cpu().numpy()
            assert len(subject_ids) == len(final_features), ("Number of subject IDs does not match the number of windows.")

            # Metadata for each subject
            subject_metadata = self.loaded_windows.description.fillna("N/A")

            # Map each window to the metadata of its subject
            labels = {
                column: [
                    subject_metadata.iloc[subject_idx][column]
                    for subject_idx in subject_ids
                ]
                for column in subject_metadata.columns
            }

        else:
            labels = self.loaded_windows.description.fillna("N/A").to_dict("list")

            if limit != 0:
                labels = {key: values[:limit] for key, values in labels.items()}

        # Remove labels whose values are all "N/A"
        labels = {
            key: values
            for key, values in labels.items()
            if not all(value == "N/A" for value in values)
        }

        # For each label, run umap
        for label_type, subject_labels in labels.items():
            # For umap: final_features.shape = [N_subjects, latent_dim]
            reducer = umap.UMAP(
                n_neighbors=min(5, len(final_features) - 1), # Adjust n_neighbors if few subjects
                min_dist=0.1, 
                metric='cosine', 
                random_state=42,
                init="random"
            )
            embedding = reducer.fit_transform(final_features)

            # Plot 2D umap graphic
            if("age" not in str(label_type).lower()):
                fig = plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    x=embedding[:, 0], 
                    y=embedding[:, 1],
                    hue=subject_labels, # List with labels
                    palette='viridis',
                    s=100 if self.final_features_name != "windows" else 20, # Point size
                    alpha=0.8 if self.final_features_name != "windows" else 0.5, # Opacity
                    linewidth=0
                )

                plt.title(f"UMAP - {label_type.capitalize()} {"(Z-Score)" if self.z_score else "(Pure Raw)"} ({self.MODEL_NAME.replace("_", " ").capitalize()})", fontsize=14)
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()

                fig_path = f"{self.DS_PLOTS_DIR}/{self.MODEL_NAME}_{self.DS_NAME}_{self.final_features_name}_umap"
                fig_path += "_with_z_score" if self.z_score else ""
                fig_path += "_converted_to_uv" if self.convert_to_uv else ""
                fig_path += f"_label_{label_type}.png"

                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 7))
                points = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=np.asarray(subject_labels, dtype=float),
                    cmap="viridis",
                    s=100 if self.final_features_name != "windows" else 20, # Point size
                    alpha=0.8 if self.final_features_name != "windows" else 0.5, # Opacity
                    linewidths=0
                )

                fig.colorbar(points, ax=ax, label=label_type.capitalize())
                ax.set_title(f"UMAP - {label_type.capitalize()} {"(Z-Score)" if self.z_score else "(Pure Raw)"} ({self.MODEL_NAME.replace("_", " ").capitalize()})")
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.grid(True, linestyle="--", alpha=0.25)
                fig.tight_layout()

                fig_path = f"{self.DS_PLOTS_DIR}/{self.MODEL_NAME}_{self.DS_NAME}_{self.final_features_name}_umap"
                fig_path += "_with_z_score" if self.z_score else ""
                fig_path += "_converted_to_uv" if self.convert_to_uv else ""
                fig_path += f"_label_{label_type}.png"

                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
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

    def use_avg_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        LATENT_SPACE_FILE = f"brainomni_{self.DS_NAME}_latent_space_avg"
        LATENT_SPACE_FILE += "_with_z_score" if z_score else ""
        LATENT_SPACE_FILE += "_converted_to_uv" if convert_to_uv else ""
        LATENT_SPACE_FILE += ".pt"
        
        self.z_score = z_score
        self.convert_to_uv = convert_to_uv
        self.final_features_name = "avg"

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
                        
                        if convert_to_uv: x = x * 1e6
                        if z_score: x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
                        
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

    def use_median_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        LATENT_SPACE_FILE = f"brainomni_{self.DS_NAME}_latent_space_median"
        LATENT_SPACE_FILE += "_with_z_score" if z_score else ""
        LATENT_SPACE_FILE += "_converted_to_uv" if convert_to_uv else ""
        LATENT_SPACE_FILE += f"_limit_{limit}" if limit != 0 else ""
        LATENT_SPACE_FILE += ".pt"

        self.z_score = z_score
        self.convert_to_uv = convert_to_uv
        self.final_features_name = "median"

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
                pos_3d = torch.tensor(
                    window_ds.ch_pos, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1)

                subject_encoded_windows = []

                # Processing each window individually to optimize RAM usage
                with tqdm(
                    total=len(window_ds),
                    desc=f"Subject {dataset_idx+1}/{limit if limit != 0 else total_datasets}",
                    unit="window"
                ) as pbar:
                    for idx in range(len(window_ds)):
                        x, _, _ = window_ds[idx]

                        if convert_to_uv: x = x * 1e6
                        if z_score: x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

                        inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                        sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(self.device)

                        latent_space = self.model.encode(inputs, pos, sensor_type)
                        subject_encoded_windows.append(latent_space.cpu())
                        pbar.update(1)

                if not subject_encoded_windows: raise ValueError(f"Subject {dataset_idx} has no windows.")

                # Calculates the median embedding for each subject
                subject_tensor = torch.cat(
                    subject_encoded_windows, dim=0
                ).median(dim=0).values.mean(dim=0)

                subject_embeddings.append(subject_tensor)

                # Free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final matrix by subject
        final_features = torch.stack(subject_embeddings)
        final_features = final_features.reshape(final_features.size(0), -1)

        print("UMAP final format (by subject):", final_features.shape)
        torch.save(final_features, file_path)
        self.final_features_path = file_path
        
    def use_std_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        LATENT_SPACE_FILE = f"brainomni_{self.DS_NAME}_latent_space_std"
        LATENT_SPACE_FILE += "_with_z_score" if z_score else ""
        LATENT_SPACE_FILE += "_converted_to_uv" if convert_to_uv else ""
        LATENT_SPACE_FILE += f"_limit_{limit}" if limit != 0 else ""
        LATENT_SPACE_FILE += ".pt"

        self.z_score = z_score
        self.convert_to_uv = convert_to_uv
        self.final_features_name = "std"

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

                # Processing each window individually to optimize RAM usage
                with tqdm(
                    total=len(window_ds),
                    desc=f"Subject {dataset_idx+1}/{limit if limit != 0 else total_datasets}",
                    unit="window"
                ) as pbar:
                    for idx in range(len(window_ds)):
                        x, _, _ = window_ds[idx]

                        if convert_to_uv: x = x * 1e6
                        if z_score: x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

                        inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                        sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(self.device)

                        latent_space = self.model.encode(inputs, pos, sensor_type)
                        subject_encoded_windows.append(latent_space.cpu())
                        pbar.update(1)

                if not subject_encoded_windows: raise ValueError(f"Subject {dataset_idx} has no windows.")

                # Calculates the standard deviation of window embeddings
                window_embeddings = torch.cat(subject_encoded_windows, dim=0).mean(dim=1)

                subject_tensor = window_embeddings.std(dim=0, unbiased=False)
                subject_embeddings.append(subject_tensor)

                # Free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final matrix by subject
        final_features = torch.stack(subject_embeddings)
        final_features = final_features.reshape(final_features.size(0), -1)

        print("UMAP final format (by subject):", final_features.shape)
        torch.save(final_features, file_path)
        self.final_features_path = file_path
    
    def use_windows_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        LATENT_SPACE_FILE = f"brainomni_{self.DS_NAME}_latent_space_windows"
        LATENT_SPACE_FILE += "_with_z_score" if z_score else ""
        LATENT_SPACE_FILE += "_converted_to_uv" if convert_to_uv else ""
        LATENT_SPACE_FILE += f"_limit_{limit}" if limit != 0 else ""
        LATENT_SPACE_FILE += ".pt"

        self.z_score = z_score
        self.convert_to_uv = convert_to_uv
        self.final_features_name = "windows"

        file_path = Path(f"{self.LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")

        if file_path.exists():
            self.final_features_path = file_path
            return

        window_embeddings = []
        window_subject_ids = []

        with torch.no_grad():
            total_datasets = len(self.loaded_windows.datasets)

            for dataset_idx, window_ds in enumerate(self.loaded_windows.datasets):
                if limit != 0 and dataset_idx >= limit:
                    break

                # Gets 3D position of subject electrodes
                pos_3d = torch.tensor(window_ds.ch_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
                pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1)

                # Processing each window individually to optimize RAM usage
                with tqdm(
                    total=len(window_ds),
                    desc=f"Subject {dataset_idx+1}/{limit if limit != 0 else total_datasets}",
                    unit="window"
                ) as pbar:
                    for idx in range(len(window_ds)):
                        x, _, _ = window_ds[idx]

                        if convert_to_uv: x = x * 1e6
                        if z_score: x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

                        inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                        sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(self.device)

                        latent_space = self.model.encode(inputs, pos, sensor_type)

                        # Calculates one embedding vector per window
                        window_embedding = latent_space.mean(dim=1).squeeze(0)

                        window_embeddings.append(window_embedding.cpu())
                        window_subject_ids.append(dataset_idx)

                        pbar.update(1)

                # Free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not window_embeddings:
            raise ValueError("No windows were processed.")

        # Final matrix by window
        final_features = torch.stack(window_embeddings)
        final_features = final_features.reshape(final_features.size(0), -1)

        print("UMAP final format (by window):", final_features.shape)
        torch.save(final_features, file_path)

        # Saves subject IDs separately to identify each window in the UMAP
        subject_ids_path = file_path.with_name(file_path.stem + "_subject_ids.pt")
        torch.save(torch.tensor(window_subject_ids), subject_ids_path)

        self.final_features_path = file_path