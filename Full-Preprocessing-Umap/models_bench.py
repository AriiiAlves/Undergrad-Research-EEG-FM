import os
import json
import torch
import sys
from fractions import Fraction
from braindecode.datautil import load_concat_dataset
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Paths
from tqdm import tqdm # Progress bar
import numpy as np
from scipy.signal import resample_poly

base_dir = Path(__file__).resolve().parent  # Parent folder
sys.path.append(str(base_dir / "BrainOmni")) # Where the brainomni files live

from BrainOmni.brainomni.model import BrainOmni
from BrainOmni.braintokenizer.model import BrainTokenizer

class BenchModel():
    """
    Generic benchmark: encodes every window with a pre-trained model, aggregates by subject and plots UMAPs.

    To add a new model, create a subclass and implement only:
        - load_model(): returns the pre-trained torch model
        - embed(x, window_ds): [batch, n_chans, n_times] tensor -> [batch, latent_dim] tensor
    And (optionally) set the input requirements: SFREQ, PATCH_SIZE, UNIT_SCALE, select_channels().
    """
    BASE_DIR = "bench_model"
    MODEL_NAME = "base"
    # Input requirements of the model (None = use the windows as they are)
    SFREQ = None        # Sampling rate expected by the model (windows are resampled if different)
    WINDOW_SIZE = None  # n_times expected by the model, after resampling (longer windows are cropped)
    PATCH_SIZE = None   # n_times must be a multiple of it (windows are cropped)
    UNIT_SCALE = 1.0    # Extra multiplier applied after converting to µV (e.g. 0.01 -> units of 100 µV)
    MIN_CHANNELS = 1    # Minimum number of usable channels (after select_channels)
    REQUIRES_POSITIONS = False # Model needs valid 3D electrode positions (montage)

    def __init__(self, windows_path, ds_name, batch_size=16):
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
        # Number of windows encoded at once (lower it if GPU runs out of memory)
        self.batch_size = batch_size
        # Load windows
        self.loaded_windows = []
        self.load_windows(windows_path, ds_name)
        # Verify if the windows fit the model specifications (before loading/downloading the model)
        self.verify_windows()
        # Variables
        self.final_features_path: str = ""
        self.final_features_name: str = ""
        self.z_score = False
        self.convert_to_uv = False
        # Set pre-trained model
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    def load_windows(self, windows_path, ds_name):
        windows_path = Path(windows_path)
        assert windows_path.is_dir(), f"{windows_path} is not a valid directory"

        self.loaded_windows = load_concat_dataset(
            path=windows_path,
            preload=False  # load by demand (lazy loading), saves RAM
        )

        self.DS_PLOTS_DIR = f"{self.PLOTS_DIR}/{ds_name}"
        os.makedirs(self.DS_PLOTS_DIR, exist_ok=True)

    # -------------------------------- MODEL SPECIFIC (override) --------------------------------
    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def embed(self, x: torch.Tensor, window_ds) -> torch.Tensor:
        """
        Receives a batch of windows [batch, n_chans, n_times] (already on device). Returns one vector per window [batch, latent_dim].
        self.current_picks / self.current_ch_names hold the channels of the batch (after select_channels).
        """
        raise NotImplementedError

    def select_channels(self, ch_names: list[str]) -> list[int]:
        """
        Indices of the channels the model can use (default: all).
        """
        return list(range(len(ch_names)))

    # -------------------------------- WINDOW INFO --------------------------------
    @staticmethod
    def window_info(window_ds):
        """
        mne.Info of the windows of a subject (works for raw-based and epochs-based window datasets).
        """
        for attr in ("_raw", "_windows"):
            obj = getattr(window_ds, attr, None)
            if obj is not None:
                return obj.info
        return window_ds._make_mne_info()

    def window_ch_names(self, window_ds) -> list[str]:
        return self.window_info(window_ds).ch_names

    # -------------------------------- VERIFICATION --------------------------------
    def __resample_ratio(self, sfreq: float):
        """
        Resampling ratio (up/down) from the windows sampling rate to the model one. None if no resampling is needed.
        """
        if self.SFREQ is None or sfreq == self.SFREQ:
            return None
        return Fraction(self.SFREQ / sfreq).limit_denominator(1000)

    def __model_n_times(self, n_times: int, sfreq: float):
        """
        Number of samples of a window after __prepare (resampling + cropping). Returns (resampled, final).
        """
        ratio = self.__resample_ratio(sfreq)
        resampled = n_times if ratio is None else int(np.ceil(n_times * ratio.numerator / ratio.denominator))
        final = resampled
        if self.WINDOW_SIZE is not None:
            final = min(final, self.WINDOW_SIZE)
        if self.PATCH_SIZE is not None:
            final = (final // self.PATCH_SIZE) * self.PATCH_SIZE
        return resampled, final

    def verify_windows(self):
        """
        Verify if the loaded windows fit the model specifications (SFREQ, WINDOW_SIZE, PATCH_SIZE, channels, positions, units).
        Raises an error for incompatibilities. Warns about what will be adapted automatically (resampling, cropping).
        """
        errors = []
        warnings = []
        sfreqs, sizes, n_chans_used = set(), set(), []
        uv_like_subjects = []

        for dataset_idx, window_ds in enumerate(self.loaded_windows.datasets):
            subject = f"Subject {dataset_idx}"
            if len(window_ds) == 0:
                errors.append(f"{subject}: has no windows")
                continue

            info = self.window_info(window_ds)
            sfreq = info["sfreq"]
            x = window_ds[0][0] # First window [n_chans, n_times]
            n_times = x.shape[-1]
            sfreqs.add(sfreq)
            sizes.add(n_times)

            # Channels
            ch_names = self.window_ch_names(window_ds)
            picks = self.select_channels(ch_names)
            n_chans_used.append(len(picks))
            if len(picks) < self.MIN_CHANNELS:
                errors.append(f"{subject}: only {len(picks)} usable channels (minimum: {self.MIN_CHANNELS})")

            # Window length (after resampling)
            resampled, final = self.__model_n_times(n_times, sfreq)
            if self.WINDOW_SIZE is not None and resampled < self.WINDOW_SIZE:
                errors.append(f"{subject}: window has {resampled} samples at {self.SFREQ or sfreq} Hz (model expects {self.WINDOW_SIZE})")
            elif final <= 0:
                errors.append(f"{subject}: window has {resampled} samples, shorter than the patch size ({self.PATCH_SIZE})")

            # Electrode positions
            if self.REQUIRES_POSITIONS and len(picks) > 0:
                pos = np.asarray(window_ds.ch_pos)[picks]
                invalid = [ch_names[picks[i]] for i, p in enumerate(pos) if np.isnan(p).any() or np.all(p == 0)]
                if invalid:
                    errors.append(f"{subject}: channels without valid positions (montage not set?): {invalid}")

            # Units (EEG in Volts is ~1e-5; values much larger probably are already in µV)
            if np.median(np.abs(x)) > 1e-3:
                uv_like_subjects.append(dataset_idx)

        # Adaptations made automatically by __prepare
        for sfreq in sorted(sfreqs):
            if self.__resample_ratio(sfreq) is not None:
                warnings.append(f"Windows at {sfreq} Hz will be resampled to {self.SFREQ} Hz")
        for n_times in sorted(sizes):
            for sfreq in sorted(sfreqs):
                resampled, final = self.__model_n_times(n_times, sfreq)
                if 0 < final < resampled:
                    warnings.append(f"Windows of {resampled} samples will be cropped to {final} ({resampled - final} samples discarded per window). Consider a window_size of {final} or a multiple of {self.PATCH_SIZE or final} in preprocessing")
        if len(sfreqs) > 1:
            warnings.append(f"Subjects have different sampling rates: {sorted(sfreqs)}")
        if len(sizes) > 1:
            warnings.append(f"Subjects have different window sizes: {sorted(sizes)}")
        if uv_like_subjects:
            warnings.append(f"Subjects {uv_like_subjects} look like they are already in µV (median |x| > 1e-3). Use convert_to_uv=False for them")

        # Report
        print("=" * 50)
        print(f"[{self.MODEL_NAME}] Windows verification")
        print(f"Subjects: {len(self.loaded_windows.datasets)} | Sampling rates: {sorted(sfreqs)} Hz | Window sizes: {sorted(sizes)} samples")
        if n_chans_used:
            print(f"Usable channels: {min(n_chans_used)}-{max(n_chans_used)}")
        print(f"Model specs: SFREQ={self.SFREQ}, WINDOW_SIZE={self.WINDOW_SIZE}, PATCH_SIZE={self.PATCH_SIZE}, MIN_CHANNELS={self.MIN_CHANNELS}, REQUIRES_POSITIONS={self.REQUIRES_POSITIONS}")
        for warning in warnings:
            print(f"[WARNING] {warning}")
        for error in errors:
            print(f"[ERROR] {error}")
        print("=" * 50)

        if errors:
            raise ValueError(f"[{self.MODEL_NAME}] Windows do not fit the model specifications ({len(errors)} errors). See the report above.")

    # -------------------------------- ENCODING --------------------------------
    def __prepare(self, x: np.ndarray, picks: list[int], sfreq: float):
        """
        Adapt one window [n_chans, n_times] to the model input requirements.
        """
        x = x[picks]

        # Resample to the model sampling rate
        ratio = self.__resample_ratio(sfreq)
        if ratio is not None:
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)

        # Crop to the model window size
        if self.WINDOW_SIZE is not None:
            assert x.shape[-1] >= self.WINDOW_SIZE, f"Window is shorter than the window size of {self.MODEL_NAME} ({self.WINDOW_SIZE} samples)"
            x = x[..., :self.WINDOW_SIZE]

        # Crop to a multiple of the patch size
        if self.PATCH_SIZE is not None:
            n_times = (x.shape[-1] // self.PATCH_SIZE) * self.PATCH_SIZE
            assert n_times > 0, f"Window is shorter than the patch size of {self.MODEL_NAME} ({self.PATCH_SIZE} samples)"
            x = x[..., :n_times]

        if self.convert_to_uv: x = x * 1e6 * self.UNIT_SCALE
        if self.z_score: x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
        return x

    def __latent_space_file(self, mode, limit):
        LATENT_SPACE_FILE = f"{self.MODEL_NAME}_{self.DS_NAME}_latent_space_{mode}"
        LATENT_SPACE_FILE += "_with_z_score" if self.z_score else ""
        LATENT_SPACE_FILE += "_converted_to_uv" if self.convert_to_uv else ""
        LATENT_SPACE_FILE += f"_limit_{limit}" if limit != 0 else ""
        LATENT_SPACE_FILE += ".pt"
        return Path(f"{self.LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")

    def encode_windows(self, limit=0):
        """
        Run the model ONCE over every window. Returns (window embeddings [N_windows, latent_dim], subject ids [N_windows]).
        Results are cached in disk, so every aggregation (avg/median/std/windows) reuses them.
        """
        file_path = self.__latent_space_file("windows", limit)
        subject_ids_path = file_path.with_name(file_path.stem + "_subject_ids.pt")

        if file_path.exists() and subject_ids_path.exists():
            return torch.load(file_path), torch.load(subject_ids_path)

        window_embeddings = []
        window_subject_ids = []

        with torch.no_grad():
            total_datasets = len(self.loaded_windows.datasets)

            for dataset_idx, window_ds in enumerate(self.loaded_windows.datasets):
                if limit != 0 and dataset_idx >= limit:
                    break

                ch_names = self.window_ch_names(window_ds)
                picks = self.select_channels(ch_names)
                assert len(picks) > 0, f"Subject {dataset_idx}: no channels usable by {self.MODEL_NAME}"
                sfreq = self.window_info(window_ds)["sfreq"]
                # Channels fed to the model (used by embed() to get names/positions)
                self.current_picks = picks
                self.current_ch_names = [ch_names[i] for i in picks]

                # Processing windows in batches to optimize RAM usage
                with tqdm(
                    total=len(window_ds),
                    desc=f"[{self.MODEL_NAME}] Subject {dataset_idx+1}/{limit if limit != 0 else total_datasets}",
                    unit="window"
                ) as pbar:
                    for start in range(0, len(window_ds), self.batch_size):
                        stop = min(start + self.batch_size, len(window_ds))
                        x = np.stack([self.__prepare(window_ds[idx][0], picks, sfreq) for idx in range(start, stop)])

                        inputs = torch.tensor(x, dtype=torch.float32).to(self.device)
                        embeddings = self.embed(inputs, window_ds)

                        window_embeddings.append(embeddings.reshape(embeddings.size(0), -1).cpu())
                        window_subject_ids.extend([dataset_idx] * (stop - start))
                        pbar.update(stop - start)

                # Free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not window_embeddings:
            raise ValueError("No windows were processed.")

        # Final matrix by window
        final_features = torch.cat(window_embeddings, dim=0)
        subject_ids = torch.tensor(window_subject_ids)

        print("Latent space format (by window):", final_features.shape)
        torch.save(final_features, file_path)
        # Saves subject IDs separately to identify each window in the UMAP
        torch.save(subject_ids, subject_ids_path)
        return final_features, subject_ids

    def use_latent_space(self, mode="avg", limit=0, z_score=False, convert_to_uv=True):
        """
        mode: "avg" | "median" | "std" (one vector by subject) or "windows" (one vector by window).
        """
        aggregations = {
            "avg": lambda t: t.mean(dim=0),
            "median": lambda t: t.median(dim=0).values,
            "std": lambda t: t.std(dim=0, unbiased=False),
        }
        assert mode in aggregations or mode == "windows", f"Invalid mode: {mode}"

        self.z_score = z_score
        self.convert_to_uv = convert_to_uv
        self.final_features_name = mode

        window_features, subject_ids = self.encode_windows(limit)

        if mode == "windows":
            self.final_features_path = self.__latent_space_file("windows", limit)
            return

        # Aggregates the window embeddings of each subject (cheap, recomputed every time)
        final_features = torch.stack([
            aggregations[mode](window_features[subject_ids == subject_idx])
            for subject_idx in subject_ids.unique()
        ])

        print(f"UMAP final format (by subject, {mode}):", final_features.shape)
        file_path = self.__latent_space_file(mode, limit)
        torch.save(final_features, file_path)
        self.final_features_path = file_path

    # Shortcuts (kept for compatibility)
    def use_avg_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        self.use_latent_space("avg", limit, z_score, convert_to_uv)

    def use_median_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        self.use_latent_space("median", limit, z_score, convert_to_uv)

    def use_std_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        self.use_latent_space("std", limit, z_score, convert_to_uv)

    def use_windows_latent_space(self, limit=0, z_score=False, convert_to_uv=True):
        self.use_latent_space("windows", limit, z_score, convert_to_uv)

    # -------------------------------- UMAP --------------------------------
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
    REQUIRES_POSITIONS = True # Uses the 3D electrode positions as input

    def load_model(self):
        return self.get_brainomni(f"{self.CHECKPOINTS_DIR}") # Inicialize ou carregue o checkpoint do BrainOmni

    def embed(self, x, window_ds):
        # Gets 3D position of subject electrodes
        pos_3d = torch.tensor(window_ds.ch_pos[self.current_picks], dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1).expand(x.size(0), -1, -1)
        sensor_type = torch.zeros((x.size(0), x.size(1)), dtype=torch.long).to(self.device)

        latent_space = self.model.encode(x, pos, sensor_type)
        # Calculates one embedding vector per window (mean over tokens)
        return latent_space.mean(dim=1)

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


# -------------------------------- BRAINDECODE FOUNDATION MODELS --------------------------------
class BenchLabram(BenchModel):
    BASE_DIR = "labram_bench"
    MODEL_NAME = "labram"
    SFREQ = 200
    PATCH_SIZE = 200
    UNIT_SCALE = 0.01 # LaBraM was pre-trained in units of 100 µV

    def load_model(self):
        from braindecode.models import Labram
        return Labram.from_pretrained("braindecode/labram-pretrained")

    def select_channels(self, ch_names):
        # LaBraM selects its positional embeddings by channel name, so unknown channels are dropped
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER
        known = {ch.upper() for ch in LABRAM_CHANNEL_ORDER}
        picks = [i for i, ch in enumerate(ch_names) if ch.upper() in known]
        dropped = [ch for ch in ch_names if ch.upper() not in known]
        if dropped: print(f"[{self.MODEL_NAME}] Channels not supported by the model (dropped): {dropped}")
        return picks

    def embed(self, x, window_ds):
        features = self.model(x, return_features=True, ch_names=self.current_ch_names)["features"]
        # Calculates one embedding vector per window (mean over channel/patch tokens)
        return features.reshape(x.size(0), -1, features.size(-1)).mean(dim=1)


class BenchCBraMod(BenchModel):
    BASE_DIR = "cbramod_bench"
    MODEL_NAME = "cbramod"
    SFREQ = 200
    PATCH_SIZE = 200
    UNIT_SCALE = 0.01 # CBraMod was pre-trained in units of 100 µV

    def load_model(self):
        from braindecode.models import CBraMod
        return CBraMod.from_pretrained("braindecode/cbramod-pretrained")

    def embed(self, x, window_ds):
        # features: [batch, n_chans, n_patches, emb_dim]
        features = self.model(x, return_features=True)["features"]
        # Calculates one embedding vector per window (mean over channel/patch tokens)
        return features.reshape(x.size(0), -1, features.size(-1)).mean(dim=1)
