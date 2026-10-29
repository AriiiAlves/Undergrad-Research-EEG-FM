import os
import json
import torch
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from fractions import Fraction
from datetime import datetime
from braindecode.datautil import load_concat_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Paths
from tqdm import tqdm # Progress bar
import numpy as np
from scipy.signal import resample_poly

base_dir = Path(__file__).resolve().parent  # Parent folder
sys.path.append(str(base_dir / "Models" / "BrainOmni")) # Tells Python which folder to look when importing BrainOmni's code

from Models.BrainOmni.brainomni.model import BrainOmni

def fit_umap(features, n_neighbors):
    """
    2D UMAP of features [N_points, latent_dim]. Runs in the child process of BenchModel.umap().
    """
    import umap # Only the child process loads numba
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', random_state=42, init="random").fit_transform(features)

class BenchModel():
    """
    Generic benchmark: encodes every window with a pre-trained model, aggregates by subject and plots UMAPs.

    To add a new model, create a subclass and implement only:
        - load_model(): returns the pre-trained torch model
        - embed(x, window_ds): [batch, n_chans, n_times] tensor -> [batch, latent_dim] tensor
    And (optionally) set the input requirements: SFREQ, WINDOW_SIZE, V_TO_MODEL, select_channels().
    """
    BASE_DIR = "Model-Benchmarks" # Every model saves in BASE_DIR/MODEL_NAME
    MODEL_NAME = "base"
    # Input requirements of the model (None = use the windows as they are)
    SFREQ = None        # Sampling rate expected by the model (windows are resampled if different)
    WINDOW_SIZE = None  # n_times expected by the model, after resampling (windows are regrouped: split or joined, the incomplete last one is dropped)
    MIN_CHANNELS = 1    # Minimum number of usable channels (after select_channels)
    REQUIRES_POSITIONS = False # Model needs valid 3D electrode positions (montage)
    V_TO_MODEL = 1.0    # Multiplier from Volts to the model units (e.g. 1e4 -> units of 100 µV)
    Z_SCORE = False     # Default per channel z-score of the windows (use_latent_space z_score overrides it)
    def __init__(self, windows_path, ds_name, batch_size=16):
        # Dataset name
        self.DS_NAME = ds_name.strip().replace(" ","")
        # Directories
        model_dir = f"{self.BASE_DIR}/{self.MODEL_NAME}"
        self.LATENT_SPACE_DIR = f"{model_dir}/latent_space_collection/{self.DS_NAME}"
        self.DS_PLOTS_DIR = f"{model_dir}/umap_plots/{self.DS_NAME}" # One subfolder per mode (avg, median, std, windows)
        self.CHECKPOINTS_DIR = f"{model_dir}/ckpt_collection"
        for directory in (self.LATENT_SPACE_DIR, self.DS_PLOTS_DIR, self.CHECKPOINTS_DIR):
            os.makedirs(directory, exist_ok=True)
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Number of windows encoded at once (lower it if GPU runs out of memory)
        self.batch_size = batch_size
        # Load windows (lazy loading, saves RAM)
        self.windows_path = windows_path
        assert Path(windows_path).is_dir(), f"{windows_path} is not a valid directory"
        self.loaded_windows = load_concat_dataset(path=Path(windows_path), preload=False)
        # Verify if the windows fit the model specifications (before loading/downloading the model)
        self.warned_channel_sets = set() # Channel sets already reported by match_model_channels
        self.channel_warnings = [] # Dropped/missing channels (saved in benchmark-info.json)
        self.verify_windows()
        # Variables (set by use_latent_space)
        self.mode = ""
        self.group_by = "recording"
        self.final_features = None
        self.subject_ids = None # Recording (dataset) index of each point
        self.z_score = self.Z_SCORE
        # Set pre-trained model
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

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

    @classmethod
    def model_channels(cls) -> list[str] | None:
        """
        Channel names the model accepts (override). None: any channel.
        """
        return None

    @classmethod
    def expected_channels(cls) -> list[str] | None:
        """
        Prints and returns the channel names the model accepts (None: any channel). Classmethod: works without loading windows/model.
        """
        channels = cls.model_channels()
        if channels is None:
            print(f"[{cls.MODEL_NAME}] Expected channels: any (no channel restriction)")
        else:
            print(f"[{cls.MODEL_NAME}] Expected channels ({len(channels)}): {channels}")
        return channels

    def match_model_channels(self, ch_names: list[str]) -> dict[int, int]:
        """
        {index in ch_names: index in model_channels()} of the channels the model knows (case-insensitive name match).
        Prints (once by channel set) the dropped channels and the model channels missing in the recording.
        """
        model_index = {ch.upper(): i for i, ch in enumerate(self.model_channels())}
        matches = {i: model_index[ch.upper()] for i, ch in enumerate(ch_names) if ch.upper() in model_index}
        if tuple(ch_names) not in self.warned_channel_sets:
            self.warned_channel_sets.add(tuple(ch_names))
            dropped = [ch for i, ch in enumerate(ch_names) if i not in matches]
            missing = [ch for i, ch in enumerate(self.model_channels()) if i not in matches.values()]
            if dropped:
                self.channel_warnings.append(f"Channels not supported by the model (dropped): {dropped}")
                print(f"[{self.MODEL_NAME}] {self.channel_warnings[-1]}")
            if missing:
                self.channel_warnings.append(f"Model channels missing in the recording ({len(missing)}/{len(model_index)}): {missing}")
                print(f"[{self.MODEL_NAME}] [WARNING] {self.channel_warnings[-1]}")
        return matches

    # -------------------------------- BENCHMARK INFO --------------------------------
    @classmethod
    def update_benchmark_info(cls, ds_name, config=None, warnings=(), errors=(), reset=False):
        """
        Saves the config, warnings and errors of the benchmark of a dataset in BASE_DIR/MODEL_NAME/benchmark-info.json (one entry by dataset).
        reset: starts the entry of the dataset again (new run). Classmethod: works even if the model could not be created.
        """
        path = Path(cls.BASE_DIR) / cls.MODEL_NAME / "benchmark-info.json"
        os.makedirs(path.parent, exist_ok=True)
        info = {}
        if path.exists():
            with open(path) as f:
                info = json.load(f)
        entry = {} if reset else info.get(ds_name, {})
        if config is not None:
            entry["config"] = config
        entry["warnings"] = entry.get("warnings", []) + list(warnings)
        entry["errors"] = entry.get("errors", []) + list(errors)
        entry["status"] = "error" if entry["errors"] else "ok"
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        info[ds_name] = entry
        with open(path, "w") as f:
            json.dump(info, f, indent=4, default=str)

    def benchmark_config(self):
        """
        Model input requirements and benchmark settings (as in params.json of the preprocessing).
        """
        return {
            "model_name": self.MODEL_NAME,
            "sfreq": self.SFREQ,
            "window_size": self.WINDOW_SIZE,
            "min_channels": self.MIN_CHANNELS,
            "requires_positions": self.REQUIRES_POSITIONS,
            "v_to_model": self.V_TO_MODEL,
            "z_score": self.Z_SCORE,
            "model_channels": self.model_channels(),
            "batch_size": self.batch_size,
            "device": str(self.device),
            "windows_path": str(self.windows_path),
        }

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

    # -------------------------------- VERIFICATION --------------------------------
    def __resample_ratio(self, sfreq: float):
        """
        Resampling ratio (up/down) from the windows sampling rate to the model one. None if no resampling is needed.
        """
        if self.SFREQ is None or sfreq == self.SFREQ:
            return None
        return Fraction(self.SFREQ / sfreq).limit_denominator(1000)

    def __resampled_n_times(self, n_times: int, sfreq: float):
        """
        Number of samples after resampling to the model sampling rate.
        """
        ratio = self.__resample_ratio(sfreq)
        return n_times if ratio is None else int(np.ceil(n_times * ratio.numerator / ratio.denominator))

    def verify_windows(self):
        """
        Verify if the loaded windows fit the model specifications (SFREQ, WINDOW_SIZE, channels, positions, units).
        Raises an error for incompatibilities. Warns about what will be adapted automatically (resampling, regrouping).
        """
        errors = []
        warnings = []
        sfreqs, sizes, n_chans_used = set(), set(), []
        not_volts_recordings = []

        for dataset_idx, window_ds in enumerate(self.loaded_windows.datasets):
            recording = f"Recording {dataset_idx} (subject {self.loaded_windows.description.iloc[dataset_idx].get('subject', 'N/A')})"
            if len(window_ds) == 0:
                errors.append(f"{recording}: has no windows")
                continue

            info = self.window_info(window_ds)
            sfreq = info["sfreq"]
            x = window_ds[0][0] # First window [n_chans, n_times]
            n_times = x.shape[-1]
            sfreqs.add(sfreq)
            sizes.add(n_times)

            # Channels
            ch_names = info.ch_names
            picks = self.select_channels(ch_names)
            n_chans_used.append(len(picks))
            if len(picks) < self.MIN_CHANNELS:
                errors.append(f"{recording}: only {len(picks)} usable channels (minimum: {self.MIN_CHANNELS})")

            # Window length (after resampling and regrouping)
            if self.WINDOW_SIZE is not None:
                total = self.__resampled_n_times(n_times * len(window_ds), sfreq)
                if total < self.WINDOW_SIZE:
                    errors.append(f"{recording}: has {total} samples at {self.SFREQ or sfreq} Hz, shorter than one model window ({self.WINDOW_SIZE})")

            # Electrode positions
            if self.REQUIRES_POSITIONS and len(picks) > 0:
                pos = np.asarray(window_ds.ch_pos)[picks]
                invalid = [ch_names[picks[i]] for i, p in enumerate(pos) if np.isnan(p).any() or np.all(p == 0)]
                if invalid:
                    errors.append(f"{recording}: channels without valid positions (montage not set?): {invalid}")

            # Units (EEG in Volts is ~1e-5; values much larger probably are not in V)
            if np.median(np.abs(x)) > 1e-3:
                not_volts_recordings.append(dataset_idx)

        # Adaptations made automatically by __prepare
        for sfreq in sorted(sfreqs):
            if self.__resample_ratio(sfreq) is not None:
                warnings.append(f"Windows at {sfreq} Hz will be resampled to {self.SFREQ} Hz")
        if self.WINDOW_SIZE is not None:
            for n_times in sorted(sizes):
                for sfreq in sorted(sfreqs):
                    resampled = self.__resampled_n_times(n_times, sfreq)
                    if resampled != self.WINDOW_SIZE:
                        warnings.append(f"Windows of {resampled} samples will be regrouped into windows of {self.WINDOW_SIZE} samples (the incomplete last window of each recording is dropped)")
        if len(sfreqs) > 1:
            warnings.append(f"Recordings have different sampling rates: {sorted(sfreqs)}")
        if len(sizes) > 1:
            warnings.append(f"Recordings have different window sizes: {sorted(sizes)}")
        if not_volts_recordings:
            warnings.append(f"Recordings {not_volts_recordings} do not look like Volts (median |x| > 1e-3). If the recordings are in µV, preprocess them with value_multiplier=1e-6")

        # Report
        print("=" * 50)
        print(f"[{self.MODEL_NAME}] Windows verification")
        print(f"Recordings: {len(self.loaded_windows.datasets)} | Sampling rates: {sorted(sfreqs)} Hz | Window sizes: {sorted(sizes)} samples")
        if n_chans_used:
            print(f"Usable channels: {min(n_chans_used)}-{max(n_chans_used)}")
        print(f"Model specs: SFREQ={self.SFREQ}, WINDOW_SIZE={self.WINDOW_SIZE}, MIN_CHANNELS={self.MIN_CHANNELS}, REQUIRES_POSITIONS={self.REQUIRES_POSITIONS}")
        for warning in warnings:
            print(f"[WARNING] {warning}")
        for error in errors:
            print(f"[ERROR] {error}")
        print("=" * 50)

        self.update_benchmark_info(self.DS_NAME, config=self.benchmark_config(), warnings=self.channel_warnings + warnings, errors=errors, reset=True)
        if errors:
            raise ValueError(f"[{self.MODEL_NAME}] Windows do not fit the model specifications ({len(errors)} errors). See the report above.")

    # -------------------------------- ENCODING --------------------------------
    def __prepare(self, window_ds, picks: list[int], sfreq: float):
        """
        Adapt every window of a subject to the model input requirements (lengths already checked by verify_windows).
        Returns [n_windows, n_chans, n_times].
        """
        windows = [window_ds[idx][0][picks] for idx in range(len(window_ds))]
        ratio = self.__resample_ratio(sfreq)
        self.__log(f"Loaded {len(windows)} windows ({len(picks)} channels picked, {sfreq} Hz)", windows[0], "per window")

        if self.WINDOW_SIZE is None:
            # Keep the windows as they are (only resampled)
            if ratio is not None:
                windows = [resample_poly(w, ratio.numerator, ratio.denominator, axis=-1) for w in windows]
                self.__log(f"Resampled windows {sfreq} -> {self.SFREQ} Hz (up={ratio.numerator}, down={ratio.denominator})", windows[0], "per window")
            else:
                self.__log("Keeping windows as they are (no resampling, no regrouping)", windows[0], "per window")
            x = np.stack(windows)
            self.__log("Stacked windows", x)
        else:
            # Windows are contiguous: rebuild the continuous signal, resample it once and split it in WINDOW_SIZE windows
            # (e.g. 50 -> 25: each window is split in two; 50 -> 75: windows are joined, only the incomplete last one is dropped)
            signal = np.concatenate(windows, axis=-1)
            self.__log("Joined windows into the continuous signal", signal, "signal")
            if ratio is not None:
                signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
                self.__log(f"Resampled signal {sfreq} -> {self.SFREQ} Hz (up={ratio.numerator}, down={ratio.denominator})", signal, "signal")
            n_windows = signal.shape[-1] // self.WINDOW_SIZE
            dropped = signal.shape[-1] - n_windows * self.WINDOW_SIZE
            signal = signal[:, :n_windows * self.WINDOW_SIZE]
            x = signal.reshape(signal.shape[0], n_windows, self.WINDOW_SIZE).transpose(1, 0, 2)
            self.__log(f"Split signal into {n_windows} windows of {self.WINDOW_SIZE} samples ({dropped} trailing samples dropped)", x)

        # Volts -> model units
        x = x * self.V_TO_MODEL
        self.__log(f"Converted windows to model units (x {self.V_TO_MODEL:g})", x)
        # Small eps: EEG in Volts has std ~1e-5, so a 1e-6 eps would distort the z-score
        if self.z_score:
            x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-12)
            self.__log("Z-scored every channel of every window", x)
        else:
            self.__log("Z-score off: keeping the values in model units", x)
        return x.astype(np.float32)

    def __log(self, step, x, label="windows"):
        """
        Print a preprocessing step with the shape and value range of the data after it.
        """
        x = np.asarray(x)
        print(f"[{self.MODEL_NAME}] {step} | {label}: shape={tuple(x.shape)}, mean={x.mean():.3g}, std={x.std():.3g}")

    def __print_section(self, title):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    def __suffix(self):
        """
        File name suffix of the current settings.
        """
        return "_with_z_score" if self.z_score else ""

    def __latent_space_file(self, mode, limit):
        limit_suffix = f"_limit_{limit}" if limit != 0 else ""
        group_suffix = f"_by_{self.group_by}" if mode != "windows" else ""
        return Path(f"{self.LATENT_SPACE_DIR}/{self.MODEL_NAME}_{self.DS_NAME}_latent_space_{mode}{group_suffix}{self.__suffix()}{limit_suffix}.pt")

    def encode_windows(self, limit=0):
        """
        Run the model ONCE over every window. Returns (window embeddings [N_windows, latent_dim], recording ids [N_windows]).
        Results are cached in disk, so every aggregation (avg/median/std/windows) reuses them.
        """
        file_path = self.__latent_space_file("windows", limit)
        subject_ids_path = file_path.with_name(file_path.stem + "_subject_ids.pt")

        if file_path.exists() and subject_ids_path.exists():
            final_features, subject_ids = torch.load(file_path), torch.load(subject_ids_path)
            print(f"[{self.MODEL_NAME}] Loaded cached window embeddings from {file_path} | shape={tuple(final_features.shape)}, recordings={len(subject_ids.unique())}")
            return final_features, subject_ids

        window_embeddings = []
        window_subject_ids = []
        datasets = self.loaded_windows.datasets[:limit or None]
        print(f"[{self.MODEL_NAME}] Encoding {len(datasets)} recordings on {self.device} (batch size {self.batch_size}, z-score {'on' if self.z_score else 'off'})")

        with torch.no_grad():
            for dataset_idx, window_ds in enumerate(datasets):
                info = self.window_info(window_ds)
                picks = self.select_channels(info.ch_names)
                # Channels fed to the model (used by embed() to get names/positions)
                self.current_picks = picks
                self.current_ch_names = [info.ch_names[i] for i in picks]

                # Windows of the subject adapted to the model (may differ in number from the loaded ones, if regrouped)
                self.__print_section(f"[{self.MODEL_NAME}] Recording {dataset_idx+1}/{len(datasets)}: preparing windows")
                model_windows = self.__prepare(window_ds, picks, info["sfreq"])
                print(f"[{self.MODEL_NAME}] Windows ready for the model | shape={model_windows.shape}, dtype={model_windows.dtype}")

                # Processing windows in batches to optimize GPU memory usage
                with tqdm(total=len(model_windows), desc=f"[{self.MODEL_NAME}] Recording {dataset_idx+1}/{len(datasets)}", unit="window") as pbar:
                    for start in range(0, len(model_windows), self.batch_size):
                        stop = min(start + self.batch_size, len(model_windows))
                        x = model_windows[start:stop]

                        embeddings = self.embed(torch.tensor(x).to(self.device), window_ds)
                        # Fails instead of silently flattening extra token dims (e.g. [batch, n_tokens, emb_dim])
                        assert embeddings.dim() == 2, f"embed() must return [batch, latent_dim], got {tuple(embeddings.shape)}"

                        window_embeddings.append(embeddings.cpu())
                        window_subject_ids.extend([dataset_idx] * (stop - start))
                        pbar.update(stop - start)
                print(f"[{self.MODEL_NAME}] Encoded windows | embedding shape per window={tuple(window_embeddings[-1].shape[1:])}")

                # Free GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final matrix by window
        final_features = torch.cat(window_embeddings, dim=0)
        subject_ids = torch.tensor(window_subject_ids)

        print(f"[{self.MODEL_NAME}] Latent space (by window) | shape={tuple(final_features.shape)}")
        torch.save(final_features, file_path)
        # Saves recording IDs separately to identify each window in the UMAP
        torch.save(subject_ids, subject_ids_path)
        print(f"[{self.MODEL_NAME}] Saved window embeddings to {file_path} (subject ids to {subject_ids_path})")
        return final_features, subject_ids

    def use_latent_space(self, mode="avg", limit=0, z_score=None, group_by="recording"):
        """
        mode: "avg" | "median" | "std" (one vector by group), "windows" (one vector by window, plots by label and event)
              or "windows-detailed" ("windows" + the events UMAPs by recording and by subject, in recordings_events/ and subjects_events/).
        z_score: None uses the model default (Z_SCORE). True/False forces it.
        group_by: "recording" (one point by subject/session/run) or "subject" (every recording of a subject together). Ignored for "windows".
        """
        aggregations = {
            "avg": lambda t: t.mean(dim=0),
            "median": lambda t: t.median(dim=0).values,
            "std": lambda t: t.std(dim=0, unbiased=False),
        }
        assert mode in aggregations or mode in ("windows", "windows-detailed"), f"Invalid mode: {mode}"
        assert group_by in ("recording", "subject"), f"Invalid group_by: {group_by}"

        # "windows-detailed" shares the embeddings, files and plots of "windows": it only adds the events UMAPs by recording/subject
        self.detailed = mode == "windows-detailed"
        mode = "windows" if self.detailed else mode
        self.mode = mode
        self.group_by = group_by
        self.z_score = self.Z_SCORE if z_score is None else z_score

        print("=" * 50)
        print(f"[{self.MODEL_NAME}] Latent space: mode={mode}, group_by={group_by}, z-score={'on' if self.z_score else 'off'}, limit={limit or 'all recordings'}")
        self.final_features, self.subject_ids = self.encode_windows(limit)

        if mode == "windows":
            print(f"[{self.MODEL_NAME}] Keeping one point per window (no aggregation) | shape={tuple(self.final_features.shape)}")
            return

        # Group of each window: its recording, or the subject of its recording
        recording_ids = self.subject_ids.numpy()
        if group_by == "subject":
            subjects = self.loaded_windows.description["subject"].astype(str).to_numpy()
            _, groups = np.unique(subjects[recording_ids], return_inverse=True)
        else:
            groups = recording_ids

        # Aggregates the window embeddings of each group (cheap, recomputed every time)
        group_values, first = np.unique(groups, return_index=True)
        self.final_features = torch.stack([
            aggregations[mode](self.final_features[torch.from_numpy(groups == group)])
            for group in group_values
        ])
        # One representative recording by point (its metadata labels the point)
        self.subject_ids = torch.from_numpy(recording_ids[first])

        print(f"[{self.MODEL_NAME}] Aggregated the windows of each {group_by} ({mode}) | shape={tuple(self.final_features.shape)}")
        torch.save(self.final_features, self.__latent_space_file(mode, limit))
        print(f"[{self.MODEL_NAME}] Saved aggregated latent space to {self.__latent_space_file(mode, limit)}")

    # -------------------------------- UMAP --------------------------------
    def umap(self, overwrite=False):
        """
        overwrite: False skips the plots that already exist (and the UMAP fit, if all of its plots exist). True regenerates them.
        """
        assert self.final_features is not None, "Call use_latent_space() before umap()"
        final_features = self.final_features.cpu().numpy()
        print(f"[{self.MODEL_NAME}] UMAP input ({self.mode}) | shape={final_features.shape} ([N_points, latent_dim])")
        if len(final_features) < 3: # UMAP needs n_neighbors >= 2
            print(f"[{self.MODEL_NAME}] Only {len(final_features)} points: skipping the UMAP")
            return

        # Metadata of the recording of each point (one row by group or by window), ignoring all "N/A" columns
        description = self.loaded_windows.description.fillna("N/A")
        metadata = description.iloc[self.subject_ids.numpy()]
        labels = {column: values for column, values in metadata.to_dict("list").items() if not all(value == "N/A" for value in values)}
        if self.mode != "windows" and self.group_by == "subject":
            # Columns that vary inside a subject (session, run, ...) do not describe a subject point
            labels = {column: values for column, values in labels.items()
                      if column == "subject" or not (description.groupby("subject")[column].nunique() > 1).any()}
        print(f"[{self.MODEL_NAME}] Labels to plot: {list(labels)}")
        os.makedirs(self.__plots_dir(), exist_ok=True)

        plot_paths = {label_type: f"{self.__plots_dir()}/{self.__plot_name()}_label_{label_type}.png" for label_type in labels}
        plot_paths = {label_type: path for label_type, path in plot_paths.items() if not self.__skip(path, overwrite)}
        if plot_paths:
            # For umap: final_features.shape = [N_points, latent_dim] (same embedding for every label)
            embedding = self.__fit_umap(final_features)

            # Plot 2D umap graphic for each label
            for label_type, plot_path in plot_paths.items():
                self.__plot_umap(embedding, label_type, labels[label_type], plot_path)

        if self.mode == "windows":
            self.__events_umap(final_features, overwrite)

    def __skip(self, plot_path, overwrite):
        """
        True if the plot already exists and must not be overwritten.
        """
        if overwrite or not os.path.exists(plot_path):
            return False
        print(f"[{self.MODEL_NAME}] Plot already exists, skipping (overwrite=False): {plot_path}")
        return True

    def __plots_dir(self):
        return f"{self.DS_PLOTS_DIR}/{self.mode}"

    def __plot_name(self):
        group_suffix = f"_by_{self.group_by}" if self.mode != "windows" else ""
        return f"{self.MODEL_NAME}_{self.DS_NAME}_{self.mode}{group_suffix}_umap{self.__suffix()}"

    def __fit_umap(self, features):
        """
        2D UMAP embedding of features [N_points, latent_dim].
        """
        # Many windows: UMAP default (15), 5 would split them in tiny islands. Few aggregated points: 5 (adjusted if fewer points)
        n_neighbors = min(15 if self.mode == "windows" else 5, len(features) - 1)
        print(f"[{self.MODEL_NAME}] Fitting UMAP on {len(features)} points (n_neighbors={n_neighbors}, min_dist=0.1, metric=cosine)...")
        # numba (UMAP) randomly segfaults while JIT-compiling on this machine: fit in a child process, so a crash is retried
        # instead of killing the run (random_state=42 makes every attempt give the same embedding)
        for attempt in range(1, 6):
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                    embedding = pool.submit(fit_umap, features, n_neighbors).result()
                break
            except BrokenProcessPool:
                if attempt == 5: raise
                print(f"[{self.MODEL_NAME}] UMAP process crashed, retrying ({attempt}/5)...")
        print(f"[{self.MODEL_NAME}] UMAP embedding | shape={embedding.shape}")
        return embedding

    def __plot_umap(self, embedding, label_type, point_labels, plot_path, title=None, hue_order=None):
        windows = self.mode == "windows"
        size = 20 if windows else 100 # Point size
        alpha = 0.5 if windows else 0.8 # Opacity

        fig, ax = plt.subplots(figsize=(10, 7))
        if "age" in str(label_type).lower():
            # Continuous label: colorbar
            points = ax.scatter(embedding[:, 0], embedding[:, 1], c=np.asarray(point_labels, dtype=float), cmap="viridis", s=size, alpha=alpha, linewidths=0)
            fig.colorbar(points, ax=ax, label=label_type.capitalize())
        elif str(label_type).lower() == "subject":
            # Too many subjects for a legend: write the subject id inside each dot instead
            subjects = sorted(set(map(str, point_labels)))
            colors = sns.color_palette("viridis", len(subjects))
            color_of = dict(zip(subjects, colors))
            ax.scatter(embedding[:, 0], embedding[:, 1], c=[color_of[str(label)] for label in point_labels], s=size * 3, alpha=alpha, linewidths=0)
            for (x, y), label in zip(embedding, point_labels):
                ax.text(x, y, str(label), ha="center", va="center", fontsize=4 if windows else 6, color="white", fontweight="bold")
        else:
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=point_labels, hue_order=hue_order, palette='viridis', s=size, alpha=alpha, linewidth=0, ax=ax)

        ax.set_title(f"UMAP - {title or label_type.capitalize()} {"(Z-Score)" if self.z_score else "(Pure Raw)"} ({self.MODEL_NAME.replace("_", " ").capitalize()})", fontsize=14)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[{self.MODEL_NAME}] Saved plot ({label_type}): {plot_path}")

    # -------------------------------- EVENTS --------------------------------
    def __window_events(self):
        """
        Event (annotation description) under the center of each encoded window, "misc" if there is none.
        The encoded windows are contiguous from the first loaded window, so the k-th one starts k window durations after it.
        """
        events = np.full(len(self.subject_ids), "misc", dtype=object)
        subject_ids = self.subject_ids.numpy()
        for dataset_idx in np.unique(subject_ids):
            positions = np.flatnonzero(subject_ids == dataset_idx)
            window_ds = self.loaded_windows.datasets[dataset_idx]
            raw = getattr(window_ds, "raw", None)
            if raw is None or len(raw.annotations) == 0:
                continue

            sfreq = raw.info["sfreq"]
            first = window_ds.metadata.iloc[0]
            duration = self.WINDOW_SIZE / (self.SFREQ or sfreq) if self.WINDOW_SIZE else (first["i_stop_in_trial"] - first["i_start_in_trial"]) / sfreq
            centers = first["i_start_in_trial"] / sfreq + (np.arange(len(positions)) + 0.5) * duration

            # Onsets in seconds from the first sample of the raw
            annotations = raw.annotations
            onsets = annotations.onset - (raw.first_time if annotations.orig_time is not None else 0)
            for position, center in zip(positions, centers):
                hit = np.flatnonzero((onsets <= center) & (center < onsets + annotations.duration))
                if hit.size:
                    events[position] = annotations.description[hit[0]]
        return events

    def __events_umap(self, final_features, overwrite=False):
        """
        UMAP of the windows inside an event (misc windows dropped), labeled by the event name: one with every recording and,
        only in "windows-detailed", one by recording (in recordings_events/) and one by subject, with every recording of the subject (in subjects_events/).
        """
        events = self.__window_events()
        keep = events != "misc"
        if not keep.any():
            print(f"[{self.MODEL_NAME}] No window inside an event: skipping the events UMAPs")
            return
        hue_order = sorted(set(events[keep])) # Same color for each event in every plot

        plot_path = f"{self.__plots_dir()}/{self.__plot_name()}_label_event.png"
        if not self.__skip(plot_path, overwrite):
            embedding = self.__fit_umap(final_features[keep])
            self.__plot_umap(embedding, "event", events[keep], plot_path, hue_order=hue_order)

        if not self.detailed:
            return

        recordings_dir = f"{self.__plots_dir()}/recordings_events"
        os.makedirs(recordings_dir, exist_ok=True)
        subject_ids = self.subject_ids.numpy()
        for dataset_idx in np.unique(subject_ids[keep]):
            recording = keep & (subject_ids == dataset_idx)
            if recording.sum() < 3: # UMAP needs n_neighbors >= 2
                continue
            plot_path = f"{recordings_dir}/{self.__plot_name()}_recording_{dataset_idx}_label_event.png"
            if self.__skip(plot_path, overwrite):
                continue
            subject = self.loaded_windows.description.iloc[dataset_idx].get("subject", "N/A")
            embedding = self.__fit_umap(final_features[recording])
            self.__plot_umap(embedding, "event", events[recording], plot_path,
                             title=f"Event (Recording {dataset_idx}, Subject {subject})", hue_order=hue_order)

        if "subject" not in self.loaded_windows.description:
            return
        subjects_dir = f"{self.__plots_dir()}/subjects_events"
        os.makedirs(subjects_dir, exist_ok=True)
        # Subject of each window (from the subject of its recording)
        window_subjects = self.loaded_windows.description["subject"].astype(str).to_numpy()[subject_ids]
        for subject in np.unique(window_subjects[keep]):
            windows = keep & (window_subjects == subject)
            if windows.sum() < 3: # UMAP needs n_neighbors >= 2
                continue
            plot_path = f"{subjects_dir}/{self.__plot_name()}_subject_{subject}_label_event.png"
            if self.__skip(plot_path, overwrite):
                continue
            n_recordings = len(np.unique(subject_ids[windows]))
            embedding = self.__fit_umap(final_features[windows])
            self.__plot_umap(embedding, "event", events[windows], plot_path,
                             title=f"Event (Subject {subject}, {n_recordings} recordings)", hue_order=hue_order)

class BenchBrainomni(BenchModel):
    '''
    Bench Brainomni model. Paper Preprocessing Setup:

    - Bandpass filter: 0.1–96 Hz
    - Notch filter: 50/60 Hz
    - Resample to 256 Hz
    - Bad channels detected (PSD-based) and interpolated
    - Average reference per sample: the mean across all sensors of each type is subtracted from each channel
    - Each channel normalised to zero mean and unit variance per sample (z-score)

    No physical unit is stated in the paper: the per-sample z-score makes the input independent of it (V, µV, ...).
    Filtering, resampling, bad channels and reference are handled in preprocessing.py.
    The z-score is on by default (Z_SCORE = True), but can be turned off with use_latent_space(z_score=False).

    Font: https://arxiv.org/abs/2505.18185 | https://github.com/OpenTSLab/BrainOmni
    '''
    MODEL_NAME = "brainomni_base"
    REQUIRES_POSITIONS = True # Uses the 3D electrode positions as input
    V_TO_MODEL = 1e6 # Unit is irrelevant after the z-score, but µV keeps the values far from the eps
    SFREQ = 256
    WINDOW_SIZE = 512 # Tokenizer window_length (2s): shorter windows are zero padded. Longer ones: 512 + k*384 (0.25 overlap) avoids padding
    Z_SCORE = True # Per channel z-score, as in the paper

    def load_model(self):
        ckpt_path = Path(self.CHECKPOINTS_DIR)
        files = ["base/model_cfg.json", "base/BrainOmni.pt"]

        # If files don't exist, download
        if not all((ckpt_path / file).exists() for file in files):
            print(f"Checkpoint not found in {ckpt_path}. Downloading files...")
            from huggingface_hub import hf_hub_download
            for file in files:
                hf_hub_download(repo_id="OpenTSLab/BrainOmni", filename=file, local_dir=ckpt_path)
            print("Download concluded!")

        with open(ckpt_path / files[0]) as f:
            model = BrainOmni(**json.load(f))
        model.load_state_dict(torch.load(ckpt_path / files[1], map_location="cpu", weights_only=True), strict=False)
        return model

    def embed(self, x, window_ds):
        # Gets 3D position of subject electrodes
        pos_3d = torch.tensor(window_ds.ch_pos[self.current_picks], dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1).expand(x.size(0), -1, -1)
        sensor_type = torch.zeros((x.size(0), x.size(1)), dtype=torch.long).to(self.device)

        # latent_space: [batch, n_neuro, n_time_tokens, emb_dim]
        latent_space = self.model.encode(x, pos, sensor_type)
        # Calculates one embedding vector per window (mean over neuro/time tokens)
        return latent_space.mean(dim=(1, 2))


# -------------------------------- BRAINDECODE FOUNDATION MODELS --------------------------------
class BenchLabram(BenchModel):
    """
    Bench LaBraM model. Paper Preprocessing Setup:

    - Remove irrelevant channels
    - Bandpass filter: 0.1–75 Hz
    - Notch filter: 50 Hz
    - Resample to 200 Hz
    - Set unit to 1µV

    Font: https://github.com/935963004/labram
    """
    MODEL_NAME = "labram"
    SFREQ = 200
    WINDOW_SIZE = 3000 # The pretrained model only accepts exactly 3000 samples (15 patches of 200): its position/temporal embeddings are fixed to 15 patches
    V_TO_MODEL = 1e6 # LaBraM was pre-trained in µV
    Z_SCORE = False

    def load_model(self):
        from braindecode.models import Labram
        return Labram.from_pretrained("braindecode/labram-pretrained")

    @classmethod
    def model_channels(cls):
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER
        return list(LABRAM_CHANNEL_ORDER)

    def select_channels(self, ch_names):
        # LaBraM selects its positional embeddings by channel name, so unknown channels are dropped
        return list(self.match_model_channels(ch_names))

    def embed(self, x, window_ds):
        features = self.model(x, return_features=True, ch_names=self.current_ch_names)["features"]
        # Calculates one embedding vector per window (mean over channel/patch tokens)
        return features.reshape(x.size(0), -1, features.size(-1)).mean(dim=1)

class BenchCBraMod(BenchModel):
    """
    Bench CBraMod model. Paper Preprocessing Setup:

    - Bandpass filter: 0.3–75 Hz
    - Notch filter: 60 Hz
    - Resample to 200 Hz
    - Window size: 30s non-overlapping
    - Set unit to 100µV

    Font: https://openreview.net/pdf?id=NPNUHgHF2w - Chapter 3.1 Pre-Training
    """
    MODEL_NAME = "cbramod"
    SFREQ = 200
    #WINDOW_SIZE = 200 # Its positional encoding works with any number of patches.
    V_TO_MODEL = 1e4 # CBraMod was pre-trained in 100 µV
    Z_SCORE = False

    def load_model(self):
        from braindecode.models import CBraMod
        # The pretrained config has no n_outputs: skip the classification head (only the encoder features are used)
        return CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)

    def embed(self, x, window_ds):
        # features: [batch, n_chans, n_patches, emb_dim]
        features = self.model(x, return_features=True)["features"]
        # Calculates one embedding vector per window (mean over channel/patch tokens)
        return features.reshape(x.size(0), -1, features.size(-1)).mean(dim=1)

class BenchBENDR(BenchModel):
    """
    Bench BENDR model. Paper Preprocessing Setup:

    - Resample to 256 Hz
    - 19 channels of the 10-20 system (fixed order) + 1 "SCALE" channel (relative amplitude of the window)
    - Channels missing in the recording are filled with zeros (as in the DN3 Deep1010 mapping used to pre-train it)
    - Each window min-max scaled to [-1, 1] (all channels together)

    Features: output of the start token of the contextualizer (the aggregate token of the paper).
    The SCALE channel uses the TUEG amplitude range (int16 in 0.1 µV: ±3276.8 µV) as the maximum (approximation of DN3).

    Font: https://doi.org/10.3389/fnhum.2021.653659 | https://github.com/SPOClab-ca/BENDR
    """
    MODEL_NAME = "bendr"
    SFREQ = 256
    WINDOW_SIZE = 1024 # 4s (any length >= 96 samples works: the encoder downsamples by 96)
    V_TO_MODEL = 1e6 # µV (only used by the SCALE channel: the min-max scaling makes the rest unit independent)
    Z_SCORE = False
    MAX_SCALE = 6553.6 # Amplitude range (µV) that maps to SCALE = 1

    @classmethod
    def model_channels(cls):
        from braindecode.models.bendr import BENDR_CHANNEL_ORDER
        return [ch for ch in BENDR_CHANNEL_ORDER if ch != "SCALE"]

    def load_model(self):
        from braindecode.models import BENDR
        return BENDR.from_pretrained("braindecode/braindecode-bendr", n_times=self.WINDOW_SIZE, sfreq=self.SFREQ)

    def select_channels(self, ch_names):
        return list(self.match_model_channels(ch_names))

    def embed(self, x, window_ds):
        # Places each channel in its BENDR position (missing ones stay 0), then appends the SCALE channel
        model_index = list(self.match_model_channels(self.current_ch_names).values())
        full = torch.zeros(x.size(0), len(self.model_channels()), x.size(-1), device=x.device, dtype=x.dtype)
        full[:, model_index] = x
        low = full.amin(dim=(1, 2), keepdim=True)
        high = full.amax(dim=(1, 2), keepdim=True)
        scale = 2 * (torch.clamp((high - low) / self.MAX_SCALE, max=1.0) - 0.5)
        full = 2 * (full - low) / (high - low + 1e-12) - 1
        full = torch.cat([full, scale.expand(-1, 1, x.size(-1))], dim=1)
        return self.model(full, return_features=True)["features"]

class BenchBIOT(BenchModel):
    """
    Bench BIOT model ("six datasets" checkpoint). Paper Preprocessing Setup:

    - Resample to 200 Hz
    - 18 bipolar channels (TUH double banana + C3-A2, C4-A1). The recording must already have them (named e.g. "FP1-F7"):
      unipolar recordings have no usable channel
    - Window size: 10s
    - Each channel divided by the 95th percentile of its absolute amplitude

    Missing bipolar channels are dropped: BIOT tokenizes each channel separately (with its own channel embedding), so any subset works.

    Font: https://arxiv.org/abs/2305.10351 | https://github.com/ycq091044/BIOT
    """
    MODEL_NAME = "biot"
    SFREQ = 200
    WINDOW_SIZE = 2000 # 10s
    V_TO_MODEL = 1e6 # µV (irrelevant after the percentile scaling)
    Z_SCORE = False

    @classmethod
    def model_channels(cls):
        from braindecode.models.biot import BIOT_CHANNEL_ORDER
        return BIOT_CHANNEL_ORDER

    def load_model(self):
        from braindecode.models import BIOT
        return BIOT.from_pretrained("braindecode/biot-pretrained-six-datasets-18chs", n_times=self.WINDOW_SIZE)

    def select_channels(self, ch_names):
        return list(self.match_model_channels(ch_names))

    def embed(self, x, window_ds):
        x = x / (torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-8)
        # Same as BIOT encoder forward, but each channel gets the embedding of its BIOT position (not of its index in x)
        encoder = self.model.encoder
        model_index = list(self.match_model_channels(self.current_ch_names).values())
        tokens = []
        for i, channel in enumerate(model_index):
            spec = encoder.patch_embedding(encoder.stft(x[:, i:i + 1, :]))
            channel_token = encoder.channel_tokens(encoder.index[channel]).expand(spec.size(0), spec.size(1), -1)
            tokens.append(encoder.positional_encoding(spec + channel_token))
        return encoder.transformer(torch.cat(tokens, dim=1)).mean(dim=1)

class BenchEEGPT(BenchModel):
    """
    Bench EEGPT model. Paper Preprocessing Setup:

    - Resample to 256 Hz
    - Window size: 4s (1024 samples)
    - Set unit to mV
    - Global average reference
    - Channels: 62 of the 10-20 system (each one has a learned channel embedding, so any subset works)

    Features: summary tokens of the encoder (4 x 512), averaged over the time patches.

    Font: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
    """
    MODEL_NAME = "eegpt"
    SFREQ = 256
    WINDOW_SIZE = 1024 # 4s
    V_TO_MODEL = 1e3 # EEGPT was pre-trained in mV
    Z_SCORE = False

    @classmethod
    def model_channels(cls):
        from braindecode.models.eegpt import EEGPT_CHANNELS
        return EEGPT_CHANNELS

    def load_model(self):
        from braindecode.models import EEGPT
        return EEGPT.from_pretrained("braindecode/eegpt-pretrained", n_times=self.WINDOW_SIZE, sfreq=self.SFREQ, chan_proj_type="none")

    def select_channels(self, ch_names):
        return list(self.match_model_channels(ch_names))

    def embed(self, x, window_ds):
        encoder = self.model.target_encoder
        # The encoder accepts any subset of its channels (by channel id): only its shape check is fixed to the 62 channels
        encoder.num_patches = (x.size(1), encoder.num_patches[1])
        chan_ids = torch.tensor(list(self.match_model_channels(self.current_ch_names).values())).unsqueeze(0)
        # z: [batch, n_patches, embed_num, embed_dim]
        z = encoder(x, chan_ids)
        return z.mean(dim=1).flatten(1)

class BenchREVE(BenchModel):
    """
    Bench REVE model (base). Paper Preprocessing Setup:

    - Bandpass filter: 0.5–99.5 Hz
    - Resample to 200 Hz
    - Z-score normalization, values clipped at 15 standard deviations
    - Any channels: 4D positional encoding from the 3D electrode positions (montage) and time
    - Input length: multiple of 1s

    Weights under the REVE Responsible Use License (https://huggingface.co/brain-bzh/reve-base).

    Font: https://arxiv.org/abs/2510.21585 | https://brain-bzh.github.io/reve/
    """
    MODEL_NAME = "reve"
    REQUIRES_POSITIONS = True # Uses the 3D electrode positions as input
    SFREQ = 200
    WINDOW_SIZE = 2000 # 10s: patches of 200 samples with stride 180 cover it exactly (11 patches)
    V_TO_MODEL = 1 # Unit is irrelevant after the z-score
    Z_SCORE = True

    def load_model(self):
        from braindecode.models import REVE
        # The position bank (downloaded at init) goes to the checkpoints folder instead of MNE_DATA
        os.environ.setdefault("REVE_POSITIONS_PATH", str(Path(self.CHECKPOINTS_DIR).resolve()))
        return REVE.from_pretrained("brain-bzh/reve-base", n_times=self.WINDOW_SIZE, sfreq=self.SFREQ, n_chans=1, n_outputs=1)

    def embed(self, x, window_ds):
        x = x.clamp(-15, 15)
        # Electrode positions in meters (same frame as the REVE position bank)
        pos = torch.tensor(window_ds.ch_pos[self.current_picks], dtype=torch.float32, device=self.device).unsqueeze(0).expand(x.size(0), -1, -1)
        # features: [batch, n_chans, n_patches, embed_dim]
        features = self.model(x, pos=pos, return_features=True)["features"]
        return features.mean(dim=(1, 2))

class BenchLUNA(BenchModel):
    """
    Bench LUNA model (base). Paper Preprocessing Setup:

    - Bandpass filter: 0.1–75 Hz
    - Notch filter: 50/60 Hz
    - Resample to 256 Hz
    - Window size: 5s non-overlapping
    - Each channel of each window z-scored
    - Any channels: learned queries unify the channels by cross-attention, using their 3D positions (montage)

    Features: latent of the temporal encoder ([n_patches, n_queries x embed_dim]), averaged over the patches.
    Weights under CC BY-ND 4.0 (https://huggingface.co/PulpBio/LUNA).

    Font: https://arxiv.org/abs/2510.22257 | https://github.com/pulp-bio/BioFoundation
    """
    MODEL_NAME = "luna"
    REQUIRES_POSITIONS = True # Uses the 3D electrode positions as input
    SFREQ = 256
    WINDOW_SIZE = 1280 # 5s (32 patches of 40 samples)
    V_TO_MODEL = 1e6 # Unit is irrelevant after the z-score
    Z_SCORE = True

    def load_model(self):
        from braindecode.models import LUNA
        # n_outputs=0: pre-training (reconstruction) head, which is not used
        return LUNA.from_pretrained("PulpBio/LUNA", filename="LUNA_base.safetensors", n_outputs=0, n_chans=1, n_times=self.WINDOW_SIZE,
                                    embed_dim=64, num_queries=4, depth=8)

    def embed(self, x, window_ds):
        model = self.model
        pos = torch.tensor(window_ds.ch_pos[self.current_picks], dtype=torch.float32, device=self.device).unsqueeze(0).expand(x.size(0), -1, -1)
        # Same as LUNA forward up to the latent (before the decoder head)
        tokens, _ = model.prepare_tokens(x, pos)
        tokens, _ = model.cross_attn(tokens) # [batch * n_patches, n_queries, embed_dim]
        latent = tokens.reshape(x.size(0), -1, tokens.size(1) * tokens.size(2))
        for block in model.blocks:
            latent = block(latent)
        return model.norm(latent).mean(dim=1)
