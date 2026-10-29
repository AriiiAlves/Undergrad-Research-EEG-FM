# Import system utils
import os
from pathlib import Path
import gc
import json
# Import libraries for graphic plot, MNE for EEG data and numpy for arrays
import eegdash.dataset
import eegdash.schemas
import matplotlib.pyplot as plt
import mne
import numpy as np
# Dataset
from braindecode.datasets import BaseConcatDataset
from braindecode import preprocessing as bd_preproc
# Preprocessing
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA, compute_proj_eog, compute_proj_ecg
# Paralelism
import joblib
from filelock import FileLock
# Forces matplotlib as MNE browser backend
mne.viz.set_browser_backend("matplotlib")
# Function types
from collections.abc import Callable
# moabb
from moabb.datasets.utils import dataset_search
from braindecode.datasets import MOABBDataset, TUH, TUHAbnormal, NMT, SleepPhysionet
from moabb import set_download_dir
# eegdash
from eegdash import EEGDash
from eegdash import EEGDashDataset
import eegdash
#pandas
import pandas as pd
#others
import contextlib

class DatasetManager():
    def __init__(self):
        self.dataset = None
        self.dataset_name = None

        self.DATA_DIR = Path("./data").resolve()
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.DATASETS_DIR = Path("./PreProcessed-Datasets").resolve()
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        self.montage = None
    
    # -------------------------------- DATASET DATABASE INFO --------------------------------
    def get_moabb_datasets(self):
        return dataset_search()

    def get_eegdash_datasets(self, dataset:str=None, task:str=None, limit:int=50):
        """
        Util: https://codebeautify.org/python-formatter-beautifier
        """
        client = EEGDash()

        filters = {}
        if dataset:
            filters["dataset"] = dataset
        if task:
            filters["task"] = task

        return client.find(filters, limit=limit)
    
    # -------------------------------- DATASET INFO --------------------------------
    def __print_section(self, title):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)


    def __print_value(self, name, value):
        print(f"{name:<30}: {value}")


    def __print_dataframe(self, name, df):
        print(f"\n{name}:")
        if df is None:
            print("  Not available")
        elif isinstance(df, pd.DataFrame):
            print(df.to_string(index=False))
        else:
            print(df)


    def __safe_getattr(self, obj, name, default="Not available"):
        try:
            return getattr(obj, name, default)
        except Exception as exc:
            return f"Could not retrieve: {exc}"


    def __inspect_epochs(self, epochs, recording_index):
        self.__print_section(f"RECORDING {recording_index}: EPOCHS INFORMATION")

        self.__print_value("Python type", type(epochs).__name__)
        self.__print_value("Number of epochs", len(epochs))
        self.__print_value("Number of channels", len(epochs.ch_names))
        self.__print_value("Sampling frequency (Hz)", epochs.info["sfreq"])
        self.__print_value("Channel names", epochs.ch_names)
        self.__print_value(
            "Epoch time range (seconds)",
            (epochs.tmin, epochs.tmax),
        )
        self.__print_value("Event IDs", epochs.event_id)
        self.__print_value("Drop log length", len(epochs.drop_log))

        print("\nFull MNE Info:")
        print(epochs.info)

        print("\nEvents array:")
        print(epochs.events[:20])

        print("\nMetadata:")
        self.__print_dataframe("Epoch metadata", epochs.metadata)

        print("\nSignal statistics:")

        for idx, channel in enumerate(epochs.ch_names):
            signal = epochs.get_data(picks=[idx])

            print(
                f"  {channel:<12} "
                f"min={np.min(signal):.6e} "
                f"max={np.max(signal):.6e} "
                f"mean={np.mean(signal):.6e} "
                f"std={np.std(signal):.6e}"
            )


    def __inspect_raw(self, raw, recording_index):
        self.__print_section(f"RECORDING {recording_index}: MNE INFORMATION")

        self.__print_value("Python type", type(raw).__name__)
        self.__print_value("Number of channels", len(raw.ch_names))
        self.__print_value("Sampling frequency (Hz)", raw.info["sfreq"])
        self.__print_value("Number of samples", raw.n_times)
        self.__print_value(
            "Duration (seconds)",
            raw.times[-1] if raw.n_times else 0,
        )
        self.__print_value("First sample", raw.first_samp)
        self.__print_value("Last sample", raw.last_samp)
        self.__print_value("Measurement date", raw.info.get("meas_date"))
        self.__print_value("Channel names", raw.ch_names)

        print("\nFull MNE Info:")
        print(raw.info)

        print("\nChannel types:")
        print(raw.get_channel_types())

        print("\nChannel counts by type:")
        channel_types = raw.get_channel_types()

        for channel_type in sorted(set(channel_types)):
            print(f"  {channel_type}: {channel_types.count(channel_type)}")

        print("\nMontage:")
        montage = raw.get_montage()
        print(montage if montage is not None else "No montage assigned")

        print("\nAnnotations:")
        annotations = raw.annotations

        if len(annotations):
            annotation_df = pd.DataFrame({
                "onset": annotations.onset,
                "duration": annotations.duration,
                "description": annotations.description,
            })
            print(annotation_df.to_string(index=False))
        else:
            print("No annotations")

        print("\nEvents derived from annotations:")

        try:
            events, event_id = mne.events_from_annotations(
                raw,
                verbose=False,
            )

            self.__print_value("Event count", len(events))
            self.__print_value("Event IDs", event_id)

            if len(events):
                print("First 20 events [sample, previous value, event ID]:")
                print(events[:20])

        except Exception as exc:
            print(f"Could not extract events: {exc}")

        print("\nSignal statistics:")

        for idx, (channel, channel_type) in enumerate(
            zip(raw.ch_names, channel_types)
        ):
            signal = raw.get_data(picks=[idx])[0]

            print(
                f"  {channel:<12} "
                f"type={channel_type:<8} "
                f"min={np.min(signal):.6e} "
                f"max={np.max(signal):.6e} "
                f"mean={np.mean(signal):.6e} "
                f"std={np.std(signal):.6e}"
            )


    def inspect_dataset(self):
        """
        Inspect the Braindecode dataset and write the report to a text file.

        The report is saved in PreProcessed-Datasets/<DATASET_NAME>/<dataset_name>_inspection.txt. Returns its path.
        """
        if self.dataset is None:
            raise ValueError(
                "No dataset loaded. Load a dataset before inspecting it."
            )
        
        self.__print_section("BRAinDECODE DATASET OVERVIEW")
        print("Inspecting dataset. It may take a while.")

        output_file = self.DATASETS_DIR / self.dataset_name.upper() / f"{self.dataset_name}_inspection.txt"
        output_path = output_file.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Redirect all print() calls inside the inspection methods to the file.
        with output_path.open("w", encoding="utf-8") as file:
            with contextlib.redirect_stdout(file):
                dataset = self.dataset

                self.__print_section("BRAinDECODE DATASET OVERVIEW")

                self.__print_value("Dataset class", type(dataset).__name__)
                self.__print_value("Number of recordings", len(dataset))

                print("\nDataset description:")
                self.__print_dataframe("Description", dataset.description)

                print("\nDescription columns:")
                print(list(dataset.description.columns))

                print("\nAvailable dataset-level attributes:")

                for name in ("datasets", "description", "transform"):
                    self.__print_value(
                        name,
                        self.__safe_getattr(dataset, name),
                    )

                self.__print_section("INDIVIDUAL RECORDINGS")

                for index, entry in enumerate(dataset.datasets):
                    self.__print_section(
                        f"RECORDING {index}: BRAinDECODE ENTRY"
                    )

                    self.__print_value("Entry class", type(entry).__name__)
                    self.__print_value(
                        "Entry length",
                        self.__safe_getattr(entry, "__len__"),
                    )
                    self.__print_value(
                        "Description",
                        self.__safe_getattr(entry, "description"),
                    )

                    print("\nEntry attributes:")
                    print([
                        name
                        for name in dir(entry)
                        if not name.startswith("_")
                    ])

                    raw = self.__safe_getattr(entry, "raw", None)
                    epochs = self.__safe_getattr(entry, "epochs", None)

                    if isinstance(raw, mne.io.BaseRaw):
                        self.__inspect_raw(raw, index)

                    elif isinstance(epochs, mne.BaseEpochs):
                        self.__inspect_epochs(epochs, index)

                    else:
                        print(
                            "No recognized MNE Raw or Epochs object found."
                        )
                        print("Inspect the entry attributes above.")

                self.__print_section("INSPECTION COMPLETE")

        print(f"Dataset inspection saved to: {output_path}")
        return output_path

    # -------------------------------- DATASET LOADER --------------------------------
    def load_moabb_dataset(self, dataset:str, subject_limit:int=0):
        """
        MOABB is a curated benchmark of about 70+ BCI datasets: motor imagery, P300/ERP, SSVEP, c-VEP and resting state. 
        Each one is wrapped by hand, with paradigm metadata and event mapping.
        """
        # dataset_search() returns dataset objects, so compare by class name (e.g. "BNCI2014_001")
        assert dataset in [type(ds).__name__ for ds in self.get_moabb_datasets()], "[ERROR] Dataset does not exist in MOABB database."

        set_download_dir(str(self.DATA_DIR))

        self.dataset = MOABBDataset(dataset_name=dataset)
        self.dataset_name = dataset

        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    def load_eegdash_dataset(self, dataset:str=None, subject_limit:int=0):
        """
        EEGDash indexes hundreds of BIDS-formatted datasets from OpenNeuro/NEMAR (ds00xxxx IDs). 
        These are mostly cognitive and clinical research recordings, not BCI benchmarks.
        """
        assert self.get_eegdash_datasets(dataset), "[ERROR] Dataset does not exist in EEGDash database."
        
        self.dataset = EEGDashDataset(cache_dir=self.DATA_DIR, dataset=dataset)
        self.dataset_name = dataset

        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    def load_tuh_dataset(self, path: str = None, abnormal: bool = True, subject_limit: int = 0):
        """
        Single dataset (TUH corpus)
        """
        assert path is not None, "[ERROR] TUH requires a local path (manual download)."
        if abnormal:
            self.dataset = TUHAbnormal(path=path, target_name="pathological")
            self.dataset_name = "TUHAbnormal"
        else:
            self.dataset = TUH(path=path)
            self.dataset_name = "TUH"

        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    def load_nmt_dataset(self, path: str = None, subject_limit: int = 0):
        """
        Single dataset
        """
        assert path is not None, "[ERROR] NMT requires a local path (manual download)."
        self.dataset = NMT(path=path)
        self.dataset_name = "NMT"

        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    def load_sleep_physionet_dataset(self, subject_limit: int = 0, crop_wake_mins: int = 30):
        """
        Single dataset (Sleep-EDF)
        """
        self.dataset = SleepPhysionet(crop_wake_mins=crop_wake_mins)
        self.dataset_name = "SleepPhysionet" 

        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    def load_generic_dataset(self, dataset:BaseConcatDataset, subject_limit:int=0):
        assert dataset, "[ERROR] Dataset does not exist in EEGDash database."

        self.dataset = dataset
        
        if(subject_limit != 0):
            self.dataset = BaseConcatDataset(self.dataset.datasets[:subject_limit])
    
    # -------------------------------- RECORDING INFO --------------------------------
    def inspect_recording(self, recording):
        print("=" * 60)
        print("Recording type:", type(recording))

        if isinstance(recording, mne.io.BaseRaw):
            print("Format: continuous")
            print("Channels:", len(recording.ch_names))
            print("Sampling rate:", recording.info["sfreq"])
            print("Duration:", recording.times[-1], "seconds")
            print("Shape:", recording.get_data().shape)
            print("Channel names:", recording.ch_names)

        elif isinstance(recording, mne.BaseEpochs):
            print("Format: epoched")
            print("Number of epochs:", len(recording))
            print("Channels:", len(recording.ch_names))
            print("Sampling rate:", recording.info["sfreq"])
            print("Epoch duration:", recording.times[-1] - recording.times[0])
            print("Shape:", recording.get_data().shape)
            print("Event IDs:", recording.event_id)
            print("Channel names:", recording.ch_names)

        else:
            print("Unsupported recording type")

    # -------------------------------- PREPROCESSING --------------------------------
    def __auto_detect_bads(self, raw:mne.io.Raw, ransac:bool=True):
        """
        Receives raw input (mne.io.raw object). Mark bad channels (raw.info['bads'])
        """
        nc = NoisyChannels(raw, random_state=42)
        nc.find_all_bads(ransac=ransac) # Set RANSAC = True if do you have a good computer.
        raw.info['bads'] = nc.get_bads()
        return raw

    def __eliminate_artifacts(self, raw:mne.io.Raw, run:bool=True, ica:bool=True):
        """
        Receives raw input (mne.io.raw object). Eliminate ECG/EOG artifacts (if there are any channels).
        """
        if(not run):
            print("[Eliminate artifacts] Run=False. Skipping")
            return raw

        # Verify existence of EOG/ECG channels
        has_eog = len(mne.pick_types(raw.info, meg=False, eeg=False, eog=True, ecg=False)) > 0
        has_ecg = len(mne.pick_types(raw.info, meg=False, eeg=False, eog=False, ecg=True)) > 0

        # Nothing to remove without EOG/ECG channels (skips the costly ICA fit)
        if(not has_eog and not has_ecg):
            print("[Eliminate artifacts] No EOG/ECG channels found. Skipping")

        # ICA has a big computational cost
        elif(ica):
            # Filtered copy for stable ICA decomposition (ICA performs poorly if trained on low-frequency drifts (< 1.0Hz)
            raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design="firwin", verbose="ERROR")
            ica = ICA(n_components=15, random_state=42, method="fastica")
            ica.fit(raw_ica) # Train ICA model with raw data

            bads = []
            # Find ocular artifacts using LOC and ROC
            if(has_eog): 
                eog_indices, _ = ica.find_bads_eog(raw)
                bads.extend(eog_indices)
            # Find cardiac artifacts using ECG
            if(has_ecg): 
                ecg_indices, _ = ica.find_bads_ecg(raw) # If ch_name = None, MNE checks raw for any channels marked as type
                bads.extend(ecg_indices)
            # Mark components to remove
            ica.exclude = list(set(bads))
            print(f"[Eliminate artifacts] ICA removed components: {ica.exclude}")
            # Remove components and reconstruct EEG signal. Applies to original raw.
            ica.apply(raw)

        # SSP is faster than ICA
        else:
            if (has_eog):
                projs_eog, _ = compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, verbose="ERROR")
                raw.add_proj(projs_eog)

            if has_ecg:
                projs_ecg, _ = compute_proj_ecg(raw, n_grad=0, n_mag=0, n_eeg=1, verbose="ERROR")
                raw.add_proj(projs_ecg)

            raw.apply_proj()

        return raw

    def __drop_aux_channels(self, raw:mne.io.Raw):
        """
        Receives raw input (mne.io.raw object). Drop aux channels (non-eeg) (optimize memory use).
        """
        aux_channels = []
        for ch in raw.ch_names:
            if('eeg' not in raw.get_channel_types(picks=ch)): aux_channels.append(ch)
        raw.drop_channels(aux_channels)
        return raw

    def __interpolate_bads_if_any(self, raw:mne.io.Raw):
        """
        Receives raw input (mne.io.raw object). Take channels marked as bad (raw.info["bads"]) and interpolate them using nearby sensors.
        """
        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=False) # Keeps bads marked so they stay visible in the PSD (after)
            # This info will be used to decide which records exclude/mantain
        return raw

    def __define_reference(self, raw:mne.io.Raw, ref_channels:list[str]=[]):
        """
        Receives raw input (mne.io.raw object). Define reference based in custom channels. If ref_channels=[] (empty), it will be calculated the average reference (CAR).
        """
        if(len(ref_channels) == 0): 
            raw.set_eeg_reference(ref_channels='average')
            return raw

        for ch in ref_channels:
            if(ch not in raw.ch_names or 'eeg' not in raw.get_channel_types(picks=ch)):
                assert False, "Bad references: It is not eeg-type or it doesn't exists"

        print(f"[Define Reference] Applying with reference channels: {ref_channels}")

        raw.set_eeg_reference(ref_channels=ref_channels)
        raw.drop_channels(ref_channels)
        return raw

    def __verify_montage(self, raw:mne.io.Raw):
        """
        Receives raw input (mne.io.raw object). Verify if the montage was successfuly. If not, raise an error.

        The error is raised if there is any channels with invalid position (Nan/Zero).
        """
        for ch_info in raw.info['chs']:
            ch_name = ch_info['ch_name']
            loc = ch_info['loc'][:3]  # Pick X, Y, Z coordinates
            
            if (np.isnan(loc).any() or np.all(loc == 0)) and ('eeg' in raw.get_channel_types(picks=ch_name)):
                assert False, f"CHANNEL WITH INVALID POSITION (NaN/Zero): {ch_name} -> Coord: {loc} -> Type: {raw.get_channel_types(picks=ch_name)}"

        return raw

    def __investigate_montage(self, raw: mne.io.Raw):
        if(self.montage):
            raw.set_montage(self.montage)
            return raw

        channel_names = raw.ch_names
        channel_types = raw.get_channel_types()
        montages = mne.channels.get_builtin_montages(descriptions=False)

        montages_not_matched = {}
        montages_matched = []

        for mnt in montages:
            mnt_channels = mne.channels.make_standard_montage(mnt).ch_names
            montages_not_matched[mnt] = []
            
            for name, ch_type in zip(channel_names, channel_types):
                if ch_type == "eeg" and name not in mnt_channels:
                    montages_not_matched[mnt].append(name)
            
            if len(montages_not_matched[mnt]) == 0:
                montages_matched.append(mnt)
        
        if len(montages_matched) > 0:
            if len(montages_matched) == 1:
                print(f"Only one matched montage: {montages_matched[0]}. Applying montage.")
                raw.set_montage(montages_matched[0])
                self.montage = montages_matched[0]
            else:
                print(f"There are {len(montages_matched)} possible montages. Please select one:")
                for i, mnt in enumerate(montages_matched):
                    print(f"[{i}] {mnt}")

                opt = -1
                while opt < 0 or opt >= len(montages_matched):
                    try:
                        opt = int(input("Enter index: "))
                    except ValueError:
                        opt = -1

                raw.set_montage(montages_matched[opt])
                self.montage = montages_matched[opt]
        else:
            print("Not matched channels for each montage:")
            for mnt, mnt_channels in montages_not_matched.items():
                print(f"{mnt} : {mnt_channels}")
            
            assert False, "[ERROR] No matched montages. Please treat channel names."
        
        return raw

    def __psd_gen(self, raw:mne.io.Raw, sub_id:str, record_idx:int, group:str, output_dir:str, sub_dir:str):
        """
        Receives raw input (mne.io.raw object). Generate PSD and PSD topomap. Bad channels are kept in the PSD plot.
        Named by recording (a subject may have many sessions/runs, each one would overwrite the previous plot).
        """
        psd_dir = self.DATASETS_DIR / output_dir / "Recordings-PSDs" / sub_dir
        psd_dir.mkdir(parents=True, exist_ok=True)
        
        psd = raw.compute_psd(verbose=False)
        fig1 = psd.plot(show=False)
        fig2 = psd.plot_topomap(show=False)
        fig1.suptitle(f"Recording {record_idx}, Subject {sub_id} ({group}) PSD")
        fig2.suptitle(f"Recording {record_idx}, Subject {sub_id} ({group}) PSD Topomap")

        fig1.savefig(psd_dir / f"psd_recording_{record_idx}_{sub_dir}.png", dpi=150, bbox_inches="tight") # DPI 150 economiza espaço/tempo
        fig2.savefig(psd_dir / f"psd_topomap_recording_{record_idx}_{sub_dir}.png", dpi=150, bbox_inches="tight")
        plt.close('all')

    def __apply_multiplier(self, raw:mne.io.Raw, value_multiplier:float=1.0):
        # Converts data units (e.g. 1e-6 for recordings stored in µV -> V, the MNE default)
        if(value_multiplier != 1.0):
            raw.load_data()
            raw.apply_function(lambda x: x * value_multiplier, channel_wise=False) # Default picks: all data channels (bads included)
        return raw

    def __notch_filter(self, raw:mne.io.Raw, notch_freqs:list[float]=[]):
        # Removes power line noise. Frequencies above Nyquist can't be filtered (and have no noise to remove)
        freqs = [f for f in notch_freqs if f < raw.info["sfreq"] / 2]
        if(freqs):
            raw.load_data()
            raw.notch_filter(freqs=freqs, method="fir", fir_design="firwin")
        return raw

    def __record_dropped(self, output_dir:str, rec_key:str, info:dict):
        """
        Record a dropped recording (and the reason) in params.json, so it is skipped on the next run (overwrite=False).
        """
        params_path = self.DATASETS_DIR / output_dir / "params.json"

        # Lock, so parallel workers don't overwrite each other's changes
        with FileLock(f"{params_path}.lock"):
            with open(params_path, "r") as f:
                params = json.load(f)

            # Initialize the key if it doesn't exist
            params.setdefault("dropped", {})
            params["dropped"][rec_key] = info

            with open(params_path, "w") as f:
                json.dump(params, f, indent=4)

    # Static: it is sent to the joblib workers, and private (__) non-static methods can't be pickled
    @staticmethod
    def __default_ch_treatment(raw):
        return raw

    # Single underscore: name-mangled (__) methods can't be pickled and sent to joblib workers
    def _process_single_subject(self, record, output_dir, channels_treatment, l_freq, h_freq, down_freq, bad_ch_tolerance, ransac, ica, window_size, ref_channels, overwrite, montage, treat_artifacts, value_multiplier, notch_freqs, record_idx):
        """
        Process a single subject, saving the result directly on disk to save RAM/VRAM
        """
        sub_id = record.description.get('subject', 'unknown')
        self.__print_section(f"PROCESSING SUBJECT: {sub_id}")

        subject_path = self.DATASETS_DIR / output_dir / "PreProcessed-EEG"
        subject_path.mkdir(parents=True, exist_ok=True)

        # Skip by recording, not by subject (a subject may have many sessions/runs): windows.save(offset=record_idx) writes each recording to folder <record_idx>/
        rec_key = f"recording_{record_idx}"
        if (subject_path / str(record_idx) / "description.json").exists():
            if(overwrite): print(f"Found recording {record_idx} (subject {sub_id}) in folder: {record_idx}. Overwriting with new data...")
            else:
                print(f"Found recording {record_idx} (subject {sub_id}) in folder: {record_idx}. Skipping..")
                return True

        if(not overwrite):
            # Lock, so it doesn't read params.json while another worker is writing it
            params_path = self.DATASETS_DIR / output_dir / "params.json"
            with FileLock(f"{params_path}.lock"), open(params_path, "r") as f:
                params = json.load(f)
                if("dropped" in params.keys() and rec_key in params["dropped"].keys()):
                    print(f"Found recording {record_idx} (subject {sub_id}) in dropped recordings ({params_path}). Skipping...")
                    return True

        # Var initializing (if subject got dropped, assure the variables to be deleted exist)
        single_ds = None
        windows = None
        
        try:
            # 1. Validates physical file
            if (not record.raw):
                print(f"[Skipping] Path missing for subject {sub_id}")
                return False

            # 2. Pipeline de pré-processamento aplicado isoladamente ao registro
            single_ds = BaseConcatDataset([record])
            group = record.description.get('group', '')

            # Stage 1: channel setup and bad channel detection
            preprocess(
                single_ds,
                [
                # Note: if apply_on_array=False, passes raw object. If not, passes raw underlying NumPy array.
                # 1. Channels treatment
                Preprocessor(channels_treatment, apply_on_array=False),
                # 2. Set montage (fixed if montage != "", otherwise auto-detected)
                Preprocessor(self.__investigate_montage, apply_on_array=False),
                # 3. Verify NaN in eeg channels
                Preprocessor(self.__verify_montage, apply_on_array=False),
                # 4. Values multiplier (unit conversion)
                Preprocessor(self.__apply_multiplier, value_multiplier=value_multiplier, apply_on_array=False),
                # 5. PSD (Before)
                Preprocessor(self.__psd_gen,
                             sub_id=sub_id,
                             record_idx=record_idx,
                             group=group,
                             output_dir=output_dir, sub_dir="before", apply_on_array=False),
                # 6. Detect bad channels
                Preprocessor(self.__auto_detect_bads, ransac=ransac, apply_on_array=False),
                ],
            )

            # Validates good recordings (by tolerance of bad ch) before the heavy processing
            ch_bads = len(record.raw.info["bads"])
            ch_total = len(mne.pick_types(record.raw.info, eeg=True, exclude=[]))
            if ch_bads > int(ch_total * bad_ch_tolerance):
                print(f"-> Dropping subject {sub_id} (Too many bad channels: {ch_bads}/{ch_total})")

                # Record the reason for dropping the subject
                self.__record_dropped(output_dir, rec_key, {
                    "subject": sub_id,
                    "reason": "Too many bad channels",
                    "bad_channels": record.raw.info["bads"],
                    "n_bad_channels": ch_bads,
                    "n_total_channels": ch_total,
                })

                return False
            else:
                # Stage 2: artifact removal, interpolation, reference, filtering and resampling
                preprocess(
                    single_ds,
                    [
                    # 7. Eliminate EOG/ECG artifacts (ICA is fit on good channels only, so it runs before interpolation)
                    Preprocessor(self.__eliminate_artifacts, run=treat_artifacts, ica=ica, apply_on_array=False),
                    # 8. Drop aux channels (non-eeg)
                    Preprocessor(self.__drop_aux_channels, apply_on_array=False),
                    # 9. Interpolate bad channels (rebuilt from the artifact-free neighbors)
                    Preprocessor(self.__interpolate_bads_if_any, apply_on_array=False),
                    # 10. Set A1/A2 reference
                    Preprocessor(self.__define_reference, ref_channels=ref_channels, apply_on_array=False),
                    # 11. Notch filter (power line noise)
                    Preprocessor(self.__notch_filter, notch_freqs=notch_freqs, apply_on_array=False),
                    # 12. Band-pass filter
                    Preprocessor(
                        "filter",
                        l_freq=l_freq,
                        h_freq=h_freq,
                        method="fir",
                        fir_design="firwin",
                    ),
                    # 13. Downsampling
                    Preprocessor("resample", sfreq=down_freq),
                    # 14. PSD (After)
                    Preprocessor(self.__psd_gen,
                                 sub_id=sub_id,
                                 record_idx=record_idx,
                                 group=group,
                                 output_dir=output_dir,
                                 sub_dir="after", apply_on_array=False),
                    ],
                )


                # Windowing and immediate recording saving -> Save by subject avoids accumulate very many window recording in RAM
                windows = create_fixed_length_windows(
                    single_ds,
                    start_offset_samples=0,
                    stop_offset_samples=None,
                    window_size_samples=window_size,
                    window_stride_samples=window_size,
                    on_last_window='drop',
                    preload=True,
                )

                windows.save(str(subject_path), overwrite=overwrite, offset=record_idx)

                return True

        except Exception as e:
            print(f"\n[ERROR] Failed processing subject {sub_id}: {e}\n")
            # Record the error, so the subject is skipped on the next run (use overwrite=True to retry)
            self.__record_dropped(output_dir, rec_key, {
                "subject": sub_id,
                "reason": "Processing error",
                "error": str(e),
            })
            return False
        finally:
            # Agressive cleanup by subject
            del single_ds
            del windows
            plt.close('all')

            # Release the preprocessed raw (the dataset keeps a reference to each record, so it would stay in RAM)
            record.raw = None

            # self.__print_section(f"filenames for {sub_id}: {record.raw.filenames}")
            # if hasattr(record, 'raw') and record.raw is not None:
            #     record.raw.close() # Close open file descriptors

            gc.collect()
    
    def preprocess_and_window(
        self,
        output_dir:str, 
        channels_treatment:Callable=None, 
        l_freq:float=1, 
        h_freq:float=40, 
        down_freq:float=200, 
        bad_ch_tolerance:float=0.15, 
        ransac:bool=True, 
        ica:bool=True, 
        window_size:int=512, # No overlapping, by default
        n_jobs:int=4, 
        ref_channels:list[str]=[], 
        overwrite:bool=False,
        montage:str="",
        treat_artifacts:bool=True,
        value_multiplier:float=1.0,
        notch_freqs:list[float]=[]): # Power line frequencies removed by notch filter (e.g. [50, 60]). Empty = no notch
        """
        Main function for optimized large dataset paralell processing.
        """
        os.makedirs(self.DATASETS_DIR / output_dir, exist_ok=True)

        # Set default channel treatment
        if(channels_treatment is None): channels_treatment = self.__default_ch_treatment

        # Fixed montage (if given)
        if(montage): self.montage = montage

        # Auto-detect the montage here, once: workers can't ask for input() and don't share self.montage
        if(not self.montage):
            first_record = self.dataset.datasets[0]
            self.__investigate_montage(channels_treatment(first_record.raw.copy()))
            first_record.raw = None # Release it (it is lazily reloaded when processed)

        print(f"[Batch Preprocessing] Starting batch processing of {len(self.dataset.datasets)} subjects with n_jobs={n_jobs}...")

        # Saving preprocessing parameters in json
        params = {
            "l_freq": l_freq,
            "h_freq": h_freq,
            "down_freq": down_freq,
            "bad_ch_tolerance": bad_ch_tolerance,
            "ransac":ransac,
            "ica":ica,
            "window_size":window_size,
            "n_jobs":n_jobs,
            "ref_channels":ref_channels,
            "montage": self.montage,
            "treat_artifacts":treat_artifacts,
            "value_multiplier":value_multiplier,
            "notch_freqs":notch_freqs
        }

        # Writes into json
        try:
            with open(f"{self.DATASETS_DIR}/{output_dir}/params.json", "r") as f:
                loaded_params = json.load(f)
            print("Already existing params.json found")
            dropped = loaded_params.pop("dropped", {})

            # Compare the saved settings with the new ones (n_jobs only changes the speed, not the result)
            # json round trip, so the new settings have the same types as the saved ones (e.g. tuple -> list)
            new_params = json.loads(json.dumps(params))
            different = [key for key in new_params if key != "n_jobs" and loaded_params.get(key, "(not saved)") != new_params[key]]
            if(different):
                self.__print_section("SAVED PARAMS.JSON vs NEW SETTINGS")
                print(f"{'Setting':<20}{'Saved (params.json)':<28}{'New':<28}")
                for key in new_params:
                    print(f"{key:<20}{str(loaded_params.get(key, '(not saved)')):<28}{str(new_params[key]):<28}{'<- different' if key in different else ''}")
                r = None
                while(r != "y" and r != "n"):
                    r = str(input("Settings are different. Use the saved params.json? (overwrite params.json with new settings will reprocess all subjects) (y/n): ")).lower()
                if(r == "y"):
                    # Saved settings replace the new ones (keys missing in an old params.json keep the new value)
                    saved = {key: loaded_params.get(key, value) for key, value in params.items()}
                    saved["n_jobs"] = n_jobs
                    params = saved
                    l_freq, h_freq, down_freq, bad_ch_tolerance = saved["l_freq"], saved["h_freq"], saved["down_freq"], saved["bad_ch_tolerance"]
                    ransac, ica, window_size, ref_channels = saved["ransac"], saved["ica"], saved["window_size"], saved["ref_channels"]
                    treat_artifacts, value_multiplier, notch_freqs = saved["treat_artifacts"], saved["value_multiplier"], saved["notch_freqs"]
                    self.montage = saved["montage"]
                else:
                    # Recordings processed with the old settings would be mixed with the new ones: reprocess everyone,
                    # and forget the dropped subjects (they were dropped with the old settings)
                    overwrite = True
                    dropped = {}
            params["dropped"] = dropped
        except FileNotFoundError: # Other errors (e.g. corrupted json) must stop, or the dropped subjects would be lost
            print("params.json not found. Creating new one.")

        with open(f"{self.DATASETS_DIR}/{output_dir}/params.json", "w") as f:
            json.dump(params, f, indent=4)

        # Lock file used by FileLock in the workers (deleted when finished)
        lock_path = self.DATASETS_DIR / output_dir / "params.json.lock"
        with open(lock_path, "w") as f:
            f.write("temporary file for params.json overwriting control when using workers")

        # Paralell processing using joblib (isolated chunk processing). n_jobs=1 runs sequentially in this process

        # Detach the dataset while the workers run (self is sent to every worker, so it would be copied once per subject)
        dataset = self.dataset
        self.dataset = None
        try:
            # mmap_mode='c': joblib sends large arrays (e.g. preloaded MOABB raws) as READ-ONLY memmaps by default, so in-place
            # MNE operations fail ("assignment destination is read-only"). Copy-on-write keeps them shared and writable
            results = joblib.Parallel(n_jobs=n_jobs, backend='loky', mmap_mode='c')(
                joblib.delayed(self._process_single_subject)(
                    record=record,
                    output_dir=output_dir,
                    channels_treatment=channels_treatment,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    down_freq=down_freq,
                    bad_ch_tolerance=bad_ch_tolerance,
                    ransac=ransac,
                    ica=ica,
                    window_size=window_size,
                    ref_channels=ref_channels,
                    overwrite=overwrite,
                    montage=montage,
                    treat_artifacts=treat_artifacts,
                    value_multiplier=value_multiplier,
                    notch_freqs=notch_freqs,
                    record_idx=record_idx
                )
                for record_idx, record in enumerate(dataset.datasets)
            )
        finally:
            self.dataset = dataset
            lock_path.unlink(missing_ok=True)

        print(f"Finished Processing. Succesfully processed subjects: {sum(results)}/{len(self.dataset.datasets)}")