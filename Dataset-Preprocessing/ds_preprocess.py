# Import system utils
import os
from pathlib import Path
import gc

# Import libraries for graphic plot, MNE for EEG data and numpy for arrays
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import time

# Import eegdash module
import eegdash
from eegdash import EEGDashDataset
# Import visual pattern styles from EEGDash
from eegdash.viz import style_figure, use_eegdash_style

# Dataset
from braindecode.datasets import BaseConcatDataset

## Preprocessing
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA, compute_proj_eog, compute_proj_ecg

# Forces matplotlib as MNE browser backend
mne.viz.set_browser_backend("matplotlib")

# Apply visual theme from EEGDash
use_eegdash_style()

# Custom function: Detect bad channels
def auto_detect_bads(raw, ransac=True):
    nc = NoisyChannels(raw, random_state=42)
    nc.find_all_bads(ransac=ransac) # Set RANSAC = True if do you have a good computer.
    raw.info['bads'] = nc.get_bads()
    return raw

# Custom function: Eliminate artifacts caused by eyes/heart.
def eliminate_artifacts(raw, ica=True, eog=True, ecg=True):
    if(not eog and not ecg):
        print("[Eliminate artifacts] Bad arguments. Skipping.")
        return

    # Verify existence of EOG/ECG channels
    has_eog = len(mne.pick_types(raw.info, meg=False, eeg=False, eog=True, ecg=False)) > 0
    has_ecg = len(mne.pick_types(raw.info, meg=False, eeg=False, eog=False, ecg=True)) > 0

    # If EOG/ECG channels doesn't exists, pick channels next to eyes
    if has_eog:
        ch_eog = None
    else:
        ch_next = ['Fp1', 'FP1', 'Fp2', 'FP2', 'AF3', 'AF4']
        for ch in ch_next:
            if ch in raw.ch_names:
                ch_eog = ch
                break

    # ICA has a big computational cost
    if(ica):
        # Filtered copy for stable ICA decomposition (ICA performs poorly if trained on low-frequency drifts (< 1.0Hz)
        raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design="firwin", verbose="ERROR")
        # ICA model
        ica = ICA(n_components=15, random_state=42, method="fastica")
        # Train ICA model with raw data
        ica.fit(raw_ica)

        bads = []
        # Find ocular artifacts using LOC and ROC
        if(eog and (has_eog or ch_eog)): 
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=ch_eog)
            bads.extend(eog_indices)
        # Find cardiac artifacts using ECG
        if(ecg and has_ecg): 
            ecg_indices, _ = ica.find_bads_ecg(raw, ch_name=None) # If ch_name = None, MNE checks raw for any channels marked as type
            bads.extend(ecg_indices)
        # Mark components to remove
        ica.exclude = list(set(bads))
        print(f"[Eliminate artifacts] ICA removed components: {ica.exclude}")
        # Remove components and reconstruct EEG signal. Applies to original raw.
        ica.apply(raw)

    # SSP is faster than ICA
    else:
        if eog and (has_eog or ch_eog):
            projs_eog, _ = compute_proj_eog(raw, ch_name=ch_eog, n_grad=0, n_mag=0, n_eeg=1, verbose="ERROR")
            raw.add_proj(projs_eog)

        if ecg and has_ecg:
            projs_ecg, _ = compute_proj_ecg(raw, ch_name=None, n_grad=0, n_mag=0, n_eeg=1, verbose="ERROR")
            raw.add_proj(projs_ecg)

        raw.apply_proj()
    
    # Drop aux channels (optimize memory use)
    aux_channels = ['LOC', 'ROC', 'ECG1', 'ECG2', 'EMG1', 'EMG2', 'Status']
    raw.drop_channels([ch for ch in aux_channels if ch in raw.ch_names])
    return raw

# Custom function: Interpolate bad channels if any were detected
def interpolate_bads_if_any(raw):
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=False) # Feedback of interpolated channels
        # This info will be used to decide which records exclude/mantain
    return raw

# Custom function: set A1/A2 reference
def define_reference(raw):
    raw.set_channel_types({'A1': 'eeg', 'A2': 'eeg'})
    raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    raw.drop_channels(['A1', 'A2'])
    return raw

# Custom function: drop non-EEG channels
def drop_non_eeg(raw):
    aux_channels = ['LOC', 'ROC', 'EMG1', 'EMG2', 'ECG1', 'ECG2', 'Status']
    channels_to_drop = [ch for ch in aux_channels if ch in raw.ch_names]
    raw.drop_channels(channels_to_drop)
    return raw

# Custom function: rename EEG channels
def rename_channels(raw):
    rename_dict = {ch: ch.split('_')[1] for ch in raw.ch_names if '_' in ch}
    raw.rename_channels(rename_dict)
    return raw

def preprocess(dataset, output_dir, l_freq = 1, h_freq = 40, down_freq = 200, bad_ch_tolerance = 0.15, ransac = True, ica = True, window_size = 512):
    """
    General function for EEG preprocessing. It requires a braindecode EEG dataset,
    """
    #Triggers EEG download
    #for record in dataset.datasets:
    #    try:
    #        _ = record.raw
    #    except Exception as e:
    #        print(f"Skipping failed subject: {e}")

    os.makedirs(output_dir, exist_ok=True) # Ensure that the folder already exists
    os.makedirs(f"{output_dir}/subjects-analysis", exist_ok=True) # Ensure that the folder already exists
    os.makedirs(f"{output_dir}/subjects-analysis/after", exist_ok=True) # Ensure that the folder already exists

    valid_datasets = []
    for record in dataset.datasets:
        try:
            # Attempt to check file existence via raw filenames
            if record.raw.filenames and os.path.exists(record.raw.filenames[0]):
                valid_datasets.append(record)
            else:
                print(f"Skipping: Path missing for subject {record.description['subject']}")
        except (FileNotFoundError, OSError, Exception) as e:
            print(f"Skipping subject {record.description["subject"]} due to missing data file: {e}")

    # Re-wrap valid list into the container
    valid_datasets = BaseConcatDataset(valid_datasets)

    # Process all the records
    preprocess(
        valid_datasets,
        [
            # Note: if apply_on_array=False, passes raw object. If not, passes raw underlying NumPy array.
            
            # 1. Rename channels
            Preprocessor(rename_channels, apply_on_array=False),
            # 2. Set montage
            Preprocessor("set_montage", montage="colin27_1020", on_missing="ignore"),
            # 3. Detect bad channels
            Preprocessor(auto_detect_bads, ransac=ransac, apply_on_array=False),
            # 4. Interpolate bad channels
            Preprocessor(interpolate_bads_if_any, apply_on_array=False),
            # 5. Eliminate EOG/ECG artifacts
            Preprocessor(eliminate_artifacts, ica=ica, apply_on_array=False),
            # 6. Set A1/A2 reference
            Preprocessor(define_reference, apply_on_array=False),
            # 7. Band-pass filter
            Preprocessor(
                "filter",
                l_freq=l_freq,
                h_freq=h_freq,
                method="fir",
                fir_design="firwin",
            ),
            # 8. Notch filter (eliminates isolated frequencies)
            # Preprocessor(
            #     "notch_filter",
            #     freqs=[50.0],  # Use [60.0] if electrical network of 60Hz
            #     method="fir",
            #     fir_design="firwin"
            # )
            # 9. Downsampling
            Preprocessor("resample", sfreq=down_freq),
        ],
    )

    for record in valid_datasets.datasets:
        print(f"Processing subject {record.description["subject"]}...")
        # Generate PSD and PSD Topomap for each subject
        fig1_path = f"./{output_dir}/subjects-analysis/after/psd_{record.description["subject"]}_after.png"
        fig2_path = f"./{output_dir}/subjects-analysis/after/psd_topomap_{record.description["subject"]}_after.png"

        psd = record.raw.compute_psd(verbose=False)
        fig1 = psd.plot(show=False);
        fig2 = psd.plot_topomap(show=False);

        fig1.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD")
        fig2.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD Topomap")
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")

    filtered_dataset = []
    for record in valid_datasets.datasets:
        # Bad subject dropping
        ch_bads = len(record.raw.info["bads"])
        ch_total = len(record.raw.ch_names)
        if ch_bads <= int(ch_total * bad_ch_tolerance):
            filtered_dataset.append(record)
        else:
            print(f"-> Dropping subject {record.description["subject"]} (Too many bad channels : {ch_bads})")

    filtered_dataset = BaseConcatDataset(filtered_dataset)

    # Create windows for this single record
    windows = create_fixed_length_windows(
        filtered_dataset,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size,
        window_stride_samples=window_size,
        on_last_window='drop',
        preload=True
    )

    # Save this individual subject's windows to disk
    subject_path = f"{output_dir}/pre-processed-eeg"
    os.makedirs(subject_path, exist_ok=True)
    windows.save(subject_path, overwrite=True)