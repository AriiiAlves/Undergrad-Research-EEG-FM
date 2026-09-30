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
from braindecode import preprocessing as bd_preproc

## Preprocessing
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA, compute_proj_eog, compute_proj_ecg

import joblib  # Para paralelismo seguro por sujeito
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
    
    # Drop aux channels (non-eeg) (optimize memory use)
    aux_channels = []
    for ch in raw.ch_names:
        if('eeg' not in raw.get_channel_types(picks=ch)): aux_channels.append(ch)
    raw.drop_channels(aux_channels)
    return raw

# Custom function: Interpolate bad channels if any were detected
def interpolate_bads_if_any(raw):
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=False) # Feedback of interpolated channels
        # This info will be used to decide which records exclude/mantain
    return raw

# Custom function: set custom reference
def define_reference(raw, ref_channels):
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

# Custom function: verify_montage
def verify_montage(raw):
    for ch_info in raw.info['chs']:
        ch_name = ch_info['ch_name']
        loc = ch_info['loc'][:3]  # Pega as coordenadas X, Y, Z (primeiros 3 elementos do array loc)
        
        # Se houver algum valor NaN ou se todas as coordenadas forem 0.0
        if (np.isnan(loc).any() or np.all(loc == 0)) and ('eeg' in raw.get_channel_types(picks=ch_name)):
            assert False, f"CHANNEL WITH INVALID POSITION (NaN/Zero): {ch_name} -> Coord: {loc} -> Type: {raw.get_channel_types(picks=ch_name)}"

    return raw

# Custom function: PSD generator (before preprocessing)
def psd_gen_before(raw, sub_id, group, output_dir):
    psd_dir = Path(output_dir) / "subjects-analysis" / "before"
    psd_dir.mkdir(parents=True, exist_ok=True)
    
    psd = raw.compute_psd(verbose=False)
    fig1 = psd.plot(show=False)
    fig2 = psd.plot_topomap(show=False)
    fig1.suptitle(f"Subject {sub_id} ({group}) PSD")
    fig2.suptitle(f"Subject {sub_id} ({group}) PSD Topomap")
    
    fig1.savefig(psd_dir / f"psd_{sub_id}_before.png", dpi=150, bbox_inches="tight") # DPI 150 economiza espaço/tempo
    fig2.savefig(psd_dir / f"psd_topomap_{sub_id}_before.png", dpi=150, bbox_inches="tight")
    plt.close('all')

def process_single_subject(record, output_dir, channels_treatment, l_freq, h_freq, down_freq, bad_ch_tolerance, ransac, ica, window_size, ref_channels, overwrite):
    """
    Process a single subject, saving the result directly on disk to save RAM/VRAM
    """
    sub_id = record.description.get('subject', 'unknown')

    subject_path = Path(output_dir) / "pre-processed-eeg" / f"sub-{sub_id}"
    if(subject_path.exists() and not overwrite): 
        print(f"[Single Subject Preprocessing] Skipping {sub_id}: already processed")
        return True

    subject_path.mkdir(parents=True, exist_ok=True)

    # Var initializing (if subject got dropped, assure the variables to be deleted exist)
    single_ds = None
    windows = None
    
    try:
        # 1. Validates physical file
        if not (record.raw.filenames and os.path.exists(record.raw.filenames[0])):
            print(f"[Skipping] Path missing for subject {sub_id}")
            return False

        # 2. Pipeline de pré-processamento aplicado isoladamente ao registro
        single_ds = BaseConcatDataset([record])
        
        preprocess(
            single_ds,
            [
            # Note: if apply_on_array=False, passes raw object. If not, passes raw underlying NumPy array.
            # 1. Channels treatment
            Preprocessor(channels_treatment, apply_on_array=False),
            # 2. Set montage
            Preprocessor("set_montage", montage="colin27_1020", on_missing="ignore"),
            # 3. Verify NaN in eeg channels
            Preprocessor(verify_montage, apply_on_array=False),
            # 4. PSD (Before)
            Preprocessor(psd_gen_before, sub_id=sub_id, group=record.description.get('group', ''), output_dir=output_dir, apply_on_array=False),
            # 5. Detect bad channels
            Preprocessor(auto_detect_bads, ransac=ransac, apply_on_array=False),
            # 6. Interpolate bad channels
            Preprocessor(interpolate_bads_if_any, apply_on_array=False),
            # 7. Eliminate EOG/ECG artifacts
            Preprocessor(eliminate_artifacts, ica=ica, apply_on_array=False),
            # 8. Set A1/A2 reference
            Preprocessor(define_reference, ref_channels=ref_channels, apply_on_array=False),
            # 9. Band-pass filter
            Preprocessor(
                "filter",
                l_freq=l_freq,
                h_freq=h_freq,
                method="fir",
                fir_design="firwin",
            ),
            # 10. Downsampling
            Preprocessor("resample", sfreq=down_freq),
            ],
        )

        # Validates good recordings (by tolerance of bad ch)
        ch_bads = len(record.raw.info["bads"])
        ch_total = len(record.raw.ch_names)
        if ch_bads > int(ch_total * bad_ch_tolerance):
            print(f"-> Dropping subject {sub_id} (Too many bad channels: {ch_bads}/{ch_total})")
            return False

        # PSD generator (after preprocessing)
        psd_dir = Path(output_dir) / "subjects-analysis" / "after"
        psd_dir.mkdir(parents=True, exist_ok=True)
        
        psd = record.raw.compute_psd(verbose=False)
        fig1 = psd.plot(show=False)
        fig2 = psd.plot_topomap(show=False)
        fig1.suptitle(f"Subject {sub_id} ({record.description.get('group', '')}) PSD")
        fig2.suptitle(f"Subject {sub_id} ({record.description.get('group', '')}) PSD Topomap")
        
        fig1.savefig(psd_dir / f"psd_{sub_id}_after.png", dpi=150, bbox_inches="tight") # DPI 150 economiza espaço/tempo
        fig2.savefig(psd_dir / f"psd_topomap_{sub_id}_after.png", dpi=150, bbox_inches="tight")
        plt.close('all')

        # Windowing and immediate recording saving -> Save by subject avoids accumulate very many window recording in RAM
        windows = create_fixed_length_windows(
            single_ds,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=window_size,
            window_stride_samples=window_size,
            on_last_window='drop',
            preload=True
        )

        windows.save(str(subject_path), overwrite=True)

        return True

    except Exception as e:
        print(f"[Error] Failed processing subject {sub_id}: {e}")
        return False
    finally:
        # Agressive cleanup by subject
        del single_ds
        del windows
        if hasattr(record, 'raw') and record.raw is not None:
            record.raw.close() # Close open file descriptors
        gc.collect()

def batch_preprocess_dataset(dataset, output_dir, channels_treatment, l_freq=1, h_freq=40, down_freq=200, 
                             bad_ch_tolerance=0.15, ransac=True, ica=True, window_size=512, n_jobs=4, ref_channels=[], overwrite=False):
    """
    Main function for optimized large dataset paralell processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Batch Preprocessing] Starting batch processing of {len(dataset.datasets)} subjects with n_jobs={n_jobs}...")

    # Paralell processing using joblib (isolated chunk processing)
    results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
        joblib.delayed(process_single_subject)(
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
            overwrite=overwrite
        )
        for record in dataset.datasets
    )
    
    # FOR TEST 
    # process_single_subject(dataset.datasets[0], output_dir, channels_treatment, l_freq, h_freq, down_freq, 
    #         bad_ch_tolerance, ransac, ica, window_size)

    print(f"Finished Processing. Succesfully processed subjects: {sum(results)}/{len(dataset.datasets)}")