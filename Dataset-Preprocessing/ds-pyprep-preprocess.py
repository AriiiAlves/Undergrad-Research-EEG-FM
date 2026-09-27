# Importa utilitarios de sistema e manipulacao de diretorios
import os
from pathlib import Path
import gc

# Importa bibliotecas para plotagem grafica, MNE para dados eletrofisiologicos e arrays numericos
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import time

# Importa o modulo eegdash e o construtor de datasets
import eegdash
from eegdash import EEGDashDataset
# Importa estilos visuais padronizados do EEGDash
from eegdash.viz import style_figure, use_eegdash_style

# Dataset
from eegdash.dataset import DS006576

## Preprocessing
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from pyprep import PrepPipeline
from braindecode.datasets import BaseConcatDataset

from pyprep.find_noisy_channels import NoisyChannels
from mne.preprocessing import ICA, compute_proj_eog, compute_proj_ecg

# Extra
import re

# Forca o backend matplotlib para o navegador do MNE para permitir captura estatica das figuras
mne.viz.set_browser_backend("matplotlib")

# Aplica o tema visual do EEGDash
use_eegdash_style()

# Define o caminho do diretorio de cache persistente
cache_dir = os.environ.get("EEGDASH_CACHE_DIR", str(Path.home() / ".eegdash_cache"))
print(f"eegdash {eegdash.__version__}; cache_dir={cache_dir}")

# Triggers EEG download
# for record in dataset.datasets:
#     try:
#         _ = record.raw
#     except Exception as e:
#         print(f"Skipping failed subject: {e}")

L_FREQ = 1
H_FREQ = 40
TARGET_SFREQ = 200
BAD_CH_TOLERANCE = 0.15
BAD_CH_TOLERANCE = 0.15
OUTPUT_DIR = "./ds1-pre-processed-eeg-pyprep"

# Define dataset
BEGIN_SUBJECT = 104
END_SUBJECT = 110

#subjects = [str(x) for x in range(BEGIN_SUBJECT,END_SUBJECT+1)]
subjects = ["108","110","112"]
dataset = DS006576(cache_dir="./data", subject=subjects) # Preload = True downloads the entire dataset.

# for record in dataset.datasets:
#     try:
#         # Attempt to check file existence via raw filenames
#         if record.raw.filenames and os.path.exists(record.raw.filenames[0]):
#             valid_datasets.append(record)
#         else:
#             print(f"Skipping: Path missing for subject {record.description['subject']}")
#     except (FileNotFoundError, OSError, Exception) as e:
#         print(f"Skipping subject {record.description["subject"]} due to missing data file: {e}")

# Re-wrap valid list into the container

os.makedirs("./subjects-analysis", exist_ok=True) # Ensure that the folder already exists
os.makedirs("./subjects-analysis/after-pyprep", exist_ok=True) # Ensure that the folder already exists
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure that the folder already exists

# Definir os parâmetros do PREP
prep_params = {
    "ref_chs": "eeg",          # Canais usados para a referência
    "reref_chs": "eeg",        # Canais a serem re-referenciados
    "line_freqs": [60.0], # Frequências de rede elétrica (ex: 50Hz no Brasil/Europa)
}

dataset_processed = []

# Custom function: rename EEG channels
def rename_channels(raw):
    rename_dict = {ch: ch.split('_')[1] for ch in raw.ch_names if '_' in ch}
    raw.rename_channels(rename_dict)

    return raw

# PRE-PROCESSING
for idx,record in enumerate(dataset.datasets):
    print(f"\n[PRE-PROCESSING] Processing subject {record.description["subject"]}...{idx}/{len(dataset.datasets)-1}\n")
    rename_channels(record.raw)
    # 1. Executes o PREP pipeline
    print(f"\n[PRE-PROCESSING] Executing PREP pipepline\n")
    prep = PrepPipeline(record.raw, prep_params, montage="colin27_1020", ransac=True)
    prep.fit()
    
    # 2. Updates raw with PREP clean signal
    print(f"\n[PRE-PROCESSING] Updating raw\n")
    record.raw = prep.raw

    # 3. Resampling
    record.raw.resample(TARGET_SFREQ, verbose=False)

    # # 4. Aplica o padrão desejado: re-referencia para as mastoides A1 e A2
    print(f"\n[PRE-PROCESSING] Applying A1/A2 reference\n")
    reref_channels = [ch for ch in ["A1", "A2"] if ch in record.raw.ch_names]
    if reref_channels:
        record.raw.set_eeg_reference(ref_channels=reref_channels, projection=False)

    # 5. Adiciona à lista processada
    if len(record.raw.info["bads"]) <= int(len(record.raw.ch_names) * BAD_CH_TOLERANCE):
        print(f"\n[PRE-PROCESSING] subject {record.description["subject"]} added to pre-processed dataset.\n")
        dataset_processed.append(record)
    else:
        print(f"\n[PRE-PROCESSING] Bad recording. Dropping.\n")

dataset_processed = BaseConcatDataset(dataset_processed)

for record in dataset_processed.datasets:
    print(f"\n[PSD Analysis] Analysing subject {record.description["subject"]}...\n")
    # Generate PSD and PSD Topomap for each subject
    fig1_path = f"./ds1-subjects-analysis/after-pyprep/psd_{record.description["subject"]}_after_pyprep.png"
    fig2_path = f"./ds1-subjects-analysis/after-pyprep/psd_topomap_{record.description["subject"]}_after_pyprep.png"

    psd = record.raw.compute_psd(verbose=True)
    fig1 = psd.plot(show=False);
    fig2 = psd.plot_topomap(show=False);

    fig1.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD")
    fig2.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD Topomap")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")

# Create windows for this single record
windows = create_fixed_length_windows(
    dataset_processed,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=4 * TARGET_SFREQ,
    window_stride_samples=4 * TARGET_SFREQ,
    on_last_window='drop',
    preload=False, # Keep False for lazy loading from disk later!
)

# Save this individual subject's windows to disk
os.makedirs(OUTPUT_DIR, exist_ok=True)
windows.save(OUTPUT_DIR, overwrite=True)