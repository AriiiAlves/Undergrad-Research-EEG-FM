from models_bench import BenchBrainomni, BenchCBraMod, BenchLabram
from preprocessing import DatasetManager
import mne

EEGDASH_DS_NAMES = [
    "ds006576", # (EEGDash) The role of REM sleep in neural differentiation of memories in the hippocampus -> montage="standard_1020"
    "ds007609", # (EEGDash) Resting-State EEG and Trait Anxiety -> montage="EGI_256"
    "ds007537", # (OpenNeuro - EEGDash) A multimodal dataset of EEG, eye-tracking, and physiological signals during smartphone interaction -> 
]

EEGDASH_DS_MONTAGES = [
    "colin27_1020",
    "EGI_256",
    "colin27_1020",
]

MOABB_DS_NAMES = [
    "BNCI2014_001"
]

def ds006576_ch_treatment(raw:mne.io.Raw):
    #raw.drop_channels(['A1','A2']) # -> Reference channels??
    raw.rename_channels({ch:ch.split('_')[-1] for ch in raw.ch_names})
    return raw

def preprocess():
    for DS_NAME in MOABB_DS_NAMES:
        dsManager = DatasetManager()
        dsManager.load_moabb_dataset(DS_NAME.upper(), subject_limit=0)
        dsManager.inspect_dataset()
        dsManager.preprocess_and_window(
            output_dir = DS_NAME.upper(), 
            channels_treatment = None,
            l_freq = 0.1, # minimal floor l_freq of models.
            h_freq = 75, # Minimal ceil h_freq of models.
            down_freq = 256, # The models needs 200Hz-256Hz. So, i'm setting the maximum here.
            notch_freqs = [50, 60], # Remove both problematic frequencies
            bad_ch_tolerance = 0.2,
            ransac = True,
            ica = True,
            window_size=512,
            n_jobs=4,
            ref_channels=[],
            overwrite=False,
            montage=None, # Auto-detect
            treat_artifacts=True,
            value_multiplier=1.0 # Necessary to convert the dataset to V
        )

    for DS_NAME, DS_MONTAGE in zip(EEGDASH_DS_NAMES, EEGDASH_DS_MONTAGES):
        dsManager = DatasetManager()
        dsManager.load_eegdash_dataset(DS_NAME.lower(), subject_limit=0) 
        dsManager.inspect_dataset()
        dsManager.preprocess_and_window(
            output_dir = DS_NAME.upper(), 
            channels_treatment = ds006576_ch_treatment if DS_NAME == "ds006576" else None,
            l_freq = 0.1, # minimal floor l_freq of models.
            h_freq = 75, # Minimal ceil h_freq of models.
            down_freq = 256, # The models needs 200Hz-256Hz. So, i'm setting the maximum here.
            notch_freqs = [50, 60], # Remove both problematic frequencies
            bad_ch_tolerance = 0.2,
            ransac = True,
            ica = True,
            window_size=512,
            n_jobs=4,
            ref_channels=[],
            overwrite=False,
            montage=DS_MONTAGE,
            treat_artifacts=True,
            value_multiplier=1.0 # Necessary to convert the dataset to V
        )

def bench():
    for DS_NAME in MOABB_DS_NAMES: 
        base_windows_path = f"./PreProcessed-Datasets/{DS_NAME.upper()}/PreProcessed-EEG"

        benchBrainomni = BenchBrainomni(base_windows_path, DS_NAME.upper())
        benchLabram = BenchLabram(base_windows_path, DS_NAME.upper())
        benchCbramod = BenchCBraMod(base_windows_path, DS_NAME.upper())

        # One point by window (no aggregation)
        benchBrainomni.use_latent_space("windows")
        benchBrainomni.umap()
        benchLabram.use_latent_space("windows")
        benchLabram.umap()
        benchCbramod.use_latent_space("windows")
        benchCbramod.umap()

        # Every aggregation (avg/median/std), by recording and by subject
        for group_by in ["recording", "subject"]:
            for mode in ["avg", "median", "std"]:
                benchBrainomni.use_latent_space(mode, group_by=group_by)
                benchBrainomni.umap()
                benchLabram.use_latent_space(mode, group_by=group_by)
                benchLabram.umap()
                benchCbramod.use_latent_space(mode, group_by=group_by)
                benchCbramod.umap()
    
    for DS_NAME in EEGDASH_DS_NAMES: 
        base_windows_path = f"./PreProcessed-Datasets/{DS_NAME.upper()}/PreProcessed-EEG"

        benchBrainomni = BenchBrainomni(base_windows_path, DS_NAME.upper())
        benchLabram = BenchLabram(base_windows_path, DS_NAME.upper())
        benchCbramod = BenchCBraMod(base_windows_path, DS_NAME.upper())

        # One point by window (no aggregation)
        benchBrainomni.use_latent_space("windows")
        benchBrainomni.umap()
        benchLabram.use_latent_space("windows")
        benchLabram.umap()
        benchCbramod.use_latent_space("windows")
        benchCbramod.umap()

        # Every aggregation (avg/median/std), by recording and by subject
        for group_by in ["recording", "subject"]:
            for mode in ["avg", "median", "std"]:
                benchBrainomni.use_latent_space(mode, group_by=group_by)
                benchBrainomni.umap()
                benchLabram.use_latent_space(mode, group_by=group_by)
                benchLabram.umap()
                benchCbramod.use_latent_space(mode, group_by=group_by)
                benchCbramod.umap()

if __name__ == "__main__":
    preprocess()
    bench()