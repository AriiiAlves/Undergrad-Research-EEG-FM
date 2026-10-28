from models_bench import BenchBrainomni, BenchCBraMod, BenchLabram
from preprocessing import DatasetManager
import mne

DS_NAME1 = "ds006576" # (EEGDash) The role of REM sleep in neural differentiation of memories in the hippocampus -> montage="standard_1020"
DS_NAME2 = "ds007609" # (EEGDash) Resting-State EEG and Trait Anxiety -> montage="EGI_256"
DS_NAME3 = "ds007537" # (OpenNeuro - EEGDash) A multimodal dataset of EEG, eye-tracking, and physiological signals during smartphone interaction -> 

DS_NAMES = [DS_NAME1, DS_NAME2, DS_NAME3]

def ds006576_ch_treatment(raw:mne.io.Raw):
    #raw.drop_channels(['A1','A2']) # -> Reference channels??
    raw.rename_channels({ch:ch.split('_')[-1] for ch in raw.ch_names})
    return raw

def preprocess():
    for DS_NAME in DS_NAMES:
        dsManager = DatasetManager()
        dsManager.load_eegdash_dataset(DS_NAME.lower(), subject_limit=0) 
        #dsManager.load_moabb_dataset(DS_NAME.lower(), subject_limit=0)
        #dsManager.inspect_dataset()
        dsManager.preprocess_and_window(
            output_dir = DS_NAME.upper(), 
            channels_treatment = ds006576_ch_treatment if DS_NAME == DS_NAME1 else None,
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

def bench():
    for DS_NAME in DS_NAMES: 
        base_windows_path = f"./PreProcessed-Datasets/{DS_NAME.upper()}/PreProcessed-EEG"

        benchBrainomni = BenchBrainomni(base_windows_path, DS_NAME.upper())
        benchLabram = BenchLabram(base_windows_path, DS_NAME.upper())
        benchCbramod = BenchCBraMod(base_windows_path, DS_NAME.upper())

        # Every aggregation (avg/windows/median/std)
        for mode in ["avg", "windows", "median", "std"]:
            benchBrainomni.use_latent_space(mode)
            benchBrainomni.umap()
            benchLabram.use_latent_space(mode)
            benchLabram.umap()
            benchCbramod.use_latent_space(mode)
            benchCbramod.umap()

if __name__ == "__main__":
    preprocess()
    bench()