from models_bench import BenchBrainomni
from preprocessing import DatasetManager
import mne

#DS_NAME = "ds006576" # (EEGDash) The role of REM sleep in neural differentiation of memories in the hippocampus -> montage="standard_1020"
#DS_NAME = "ds007609" # (EEGDash) Resting-State EEG and Trait Anxiety -> montage="EGI_256"
DS_NAME = "ds007537" # (OpenNeuro - EEGDash) A multimodal dataset of EEG, eye-tracking, and physiological signals during smartphone interaction -> 

def default_ch_treatment(raw):
    return raw

def ds006576_ch_treatment(raw:mne.io.Raw):
    #raw.drop_channels(['A1','A2']) # -> Reference channels??
    raw.rename_channels({ch:ch.split('_')[-1] for ch in raw.ch_names})
    return raw

def test_preprocessing():
    dsManager = DatasetManager()
    dsManager.load_eegdash_dataset(DS_NAME.lower(), subject_limit=0) 
    #dsManager.load_moabb_dataset(DS_NAME.lower(), subject_limit=0)
    #dsManager.inspect_dataset()
    dsManager.preprocess_and_window(
        output_dir = DS_NAME.upper(), 
        channels_treatment = ds006576_ch_treatment,
        l_freq = 0.5,
        h_freq = 40,
        down_freq = 200,
        bad_ch_tolerance = 0.2,
        ransac = True,
        ica = True,
        window_size=512,
        n_jobs=1,
        ref_channels=[],
        overwrite=False,
        montage="standard_1020",
        treat_artifacts=True
    )

def test_bench():
    base_windows_path = f"/home/arielalves/Undergrad-Research-EEG-FM/Full-Preprocessing-Umap/{DS_NAME.upper()}/pre-processed-eeg"

    benchBrainomni = BenchBrainomni(base_windows_path, DS_NAME.upper())

    # avg
    benchBrainomni.use_avg_latent_space(z_score=True, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_avg_latent_space(z_score=True, convert_to_uv=False)
    benchBrainomni.umap()
    benchBrainomni.use_avg_latent_space(z_score=False, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_avg_latent_space(z_score=False, convert_to_uv=False)
    benchBrainomni.umap()

    # all windows
    benchBrainomni.use_windows_latent_space(z_score=True, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_windows_latent_space(z_score=True, convert_to_uv=False)
    benchBrainomni.umap()
    benchBrainomni.use_windows_latent_space(z_score=False, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_windows_latent_space(z_score=False, convert_to_uv=False)
    benchBrainomni.umap()

    # # median windows
    benchBrainomni.use_median_latent_space(z_score=True, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_median_latent_space(z_score=True, convert_to_uv=False)
    benchBrainomni.umap()
    benchBrainomni.use_median_latent_space(z_score=False, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_median_latent_space(z_score=False, convert_to_uv=False)
    benchBrainomni.umap()

    # # std windows
    benchBrainomni.use_std_latent_space(z_score=True, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_std_latent_space(z_score=True, convert_to_uv=False)
    benchBrainomni.umap()
    benchBrainomni.use_std_latent_space(z_score=False, convert_to_uv=True)
    benchBrainomni.umap()
    benchBrainomni.use_std_latent_space(z_score=False, convert_to_uv=False)
    benchBrainomni.umap()

if __name__ == "__main__":
    #test_preprocessing()
    test_bench()