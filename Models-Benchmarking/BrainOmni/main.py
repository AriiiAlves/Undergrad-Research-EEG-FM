from gen_umap import gen_umap

if __name__ == "__main__":
    base_windows_path = "/home/arielalves/Undergrad-Research-EEG-FM/Dataset-Preprocessing/DS006576/pre-processed-eeg"
    gen_umap(base_windows_path, limit=1, option="windows_without_z_score")