#from gen_umap import gen_umap
from models_bench import BenchBrainomni
from preprocessing import preprocess, batch_preprocess_dataset
from eegdash import EEGDashDataset
import numpy as np

if __name__ == "__main__":
    # --------------------------------------------------------
    ## Define dataset
    #BEGIN_SUBJECT = 104
    #END_SUBJECT = 110
    #subjects = [str(x) for x in range(BEGIN_SUBJECT,END_SUBJECT+1)]
    #subjects = ["108","110","112"]
    #dataset = DS006576(cache_dir="./data", subject=subjects) # Preload = True downloads the entire dataset.
    #dataset = DS006576(cache_dir="./data") # Preload = True downloads the entire dataset.
    dataset = EEGDashDataset(cache_dir="./data", dataset='ds006866')
    # --------------------------------------------------------

    #dataset = EEGDashDataset(cache_dir="./data", dataset='ds004019')

    # Custom function: rename EEG channels
    def channels_treatment(raw):
        '''
        print(dataset.datasets[0].raw.ch_names)

        ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 
        'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 
        'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', 'HEO', 'VEO', 'EKG', 'GSR', 'Trigger']
        '''
        
        # Manual mapping
        rename_dict = {
            'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2', 
            'AF3': 'AF3', 'AF4': 'AF4', 
            'F7': 'F7', 'F5': 'F5', 'F3': 'F3', 'F1': 'F1', 'FZ': 'Fz', 
            'F2': 'F2', 'F4': 'F4', 'F6': 'F6', 'F8': 'F8', 
            'FT7': 'FT7', 'FC5': 'FC5', 'FC3': 'FC3', 'FC1': 'FC1', 'FCZ': 'FCz', 
            'FC2': 'FC2', 'FC4': 'FC4', 'FC6': 'FC6', 'FT8': 'FT8', 
            'T7': 'T7', 'C5': 'C5', 'C3': 'C3', 'C1': 'C1', 'CZ': 'Cz', 
            'C2': 'C2', 'C4': 'C4', 'C6': 'C6', 'T8': 'T8', 
            'M1': 'M1', 'TP7': 'TP7', 'CP5': 'CP5', 'CP3': 'CP3', 'CP1': 'CP1', 
            'CPZ': 'CPz', 'CP2': 'CP2', 'CP4': 'CP4', 'CP6': 'CP6', 'TP8': 'TP8', 'M2': 'M2', 
            'P7': 'P7', 'P5': 'P5', 'P3': 'P3', 'P1': 'P1', 'PZ': 'Pz', 
            'P2': 'P2', 'P4': 'P4', 'P6': 'P6', 'P8': 'P8', 
            'PO7': 'PO7', 'PO5': 'PO5', 'PO3': 'PO3', 'POZ': 'POz', 
            'PO4': 'PO4', 'PO6': 'PO6', 'PO8': 'PO8', 
            'CB1': 'CB1', 'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2', 'CB2': 'CB2', 
            'HEO': 'HEO', 'VEO': 'VEO', 'EKG': 'EKG', 'GSR': 'GSR', 'Trigger': 'Trigger'
        }

        fixes = {ch: rename_dict[k] for ch in raw.ch_names for k in rename_dict if ch.upper() == k.upper()}
        raw.rename_channels(fixes)

        # Channel type definition
        type_mapping = {}
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if any(x in ch_upper for x in ['HEO', 'VEO']):
                type_mapping[ch] = 'eog'
            elif any(x in ch_upper for x in ['EKG']):
                type_mapping[ch] = 'ecg'
            elif any(x in ch_upper for x in ['TRIGGER', 'STATUS', 'GSR', 'CB1', 'CB2', 'EMG1', 'EMG2']):
                type_mapping[ch] = 'misc'
            else:
                type_mapping[ch] = 'eeg'
        
        if type_mapping:
            raw.set_channel_types(type_mapping)

        return raw

    batch_preprocess_dataset(
        dataset = dataset, 
        output_dir = "DS006866", 
        channels_treatment = channels_treatment,
        l_freq = 1,
        h_freq = 40,
        down_freq = 200,
        bad_ch_tolerance = 0.15,
        ransac = True,
        ica = True,
        window_size=512,
        n_jobs=2,
        ref_channels=['M1', 'M2'],
        overwrite=False
    )

    base_windows_path = "/home/arielalves/Undergrad-Research-EEG-FM/Full-Preprocessing-Umap/DS006866/pre-processed-eeg" #(New)
    #base_windows_path = "/home/arielalves/Undergrad-Research-EEG-FM/Full-Preprocessing-Umap/DS006576/pre-processed-eeg" #(OK)

    benchBrainomni = BenchBrainomni(base_windows_path, "DS006866")
    benchBrainomni.use_avg_latent_space(z_score=True)
    benchBrainomni.umap()
    benchBrainomni.use_avg_latent_space(z_score=False)
    benchBrainomni.umap()