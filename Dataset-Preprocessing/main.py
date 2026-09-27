from ds_preprocess import preprocess
from eegdash.dataset import DS005407
from eegdash.dataset import DS006576

if __name__ == "__main__":
    # --------------------------------------------------------
    OUTPUT_DIR = "./DS006576"

    ## Define dataset
    #BEGIN_SUBJECT = 104
    #END_SUBJECT = 110
    #subjects = [str(x) for x in range(BEGIN_SUBJECT,END_SUBJECT+1)]
    #subjects = ["108","110","112"]
    #dataset = DS006576(cache_dir="./data", subject=subjects) # Preload = True downloads the entire dataset.
    dataset = DS006576(cache_dir="./data") # Preload = True downloads the entire dataset.
    # --------------------------------------------------------

    #dataset = EEGDashDataset(cache_dir="./data", dataset='ds004019')

    preprocess(
        dataset = dataset, 
        output_dir = OUTPUT_DIR, 
        l_freq = 1,
        h_freq = 40,
        down_freq = 200,
        bad_ch_tolerance = 0.15,
        ransac = True,
        ica = True,
        window_size=512
    )