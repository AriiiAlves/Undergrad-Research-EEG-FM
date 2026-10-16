# Coisas a se fazer

- Documentar como utilizar a função de benchmark de modelos, e as especificações.
- Corrigir falha em paralelizar (n_jobs > 1)
- Criar UMAPs que o Diego sugeriu

Ver isso melhor dos nomes dos canais...

# Corrigir bugs

- FileExistsError: Subdirectory DS006923/pre-processed-eeg/0 already exists. Please select a different directory, set overwrite=True, or resolve manually.

# `preprocessing` script

## Main function

Python script for EEG dataset preprocessing. 

```py
def batch_preprocess_dataset(
        dataset:BaseConcatDataset, 
        output_dir:str, 
        channels_treatment:Callable, 
        l_freq:float=1, 
        h_freq:float=40, 
        down_freq:float=200, 
        bad_ch_tolerance:float=0.15, 
        ransac:bool=True, 
        ica:bool=True, 
        window_size:int=512, 
        n_jobs:int=4, 
        ref_channels:list[str]=[], 
        overwrite:bool=False):
```

- `dataset` - Braindecode BaseConcatDataset instance
- `output_dir` - Name of directory where the preprocessed dataset will be stored
- `channels_treatment` - Function that receives mne `raw` instance and modify its channel names
- `l_freq` - High-pass filter frequency
- `h_freq` - Low-pass filter frequency
- `down_freq` - Downsampling frequency
- `bad_ch_tolerance` - Percentage (0.0 to 1.0) of bad channels necessary to drop a subject recording
- `ransac` - Set `True` to run RANSAC.
- `ica` - Set `True` to run ICA.
- `window_size` - Final recordings window size with no overlapping.
- `n_jobs` - Set `>1` to process multiple recordings at same time (paralell processing). WARNING: It consumes a lot of RAM/VRAM. Please certify that your computer supports it.
- `ref_channels` - List of channels which the other channels will use for referencing. If null, all the channels will be used for reference (Custom Average Reference).
- `overwrite` - Set `True` to overwrite already processed recordings. If `False`, it will skip already processed recordings (it also skip PDS generation)

The function generates a `params.json` file with specified parameters.

## How does it works?

The preprocessing script has an internal pipeline. It will be explained here.

### 1. Channels treatment

The custom function to treat channel names is applied for the entire dataset.

### 2. Set montage

Using the treated channel names, the EEGDash preprocessor define a montage: assign for every channel its 3D position (vector of 3 elements)

### 3. NaN in 3D positions verification

If an EEG channel position contain NaN numbers, it will raise a warning and stop script execution. It means that a EEG channel contains a name that is not defined in the montage. In other words, the channel name treatment failed.

### 4. PSD Generation

Before applying filters, it is generated a PSD and topomap PSD image for every recording. It will be stored in `output_dir/subjects-analysis/before`. The PSD is very utile for bad recordings analysis.

### 5. Detect Bad Channels

Bad channels contains noise, outlier frequency behavior, etc. The bad channels are detected and marked as "bad" by `NoiseChannels` module from `pyprep`.

If RANSAC (Random Sample A) is set to `True`, it will use RANSAC to detect bad channels. RANSAC isolates these corrupt data pathways by determining which channels behave consistently with the rest of the layout. It is good for bad channel detection, but it takes a while. See more about RANSAC [Here](https://www.youtube.com/watch?v=9D5rrtCC_E0).

If RANSAC is set to `False`, nothing will be done.

### 6. Interpolate Bad Channels

After detecting which channels are bad, they are reconstructed using the good channels nearby.

If there are too many bad channels, the reconstruction will be bad, as the final preprocessed recording. The `bad_ch_tolerance` define how many bad channels a recording may have, and drop it (does not continue the preprocessing and window storage).

### 7. Eliminate artifacts

If there are `eog` or `ecg` channel types in available channels, they will be used for artifact remotion. If a person blinks, the muscle pulse contaminates the EEG recording. It is necessary to remove it using dedicated channels for blink detection (the same for ECG cardiac beats).

If ICA (Independent Component Analysis) is set to `True`, it will be used for artifact remotion. It is an unsupervised machine learning and signal processing algorithm that separates a multivariate mixed signal into its original, statistically independent underlying source components. Then, the "bad" components may be removed, and the signal, reconstructed. It takes a while. 

If ICA is set to `False`, it will run SSP (Signal-Space Projection). SSP is a linear algebraic method to remove heavy background noise and biological artifacts like eye blinks or heartbeats. It is widely used as a faster, non-iterative alternative to ICA.

### 8. Channels referencing

The electrodes measurements are a potential differential. To standardize it, it is necessary to set a reference. There are some tpes of referencing:It may be a set of electrodes, a single electrode, or all the electrodes.

- Single electrode: It will be the "zero": its voltage will be subtracted from any other electrodes. 
- Common Reference Electrodes: it will be calculated the average voltage between an electrodes set. It will be subtracted from any other electrodes.
- Common Average Reference (CAR): The average voltage between all the electrodes will be calculated and subtracted from every electrode.

### 9. Band-Pass Filter

The frequencies below `l_freq` and above `h_freq` will be cut by band-pass filter.

### 10. Downsampling

It reduces the number of points in the recording vector. There are, e.g., datasets with 1000Hz recordings, what means 1000 points by second. A downsampling to 200Hz reduces the vector size by 80%. It is much better for RAM consumption and accelerated dataset evaluation. The dataset analysis quality does not change.

# Available Montages

Defining the montage is very important. If the dataset does not contain exaclty the same channel names of selected montage, the preprocessing is not going to work.

The available montages is defined by `mne` (it is possible to see with `mne.channels.get_builtin_montages(descriptions=True)`)

- `standard_1005` - Electrodes are named and positioned according to the international 10-05 system (343+3 locations)
	- Fp1, Fpz, Fp2, AF9, AF7, AF5, AF3, AF1, AFz, AF2, AF4, AF6, AF8, AF10, F9, F7, F5, F3, F1, Fz, F2, F4, F6, F8, F10, FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10, T9, T7, C5, C3, C1, Cz, C2, C4, C6, T8, T10, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8, TP10, P9, P7, P5, P3, P1, Pz, P2, P4, P6, P8, P10, PO9, PO7, PO5, PO3, PO1, POz, PO2, PO4, PO6, PO8, PO10, O1, Oz, O2, I1, Iz, I2, AFp9h, AFp7h, AFp5h, AFp3h, AFp1h, AFp2h, AFp4h, AFp6h, AFp8h, AFp10h, AFF9h, AFF7h, AFF5h, AFF3h, AFF1h, AFF2h, AFF4h, AFF6h, AFF8h, AFF10h, FFT9h, FFT7h, FFC5h, FFC3h, FFC1h, FFC2h, FFC4h, FFC6h, FFT8h, FFT10h, FTT9h, FTT7h, FCC5h, FCC3h, FCC1h, FCC2h, FCC4h, FCC6h, FTT8h, FTT10h, TTP9h, TTP7h, CCP5h, CCP3h, CCP1h, CCP2h, CCP4h, CCP6h, TTP8h, TTP10h, TPP9h, TPP7h, CPP5h, CPP3h, CPP1h, CPP2h, CPP4h, CPP6h, TPP8h, TPP10h, PPO9h, PPO7h, PPO5h, PPO3h, PPO1h, PPO2h, PPO4h, PPO6h, PPO8h, PPO10h, POO9h, POO7h, POO5h, POO3h, POO1h, POO2h, POO4h, POO6h, POO8h, POO10h, OI1h, OI2h, Fp1h, Fp2h, AF9h, AF7h, AF5h, AF3h, AF1h, AF2h, AF4h, AF6h, AF8h, AF10h, F9h, F7h, F5h, F3h, F1h, F2h, F4h, F6h, F8h, F10h, FT9h, FT7h, FC5h, FC3h, FC1h, FC2h, FC4h, FC6h, FT8h, FT10h, T9h, T7h, C5h, C3h, C1h, C2h, C4h, C6h, T8h, T10h, TP9h, TP7h, CP5h, CP3h, CP1h, CP2h, CP4h, CP6h, TP8h, TP10h, P9h, P7h, P5h, P3h, P1h, P2h, P4h, P6h, P8h, P10h, PO9h, PO7h, PO5h, PO3h, PO1h, PO2h, PO4h, PO6h, PO8h, PO10h, O1h, O2h, I1h, I2h, AFp9, AFp7, AFp5, AFp3, AFp1, AFpz, AFp2, AFp4, AFp6, AFp8, AFp10, AFF9, AFF7, AFF5, AFF3, AFF1, AFFz, AFF2, AFF4, AFF6, AFF8, AFF10, FFT9, FFT7, FFC5, FFC3, FFC1, FFCz, FFC2, FFC4, FFC6, FFT8, FFT10, FTT9, FTT7, FCC5, FCC3, FCC1, FCCz, FCC2, FCC4, FCC6, FTT8, FTT10, TTP9, TTP7, CCP5, CCP3, CCP1, CCPz, CCP2, CCP4, CCP6, TTP8, TTP10, TPP9, TPP7, CPP5, CPP3, CPP1, CPPz, CPP2, CPP4, CPP6, TPP8, TPP10, PPO9, PPO7, PPO5, PPO3, PPO1, PPOz, PPO2, PPO4, PPO6, PPO8, PPO10, POO9, POO7, POO5, POO3, POO1, POOz, POO2, POO4, POO6, POO8, POO10, OI1, OIz, OI2, T3, T5, T4, T6, M1, M2, A1, A2, 
- `standard_1020` - Electrodes are named and positioned according to the international 10-20 system (94+3 locations)
	- Fp1, Fpz, Fp2, AF9, AF7, AF5, AF3, AF1, AFz, AF2, AF4, AF6, AF8, AF10, F9, F7, F5, F3, F1, Fz, F2, F4, F6, F8, F10, FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10, T9, T7, C5, C3, C1, Cz, C2, C4, C6, T8, T10, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8, TP10, P9, P7, P5, P3, P1, Pz, P2, P4, P6, P8, P10, PO9, PO7, PO5, PO3, PO1, POz, PO2, PO4, PO6, PO8, PO10, O1, Oz, O2, O9, Iz, O10, T3, T5, T4, T6, M1, M2, A1, A2, 
- `standard_alphabetic` - Electrodes are named with LETTER-NUMBER combinations (A1, B2, F4, …) (65+3 locations)
	- Fp1, Fpz, Fp2, B3, B1, Bz, B2, B4, F7, F5, F3, F1, Fz, F2, F4, F6, F8, D7, D5, D3, D1, Dz, D2, D4, D6, D8, T3, C5, C3, C1, Cz, C2, C4, C6, T4, E7, E5, E3, E1, Ez, E2, E4, E6, E8, T5, P5, P3, P1, Pz, P2, P4, P6, T6, H3, H1, Hz, H2, H4, O1, Oz, O2, M1, M2, A1, A2, 
- `standard_postfixed` - Electrodes are named according to the international 10-20 system using postfixes for intermediate positions (100+3 locations)
	- Fp1, Fpz, Fp2, F7a, F5a, F3a, F1a, Fza, F2a, F4a, F6a, F8a, F7, F5, F3, F1, Fz, F2, F4, F6, F8, F7p, F5p, F3p, F1p, Fzp, F2p, F4p, F6p, F8p, T1, T3a, C5a, C3a, C1a, Cza, C2a, C4a, C6a, T4a, T2, T3, C5, C3, C1, Cz, C2, C4, C6, T4, T3p, C5p, C3p, C1p, Czp, C2p, C4p, C6p, T4p, T5a, P5a, P3a, P1a, Pza, P2a, P4a, P6a, T6a, Cb1a, T5, P5, P3, P1, Pz, P2, P4, P6, T6, Cb2a, Cb1, O1a, P5p, P3p, P1p, Pzp, P2p, P4p, P6p, O2a, Cb2, O1, Oz, O2, Cb1p, Iz, Cb2p, M1, M2, A1, A2, 
- `standard_prefixed` - Electrodes are named according to the international 10-20 system using prefixes for intermediate positions (74+3 locations)
	- Fp1, Fpz, Fp2, aF3, aF1, aFz, aF2, aF4, F7, F5, F3, F1, Fz, F2, F4, F6, F8, iT1, T1, pF5, pF3, pF1, pFz, pF2, pF4, pF6, T2, iT2, iT3, T3, C5, C3, C1, Cz, C2, C4, C6, T4, iT4, T3A, pC5, pC3, pC1, pCz, pC2, pC4, pC6, T4A, iT5, T5, P5, P3, P1, Pz, P2, P4, P6, T6, iT6, pO5, pO3, pO1, pOz, pO2, pO4, pO6, O1, Oz, O2, Iz, M1, M2, A1, A2, 
- `standard_primed` - Electrodes are named according to the international 10-20 system using prime marks (' and '') for intermediate positions (100+3 locations)
	- Fp1, Fpz, Fp2, F7', F5', F3', F1', Fz', F2', F4', F6', F8', F7, F5, F3, F1, Fz, F2, F4, F6, F8, F7'', F5'', F3'', F1'', Fz'', F2'', F4'', F6'', F8'', T1, T3', C5', C3', C1', Cz', C2', C4', C6', T4', T2, T3, C5, C3, C1, Cz, C2, C4, C6, T4, T3'', C5'', C3'', C1'', Cz'', C2'', C4'', C6'', T4'', T5', P5', P3', P1', Pz', P2', P4', P6', T6', Cb1', T5, P5, P3, P1, Pz, P2, P4, P6, T6, Cb2', Cb1, O1', P5'', P3'', P1'', Pz'', P2'', P4'', P6'', O2', Cb2, O1, Oz, O2, Cb1'', Iz, Cb2'', M1, M2, A1, A2, 
- `biosemi16` - BioSemi cap with 16 electrodes (16+3 locations)
	- Fp1, Fp2, F4, Fz, F3, T7, C3, Cz, C4, T8, P4, Pz, P3, O1, Oz, O2, 
- `biosemi32` - BioSemi cap with 32 electrodes (32+3 locations)
	- Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz, 
- `biosemi64` - BioSemi cap with 64 electrodes (64+3 locations)
	- Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1, C1, C3, C5, T7, TP7, CP5, CP3, CP1, P1, P3, P5, P7, P9, PO7, PO3, O1, Iz, Oz, POz, Pz, CPz, Fpz, Fp2, AF8, AF4, AFz, Fz, F2, F4, F6, F8, FT8, FC6, FC4, FC2, FCz, Cz, C2, C4, C6, T8, TP8, CP6, CP4, CP2, P2, P4, P6, P8, P10, PO8, PO4, O2, 
- `biosemi128` - BioSemi cap with 128 electrodes (128+3 locations)
	- A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31, B32, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C30, C31, C32, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, 
- `biosemi160` - BioSemi cap with 160 electrodes (160+3 locations)
	- A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31, B32, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C30, C31, C32, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32
- `biosemi256` - BioSemi cap with 256 electrodes (256+3 locations)
	- A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31, B32, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C30, C31, C32, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31, F32, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23, G24, G25, G26, G27, G28, G29, G30, G31, G32, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20, H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31, H32
- `easycap-M1` - EasyCap with 10-05 electrode names (74 locations)
	- Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, F1, F2, FC1, FC2, C1, C2, CP1, CP2, P1, P2, AF3, AF4, FC3, FC4, CP3, CP4, PO3, PO4, F5, F6, FC5, FC6, C5, C6, CP5, CP6, P5, P6, AF7, AF8, FT7, FT8, TP7, TP8, PO7, PO8, F9, F10, FT9, FT10, TP9, TP10, P9, P10, PO9, PO10, O9, O10, Fpz, AFz, FCz, CPz, POz, Oz, Iz
- `easycap-M10` - EasyCap with numbered electrodes (61 locations)
	- 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
- `easycap-M43` - EasyCap with numbered electrodes (64 locations)
	- 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, Ref
- `EGI_256` - Geodesic Sensor Net (256 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, E65, E66, E67, E68, E69, E70, E71, E72, E73, E74, E75, E76, E77, E78, E79, E80, E81, E82, E83, E84, E85, E86, E87, E88, E89, E90, E91, E92, E93, E94, E95, E96, E97, E98, E99, E100, E101, E102, E103, E104, E105, E106, E107, E108, E109, E110, E111, E112, E113, E114, E115, E116, E117, E118, E119, E120, E121, E122, E123, E124, E125, E126, E127, E128, E129, E130, E131, E132, E133, E134, E135, E136, E137, E138, E139, E140, E141, E142, E143, E144, E145, E146, E147, E148, E149, E150, E151, E152, E153, E154, E155, E156, E157, E158, E159, E160, E161, E162, E163, E164, E165, E166, E167, E168, E169, E170, E171, E172, E173, E174, E175, E176, E177, E178, E179, E180, E181, E182, E183, E184, E185, E186, E187, E188, E189, E190, E191, E192, E193, E194, E195, E196, E197, E198, E199, E200, E201, E202, E203, E204, E205, E206, E207, E208, E209, E210, E211, E212, E213, E214, E215, E216, E217, E218, E219, E220, E221, E222, E223, E224, E225, E226, E227, E228, E229, E230, E231, E232, E233, E234, E235, E236, E237, E238, E239, E240, E241, E242, E243, E244, E245, E246, E247, E248, E249, E250, E251, E252, E253, E254, E255, E256
- `GSN-HydroCel-32` - HydroCel Geodesic Sensor Net and Cz (33+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, Cz
- `GSN-HydroCel-64_1.0` - HydroCel Geodesic Sensor Net (64+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64
- `GSN-HydroCel-65_1.0` - HydroCel Geodesic Sensor Net and Cz (65+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, Cz
- `GSN-HydroCel-128` - HydroCel Geodesic Sensor Net (128+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, E65, E66, E67, E68, E69, E70, E71, E72, E73, E74, E75, E76, E77, E78, E79, E80, E81, E82, E83, E84, E85, E86, E87, E88, E89, E90, E91, E92, E93, E94, E95, E96, E97, E98, E99, E100, E101, E102, E103, E104, E105, E106, E107, E108, E109, E110, E111, E112, E113, E114, E115, E116, E117, E118, E119, E120, E121, E122, E123, E124, E125, E126, E127, E128
- `GSN-HydroCel-129` - HydroCel Geodesic Sensor Net and Cz (129+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, E65, E66, E67, E68, E69, E70, E71, E72, E73, E74, E75, E76, E77, E78, E79, E80, E81, E82, E83, E84, E85, E86, E87, E88, E89, E90, E91, E92, E93, E94, E95, E96, E97, E98, E99, E100, E101, E102, E103, E104, E105, E106, E107, E108, E109, E110, E111, E112, E113, E114, E115, E116, E117, E118, E119, E120, E121, E122, E123, E124, E125, E126, E127, E128, Cz
- `GSN-HydroCel-256` - HydroCel Geodesic Sensor Net (256+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, E65, E66, E67, E68, E69, E70, E71, E72, E73, E74, E75, E76, E77, E78, E79, E80, E81, E82, E83, E84, E85, E86, E87, E88, E89, E90, E91, E92, E93, E94, E95, E96, E97, E98, E99, E100, E101, E102, E103, E104, E105, E106, E107, E108, E109, E110, E111, E112, E113, E114, E115, E116, E117, E118, E119, E120, E121, E122, E123, E124, E125, E126, E127, E128, E129, E130, E131, E132, E133, E134, E135, E136, E137, E138, E139, E140, E141, E142, E143, E144, E145, E146, E147, E148, E149, E150, E151, E152, E153, E154, E155, E156, E157, E158, E159, E160, E161, E162, E163, E164, E165, E166, E167, E168, E169, E170, E171, E172, E173, E174, E175, E176, E177, E178, E179, E180, E181, E182, E183, E184, E185, E186, E187, E188, E189, E190, E191, E192, E193, E194, E195, E196, E197, E198, E199, E200, E201, E202, E203, E204, E205, E206, E207, E208, E209, E210, E211, E212, E213, E214, E215, E216, E217, E218, E219, E220, E221, E222, E223, E224, E225, E226, E227, E228, E229, E230, E231, E232, E233, E234, E235, E236, E237, E238, E239, E240, E241, E242, E243, E244, E245, E246, E247, E248, E249, E250, E251, E252, E253, E254, E255, E256
- `GSN-HydroCel-257` - HydroCel Geodesic Sensor Net and Cz (257+3 locations)
	- E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49, E50, E51, E52, E53, E54, E55, E56, E57, E58, E59, E60, E61, E62, E63, E64, E65, E66, E67, E68, E69, E70, E71, E72, E73, E74, E75, E76, E77, E78, E79, E80, E81, E82, E83, E84, E85, E86, E87, E88, E89, E90, E91, E92, E93, E94, E95, E96, E97, E98, E99, E100, E101, E102, E103, E104, E105, E106, E107, E108, E109, E110, E111, E112, E113, E114, E115, E116, E117, E118, E119, E120, E121, E122, E123, E124, E125, E126, E127, E128, E129, E130, E131, E132, E133, E134, E135, E136, E137, E138, E139, E140, E141, E142, E143, E144, E145, E146, E147, E148, E149, E150, E151, E152, E153, E154, E155, E156, E157, E158, E159, E160, E161, E162, E163, E164, E165, E166, E167, E168, E169, E170, E171, E172, E173, E174, E175, E176, E177, E178, E179, E180, E181, E182, E183, E184, E185, E186, E187, E188, E189, E190, E191, E192, E193, E194, E195, E196, E197, E198, E199, E200, E201, E202, E203, E204, E205, E206, E207, E208, E209, E210, E211, E212, E213, E214, E215, E216, E217, E218, E219, E220, E221, E222, E223, E224, E225, E226, E227, E228, E229, E230, E231, E232, E233, E234, E235, E236, E237, E238, E239, E240, E241, E242, E243, E244, E245, E246, E247, E248, E249, E250, E251, E252, E253, E254, E255, E256, Cz
- `mgh60` - The (older) 60-channel cap used at MGH (60+3 locations)
	- EEG001, EEG002, EEG003, EEG004, EEG005, EEG006, EEG007, EEG008, EEG009, EEG010, EEG011, EEG012, EEG013, EEG014, EEG015, EEG016, EEG017, EEG018, EEG019, EEG020, EEG021, EEG022, EEG023, EEG024, EEG025, EEG026, EEG027, EEG028, EEG029, EEG030, EEG031, EEG032, EEG033, EEG034, EEG035, EEG036, EEG037, EEG038, EEG039, EEG040, EEG041, EEG042, EEG043, EEG044, EEG045, EEG046, EEG047, EEG048, EEG049, EEG050, EEG051, EEG052, EEG053, EEG054, EEG055, EEG056, EEG057, EEG058, EEG059, EEG060
- `mgh70` - The (newer) 70-channel BrainVision cap used at MGH (70+3 locations)
	- EEG001, EEG002, EEG003, EEG004, EEG005, EEG006, EEG007, EEG008, EEG009, EEG010, EEG011, EEG012, EEG013, EEG014, EEG015, EEG016, EEG017, EEG018, EEG019, EEG020, EEG021, EEG022, EEG023, EEG024, EEG025, EEG026, EEG027, EEG028, EEG029, EEG030, EEG031, EEG032, EEG033, EEG034, EEG035, EEG036, EEG037, EEG038, EEG039, EEG040, EEG041, EEG042, EEG043, EEG044, EEG045, EEG046, EEG047, EEG048, EEG049, EEG050, EEG051, EEG052, EEG053, EEG054, EEG055, EEG056, EEG057, EEG058, EEG059, EEG060, EEG065, EEG066, EEG067, EEG068, EEG069, EEG070, EEG071, EEG072, EEG073, EEG074
- `artinis-octamon` - Artinis OctaMon fNIRS (8 sources, 2 detectors)
	- D1, D2, S1, S2, S3, S4, S5, S6, S7, S8
- `artinis-brite23` - Artinis Brite23 fNIRS (11 sources, 7 detectors)
	- D1, D2, D3, D4, D5, D6, D7, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11
- `brainproducts-RNP-BA-128` - Brain Products with 10-10 electrode names (128 channels)
	- Fp1, Fz, F3, F7, F9, FC5, FC1, C3, T7, CP5, CP1, Pz, P3, P7, P9, O1, Oz, O2, P10, P8, P4, CP2, CP6, T8, C4, Cz, FC2, FC6, F10, F8, F4, Fp2, AF7, AF3, AFz, F1, F5, FT7, FC3, C1, C5, TP7, CP3, P1, P5, PO7, PO3, Iz, POz, PO4, PO8, P6, P2, CPz, CP4, TP8, C6, C2, FC4, FT8, F6, F2, AF4, AF8, AFF3h, FFC1h, FFC5h, FT9, FTT7h, FCC3h, CCP1h, CCP5h, TP9, TPP7h, CPP3h, PPO3h, PPO9h, POO1, PO9, I1, I2, PO10, POO2, PPO10h, PPO4h, CPP4h, TPP8h, TP10, CCP6h, CCP2h, FCC4h, FTT8h, FT10, FFC6h, FFC2h, AFF4h, AFp1, AFF1h, AFF5h, FFT7h, FFC3h, FCC1h, FCC5h, TTP7h, CCP3h, CPP1h, CPP5h, TPP9h, PPO5h, PPO1h, POO9h, OI1h, OI2h, POO10h, PPO2h, PPO6h, TPP10h, CPP6h, CPP2h, CCP4h, TTP8h, FCC6h, FCC2h, FFC4h, FFT8h, AFF6h, AFF2h, AFp2, FCz, Fpz

Script used:

```py
montages = mne.channels.get_builtin_montages(descriptions=True)

for mnt, description in montages:
    channels = mne.channels.make_standard_montage(mnt).ch_names
    print(f"- `{mnt}` - {description}\n\t- ", end="")
    for ch in channels:
        print(ch, end=", ")
    print("")
```

# Good material

- https://neuroanalyzer.org/tutorials/proc_ref.html