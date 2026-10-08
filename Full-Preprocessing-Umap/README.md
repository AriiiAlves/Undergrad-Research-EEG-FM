# Coisas a se fazer

1. ~~Documentar como utilizar a função de pré-processamento, e as especificações.~~
2. Documentar como utilizar a função de benchmark de modelos, e as especificações.
3. Ao processar dataset, gerar json com informações de processamento, datasets dropados com quantidade de canais ruins, etc
4. Tentar fazer um dicionário completo de nomes padrão (std 10-20) e verificar: se alguma chave não consta nos nomes padrão, soltar um alerta. A pessoa vai ter que definir manualmente. PESQUISAR TODOS OS NOMES PADRÃO DE STANDARD 10-20.
5. Criar UMAPs que o Diego sugeriu

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
- `overwrite` - Set `True` to overwrite already processed recordings. If `False`, it will skip already processed recordings.

## How it works?

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

# Good material

- https://neuroanalyzer.org/tutorials/proc_ref.html