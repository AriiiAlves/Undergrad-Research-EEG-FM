# Full Preprocessing + UMAP

Two modules, used one after the other:

1. **[`preprocessing.py`](preprocessing.py)**: `DatasetManager` loads an EEG dataset (**EEGDash**, **MOABB**, TUH, NMT, Sleep-EDF or any Braindecode `BaseConcatDataset`), cleans every recording (one recording per worker, optionally in parallel), cuts it into fixed-length windows, and saves the windows to disk.
2. **[`models_bench.py`](models_bench.py)**: `BenchModel` and its subclasses (`BenchBrainomni`, `BenchLabram`, `BenchCBraMod`) adapt those windows to each pre-trained EEG foundation model, encode them into latent vectors, aggregate them by recording or by subject, and plot UMAPs of the latent space (including UMAPs of the windows labelled by event).

[`main.py`](main.py) runs both for the datasets used in this project. [MONTAGES.md](MONTAGES.md) lists every MNE montage and its channel names.

---

## Contents

**Preprocessing (`preprocessing.py`)**

1. [Quick start](#1-quick-start)
2. [Pipeline overview](#2-pipeline-overview)
3. [Parallel processing](#3-parallel-processing)
4. [Output layout](#4-output-layout)
5. [Public API](#5-public-api)
6. [Internal (private) methods](#6-internal-private-methods)
7. [Resuming, skipping and dropped recordings](#7-resuming-skipping-and-dropped-recordings)
8. [Writing a `channels_treatment` function](#8-writing-a-channels_treatment-function)
9. [Known limitations and caveats](#9-known-limitations-and-caveats)

**Model benchmark (`models_bench.py`)**

10. [Quick start](#10-models_bench-quick-start)
11. [How a benchmark runs](#11-how-a-benchmark-runs)
12. [Model input requirements](#12-model-input-requirements)
13. [Supported models](#13-supported-models)
14. [Public API](#14-benchmark-public-api)
15. [Output layout and caching](#15-benchmark-output-layout-and-caching)
16. [Adding a new model](#16-adding-a-new-model)
17. [Known limitations and caveats](#17-benchmark-limitations-and-caveats)

---

# Part 1: `preprocessing.py`

## 1. Quick start

```python
import mne
from preprocessing import DatasetManager

def my_ch_treatment(raw: mne.io.Raw) -> mne.io.Raw:
    # e.g. "EEG_Fp1" -> "Fp1", so names match a standard montage
    raw.rename_channels({ch: ch.split('_')[-1] for ch in raw.ch_names})
    return raw

dm = DatasetManager()
dm.load_eegdash_dataset("ds006576", subject_limit=0)   # 0 = all recordings
# dm.inspect_dataset()                                 # optional: text report

dm.preprocess_and_window(
    output_dir="DS006576",
    channels_treatment=my_ch_treatment,  # None = keep channels as they are
    l_freq=0.1, h_freq=75,       # band-pass (Hz)
    notch_freqs=[50, 60],        # power line frequencies to remove ([] = no notch)
    down_freq=256,               # resample to 256 Hz
    bad_ch_tolerance=0.2,        # drop a recording if >20% of its EEG channels are bad
    ransac=True,                 # RANSAC bad-channel detection (slow, better)
    ica=True,                    # ICA for EOG/ECG removal (else SSP)
    window_size=512,             # samples per window (512 / 256 Hz = 2 s)
    n_jobs=4,                    # recordings processed at the same time
    ref_channels=[],             # [] = Common Average Reference
    overwrite=False,             # skip recordings already processed / dropped
    montage="colin27_1020",      # "" = auto-detect from channel names
    treat_artifacts=True,
    value_multiplier=1.0,        # 1e-6 if the recordings are stored in µV
)
```

See [main.py](main.py) (`preprocess`) for the call used in this project.

---

## 2. Pipeline overview

```
preprocess_and_window()                                   (main process)
 │
 ├─ channels_treatment is None? ──► use __default_ch_treatment (returns raw unchanged)
 ├─ no montage set? ──► __investigate_montage on the first recording (may ask with input())
 ├─ params.json already exists?
 │      settings equal (n_jobs ignored) ──► keep its "dropped" entries
 │      settings differ ──► prints a saved-vs-new table and asks (input()):
 │           y ─► use the SAVED settings (keeps "dropped")
 │           n ─► use the NEW settings, overwrite=True, forget "dropped" (everything is reprocessed)
 ├─ write PreProcessed-Datasets/<output_dir>/params.json
 │
 └─ joblib.Parallel(n_jobs, mmap_mode='c') over every recording (record_idx = position in the dataset):
      _process_single_subject()                           (worker processes)
       │
       ├─ already saved? (PreProcessed-EEG/<record_idx>/description.json) ──► skip unless overwrite
       ├─ "recording_<record_idx>" in params.json["dropped"]?             ──► skip unless overwrite
       │
       ├─ STAGE 1 ──────────────────────────────────────────────────────────────
       │   1. channels_treatment(raw)        user function (rename / drop channels)
       │   2. __investigate_montage          apply the montage chosen above
       │   3. __verify_montage               every EEG channel needs a valid 3D position
       │   4. __apply_multiplier             unit conversion (e.g. µV -> V), skipped if 1.0
       │   5. __psd_gen (before)             PSD + topomap PNGs of the raw signal
       │   6. __auto_detect_bads             pyprep NoisyChannels -> raw.info["bads"]
       │
       ├─ bad channels > bad_ch_tolerance × n_EEG?
       │      yes ─► __record_dropped("Too many bad channels") ─► return False
       │
       ├─ STAGE 2 ──────────────────────────────────────────────────────────────
       │   7. __eliminate_artifacts          ICA or SSP using EOG/ECG channels
       │   8. __drop_aux_channels            keep only EEG channels
       │   9. __interpolate_bads_if_any      spherical-spline interpolation of bads
       │  10. __define_reference             CAR or custom reference channels
       │  11. __notch_filter(notch_freqs)    FIR notch at the power line frequencies
       │  12. filter(l_freq, h_freq)         FIR band-pass (firwin)
       │  13. resample(down_freq)
       │  14. __psd_gen (after)              PSD + topomap PNGs of the clean signal
       │
       ├─ create_fixed_length_windows        non-overlapping windows, last partial dropped
       ├─ windows.save(offset=record_idx)    written straight to disk, in folder <record_idx>/
       │
       ├─ on any exception ─► __record_dropped("Processing error") ─► return False
       └─ finally: free memory (del, record.raw = None, gc.collect())
 │
 └─ print "Finished Processing. Succesfully processed subjects: X/N"   (X and N count recordings)
```

### Why two stages?

Bad-channel detection (step 6) is cheap compared with ICA, filtering and resampling. Checking `bad_ch_tolerance` between the stages means unusable recordings are dropped before the expensive work runs.

### Why everything is per recording, not per subject?

A subject can have several recordings (sessions, runs, tasks). BNCI2014_001, for example, has 12 per subject (2 sessions × 6 runs). Skipping, dropping and the PSD file names all use the recording index (`record_idx`), so one bad run drops only that run, and a subject's later runs are not mistaken for "already saved". The subject id is still printed and stored (in `description.json` and in each `dropped` entry), and `models_bench.py` can aggregate by subject afterwards (`group_by="subject"`).

### Why the unit conversion comes early (step 4)?

MNE expects EEG in **Volts**. Converting before the "before" PSD and the bad-channel detection means every later step (and the models) sees the same unit. Some datasets store µV; for those use `value_multiplier=1e-6`. `models_bench.py` warns when the saved windows don't look like Volts.

### Why artifacts before interpolation?

ICA is fitted on the good channels. Interpolated channels are linear combinations of their neighbours, so they add no independent information and would distort the decomposition. Bad channels are rebuilt after the artifacts have been removed from their neighbours.

### Why notch both 50 and 60 Hz?

The notch removes a band well under 1 Hz wide, so little EEG is lost. Applying the **same** notch to every dataset matters more than matching each dataset's own mains frequency: a notch leaves a dip in the spectrum, and if datasets differ, the models can pick up that difference and the UMAPs cluster by dataset instead of by subject. Harmonics (100, 120 Hz) are already removed by the band-pass when `h_freq < 100`.

---

## 3. Parallel processing

`preprocess_and_window` runs `_process_single_subject` once per recording with `joblib.Parallel(n_jobs=n_jobs, backend='loky')`. Each worker is a separate Python process, so it gets its own **copy** of the `DatasetManager` and of its arguments.

- **`n_jobs=1`** runs everything sequentially in the main process (no workers).
- **Memory:** each worker loads one full recording and runs ICA/RANSAC on it, so RAM grows roughly linearly with `n_jobs`. Start low (e.g. 4) and watch memory before going higher.
- **Montage:** workers can't ask for `input()`, and a montage found in one worker isn't seen by the others. So when no montage is given, it is detected **once in the main process**, using the first recording, before the workers start. If several montages match, you're asked to choose at that point. For unattended runs (`nohup`, cluster jobs), pass `montage=` explicitly.
- **Read-only arrays (`mmap_mode='c'`):** by default joblib sends large NumPy arrays to the workers as **read-only** memory-mapped files. Datasets that arrive with their data already in memory (e.g. MOABB) would then fail at the first in-place MNE operation with `assignment destination is read-only`. Copy-on-write (`mmap_mode='c'`) keeps the arrays shared between workers, and a worker copies an array only when it writes to it.
- **`params.json`:** workers read and write it with a `FileLock` (`params.json.lock`), so two recordings dropped at the same time don't overwrite each other. If two workers need it at once, the second **waits** until the first finishes its read/write. Only that short block is serialized; the rest of the processing stays parallel. `params.json.lock` is created before the workers start and deleted when processing finishes (even after an error).
- **Dataset not copied per task:** `self.dataset` is set to `None` while the workers run (and restored afterwards). Otherwise the whole dataset would be pickled and sent with every recording.
- **Pickling rule:** anything sent to the workers must be picklable. That's why `_process_single_subject` uses one underscore, and `__default_ch_treatment` is a `@staticmethod`. A regular `__name` method can't be pickled, because of Python name mangling. Your `channels_treatment` must be a module-level function (not a lambda defined inside another function).

---

## 4. Output layout

The class creates two directories relative to the **current working directory** when it is instantiated:

| Attribute | Path | Content |
|---|---|---|
| `DATA_DIR` | `./data` | Raw downloads / cache (EEGDash `cache_dir`, MOABB download dir) |
| `DATASETS_DIR` | `./PreProcessed-Datasets` | Everything the preprocessing produces |

After `preprocess_and_window(output_dir="DS006576", ...)`:

```
./PreProcessed-Datasets/DS006576/
├── params.json                              ← parameters + dropped recordings
├── params.json.lock                         ← only while running: lock file for the workers (deleted at the end)
├── ds006576_inspection.txt                  ← only if inspect_dataset() was called
├── PreProcessed-EEG/
│   ├── 0/                                   ← one folder per recording (folder name = record_idx)
│   │   ├── 0-raw.fif                        ← the preprocessed signal (keeps the annotations / events)
│   │   ├── metadata_df.pkl                  ← window start/stop indices
│   │   ├── description.json                 ← subject, session, run... (its presence marks the recording as processed)
│   │   └── *_kwargs.json                    ← Braindecode window/preprocessing kwargs
│   ├── 1/
│   └── ...                                  ← gaps in the numbering = dropped recordings
└── Recordings-PSDs/
    ├── before/
    │   ├── psd_recording_<record_idx>_before.png
    │   └── psd_topomap_recording_<record_idx>_before.png
    └── after/
        ├── psd_recording_<record_idx>_after.png
        └── psd_topomap_recording_<record_idx>_after.png
```

`record_idx` is the recording's position in the loaded dataset (`self.dataset.datasets`), so folder `<record_idx>/`, the PSD files and the `dropped` key `recording_<record_idx>` all refer to the same recording. The PSD titles also show the subject id.

The saved windows can be loaded back with Braindecode (this is what `models_bench.py` does):

```python
from braindecode.datautil import load_concat_dataset
windows = load_concat_dataset("PreProcessed-Datasets/DS006576/PreProcessed-EEG", preload=False)
```

### `params.json` example

```json
{
    "l_freq": 0.1,
    "h_freq": 75,
    "down_freq": 256,
    "bad_ch_tolerance": 0.2,
    "ransac": true,
    "ica": true,
    "window_size": 512,
    "n_jobs": 4,
    "ref_channels": [],
    "montage": "colin27_1020",
    "treat_artifacts": true,
    "value_multiplier": 1.0,
    "notch_freqs": [50, 60],
    "dropped": {
        "recording_35": {
            "subject": 3,
            "reason": "Too many bad channels",
            "bad_channels": ["C5", "C4", "FC3", "C6", "FC4"],
            "n_bad_channels": 5,
            "n_total_channels": 22
        },
        "recording_142": {
            "subject": 142,
            "reason": "Processing error",
            "error": "..."
        }
    }
}
```

`"montage"` is the montage that was actually used, including one that was auto-detected. `n_jobs` is saved but ignored when comparing settings (it changes the speed, not the result).

> Older `params.json` files keyed `dropped` by subject id (e.g. `"105"`). Those keys no longer match any recording, so those recordings are retried once and, if still bad, dropped again under the new `recording_<idx>` key.

---

## 5. Public API

### `DatasetManager()`

Constructor. Creates `./data` and `./PreProcessed-Datasets` and initialises:

| Attribute | Type | Meaning |
|---|---|---|
| `dataset` | `BaseConcatDataset \| None` | Loaded dataset (one entry per recording) |
| `dataset_name` | `str \| None` | Name used to load the dataset |
| `montage` | `str \| None` | Montage in use (`None` until fixed or auto-detected) |
| `DATA_DIR` | `Path` | Download/cache directory |
| `DATASETS_DIR` | `Path` | Root of preprocessed outputs |

---

### Finding datasets

#### `get_moabb_datasets()`
Returns the list of MOABB dataset **objects** (`moabb.datasets.utils.dataset_search()`). Use `type(d).__name__` to get a dataset's name.

#### `get_eegdash_datasets(dataset=None, task=None, limit=50)`
Queries the EEGDash metadata database.

| Param | Type | Description |
|---|---|---|
| `dataset` | `str` | Dataset id, e.g. `"ds006576"` |
| `task` | `str` | Filter by task name |
| `limit` | `int` | Max records returned |

Returns a list of record dicts (one per recording file). An empty list means nothing matched.

```python
dm.get_eegdash_datasets(dataset="ds006576", limit=5)
```

---

### Loading datasets

#### `load_eegdash_dataset(dataset, subject_limit=0)`
Checks that `dataset` exists in EEGDash, then builds an `EEGDashDataset` cached in `DATA_DIR`. Sets `self.dataset` and `self.dataset_name`.

- `subject_limit`: if `> 0`, keeps only the first *N* **recordings** (not subjects: a subject with several runs counts once per run).

#### `load_moabb_dataset(dataset, subject_limit=0)`
Same idea for MOABB. `dataset` is the MOABB class name (e.g. `"BNCI2014_001"`) and is checked against `get_moabb_datasets()`. It sets the MOABB download dir to `DATA_DIR` and builds a Braindecode `MOABBDataset`. MOABB recordings come with their data already loaded in memory (see the `mmap_mode` note in [§3](#3-parallel-processing)).

#### `load_tuh_dataset(path, abnormal=True, subject_limit=0)`
Loads the TUH EEG corpus from a local folder (it needs a manual download). `abnormal=True` builds `TUHAbnormal` (with `target_name="pathological"`, `dataset_name="TUHAbnormal"`); `False` builds the full `TUH` (`dataset_name="TUH"`).

#### `load_nmt_dataset(path, subject_limit=0)`
Loads the NMT scalp EEG dataset from a local folder (manual download). `dataset_name="NMT"`.

#### `load_sleep_physionet_dataset(subject_limit=0, crop_wake_mins=30)`
Loads Sleep-EDF through Braindecode's `SleepPhysionet` (downloaded automatically). `crop_wake_mins` keeps only that many minutes of wake before and after sleep. `dataset_name="SleepPhysionet"`.

#### `load_generic_dataset(dataset, subject_limit=0)`
Uses any Braindecode `BaseConcatDataset` you already built. It doesn't set `dataset_name`.

All loaders apply `subject_limit` the same way: the first *N* recordings are kept.

---

### Inspecting data

#### `inspect_dataset()`
Writes a full text report of the loaded dataset to `PreProcessed-Datasets/<DATASET_NAME>/<dataset_name>_inspection.txt` and returns its path. The report contains:

- dataset class, number of recordings, the `description` DataFrame and its columns;
- for each recording: entry attributes, then either
  - **Raw**: channels, sfreq, samples, duration, measurement date, channel types and counts, montage, annotations, events (first 20), and per-channel min/max/mean/std; or
  - **Epochs**: number of epochs, channels, sfreq, time range, event IDs, events, metadata, per-channel stats.

It raises `ValueError` if no dataset is loaded. It loads every recording, so it **can be slow and memory-hungry on large datasets**.

#### `inspect_recording(recording)`
Prints a short summary of one `mne.io.BaseRaw` (continuous) or `mne.BaseEpochs` (epoched) object: shape, sfreq, duration, channel names, and event IDs for epochs.

```python
dm.inspect_recording(dm.dataset.datasets[0].raw)
```

---

### Preprocessing

#### `preprocess_and_window(output_dir, channels_treatment=None, ...)`
The main entry point. It chooses the montage, compares the settings with an existing `params.json` (asking which ones to use if they differ, see [§7](#7-resuming-skipping-and-dropped-recordings)), writes `params.json`, then runs `_process_single_subject` on every recording in `self.dataset.datasets`, in parallel when `n_jobs > 1`.

| Param | Default | Description |
|---|---|---|
| `output_dir` | — | Name of the output folder inside `./PreProcessed-Datasets/` |
| `channels_treatment` | `None` | `Callable[[mne.io.Raw], mne.io.Raw]`, applied first to each recording. `None` = keep channels unchanged. See [§8](#8-writing-a-channels_treatment-function) |
| `l_freq` | `1` | High-pass cutoff (Hz) |
| `h_freq` | `40` | Low-pass cutoff (Hz) |
| `down_freq` | `200` | Target sampling rate after resampling (Hz) |
| `bad_ch_tolerance` | `0.15` | Max fraction (0–1) of bad EEG channels. A recording is dropped if `n_bad > int(n_eeg × tolerance)` |
| `ransac` | `True` | Include RANSAC in pyprep's bad-channel detection (more accurate, much slower) |
| `ica` | `True` | `True` = ICA (15 components, FastICA) for EOG/ECG removal; `False` = SSP (faster) |
| `window_size` | `512` | Window length in **samples** at `down_freq`. Stride = `window_size` (no overlap) |
| `n_jobs` | `4` | Number of recordings processed at the same time (worker processes). `1` = sequential. See [§3](#3-parallel-processing) |
| `ref_channels` | `[]` | `[]` = Common Average Reference; otherwise these channels are the reference and are **dropped** afterwards |
| `overwrite` | `False` | `False`: skip recordings that are already saved or dropped. `True`: process everything again. Forced to `True` if you choose the new settings when they differ from `params.json` |
| `montage` | `""` | MNE built-in montage name (e.g. `"colin27_1020"`). `""` / `None` = auto-detect from the first recording. See [MONTAGES.md](MONTAGES.md) |
| `treat_artifacts` | `True` | `False` skips step 7 entirely |
| `value_multiplier` | `1.0` | Multiplies all data channels (step 4) to convert them to **Volts**. `1e-6` for recordings stored in µV. `1.0` = no change |
| `notch_freqs` | `[]` | Power line frequencies removed by the notch filter (step 11), e.g. `[50, 60]`. `[]` = no notch. Frequencies at or above Nyquist are ignored |

Returns nothing. Progress and errors are printed to stdout. At the end it prints how many recordings returned `True`; skipped recordings count as successful.

> **Choosing `down_freq` and `window_size` for the models:** `models_bench.py` resamples and regroups the windows for each model ([§12](#12-model-input-requirements)), so one preprocessed dataset can feed all of them. Preprocess at the highest rate any model needs (256 Hz for BrainOmni) and downsample in the benchmark, never the other way around.

> **Montage names:** MNE 1.13 deprecates `standard_1020` / `standard_1005` / `standard_*` in favour of the identical `colin27_1020` / `colin27_1005` / `colin27_*`. The old names still work with a warning, but **MNE 1.14 removes them**. [MONTAGES.md](MONTAGES.md) lists every montage and its channels.

---

## 6. Internal (private) methods

Most of these use Python name mangling (`__name`), so they can't be called from outside the class. The exception is `_process_single_subject` (one underscore), because it must be picklable for the workers (see [§3](#3-parallel-processing)). Steps 1–14 refer to the pipeline in [§2](#2-pipeline-overview).

### Printing helpers (used by `inspect_dataset`)

| Method | Purpose |
|---|---|
| `__print_section(title)` | Prints a `=====` banner with a title |
| `__print_value(name, value)` | Prints `name : value` with aligned columns |
| `__print_dataframe(name, df)` | Prints a DataFrame in full (or "Not available") |
| `__safe_getattr(obj, name, default)` | `getattr` that returns an error string instead of raising |
| `__inspect_raw(raw, idx)` | Detailed report for an `mne.io.BaseRaw` |
| `__inspect_epochs(epochs, idx)` | Detailed report for an `mne.BaseEpochs` |

### Pipeline steps

#### `__default_ch_treatment(raw)` (static)
Returns `raw` unchanged. Used when `channels_treatment=None`. It is a `@staticmethod` so it can be sent to the workers.

#### `__investigate_montage(raw)`: step 2
Assigns 3D electrode positions.

- If `self.montage` is set (passed via `montage=` or found earlier), applies it directly.
- Otherwise it tests **every** MNE built-in montage and keeps those that contain all of the recording's EEG channel names:
  - **one match** → applied and stored in `self.montage`;
  - **several matches** → asks the user to choose with `input()` (interactive);
  - **no match** → prints the missing channels for each montage and stops with an `AssertionError`.

`preprocess_and_window` calls it once in the main process when no montage is set, so inside the workers `self.montage` is always set and the montage is simply applied.

#### `__verify_montage(raw)`: step 3
Checks that each EEG channel's position (`info['chs'][i]['loc'][:3]`) is not NaN and not all zeros. If one is, it stops with an `AssertionError` naming the channel. This usually means `channels_treatment` left a name that the montage doesn't know.

#### `__apply_multiplier(raw, value_multiplier=1.0)`: step 4
If `value_multiplier != 1.0`, multiplies the data with `raw.apply_function(lambda x: x * value_multiplier, channel_wise=False)`. MNE has no dedicated scaling method; `apply_function` with default `picks` scales every data channel (bad channels included) and leaves stim/misc channels untouched.

#### `__psd_gen(raw, sub_id, record_idx, group, output_dir, sub_dir)`: steps 5 and 14
Computes the power spectral density (`raw.compute_psd()`) and saves two figures at 150 dpi in `PreProcessed-Datasets/<output_dir>/Recordings-PSDs/<sub_dir>/`:
- `psd_recording_<record_idx>_<sub_dir>.png`: PSD per channel;
- `psd_topomap_recording_<record_idx>_<sub_dir>.png`: topographic map per frequency band.

Files are named by recording, so several recordings of one subject don't overwrite each other; the title shows both the recording index and the subject id (plus `group`, if the dataset has one). `sub_dir` is `"before"` or `"after"`. Bad channels are kept in the plot so problems stay visible. Use these figures to judge recording quality (and to check that the notch removed the line noise).

#### `__auto_detect_bads(raw, ransac=True)`: step 6
Runs `pyprep.NoisyChannels(raw, random_state=42).find_all_bads(ransac=ransac)` and stores the result in `raw.info['bads']`. pyprep always runs its NaN/flat, deviation, high-frequency noise, correlation and SNR checks. `ransac=True` adds RANSAC, which predicts each channel from a random subset of the others and flags channels that are predicted poorly.

#### `__eliminate_artifacts(raw, run=True, ica=True)`: step 7
Removes ocular (EOG) and cardiac (ECG) artifacts. It needs channels **typed** `eog`/`ecg` in MNE.

- `run=False` (`treat_artifacts=False`) → returns immediately.
- No EOG/ECG channels → skipped (avoids an unnecessary ICA fit).
- `ica=True`:
  1. fits FastICA (15 components) on a **1 Hz high-passed copy** (slow drifts hurt ICA);
  2. finds components correlated with EOG (`find_bads_eog`) and/or ECG (`find_bads_ecg`);
  3. excludes them and applies the ICA to the original `raw`.
- `ica=False` → SSP: computes one EEG projector each for EOG and ECG and applies them.

#### `__drop_aux_channels(raw)`: step 8
Drops every non-EEG channel (EOG, ECG, EMG, stim, misc…). This saves memory and makes the windows contain EEG only.

#### `__interpolate_bads_if_any(raw)`: step 9
If `raw.info['bads']` is not empty, rebuilds those channels from their neighbours (spherical spline). `reset_bads=False` keeps them marked as bad, so they remain visible in the "after" PSD and can be traced later.

#### `__define_reference(raw, ref_channels=[])`: step 10
- `[]` → `set_eeg_reference('average')` (Common Average Reference, CAR).
- otherwise → checks that each reference channel exists and is EEG (`AssertionError` if not), re-references to their mean, then **drops** them.

#### `__notch_filter(raw, notch_freqs=[])`: step 11
Keeps only the frequencies below Nyquist (`sfreq / 2`), then runs `raw.notch_filter(freqs, method="fir", fir_design="firwin")` (the same FIR design as the band-pass, zero-phase). Does nothing if the list ends up empty. It runs at the original sampling rate, before the band-pass and the resampling.

#### `__record_dropped(output_dir, rec_key, info)`
Under a `FileLock` (`params.json.lock`), reads `PreProcessed-Datasets/<output_dir>/params.json`, sets `params["dropped"][rec_key] = info`, and writes the file back. `rec_key` is `"recording_<record_idx>"`, and `info` includes the `subject` plus the reason.

#### `_process_single_subject(record, output_dir, ..., record_idx)`
Runs the whole pipeline for **one recording** (see the diagram in §2); the name is historical. It runs inside a worker when `n_jobs > 1`.

1. Gets the subject id from `record.description['subject']` (used in prints, PSD titles and `dropped` entries) and builds `rec_key = "recording_<record_idx>"`.
2. Skip checks, only when `overwrite=False`: `PreProcessed-EEG/<record_idx>/description.json` exists (already saved), then `rec_key` in `params.json["dropped"]` (read under the lock).
3. Wraps the record in its own `BaseConcatDataset` so preprocessing touches only that recording.
4. Stage 1 → tolerance check → Stage 2.
5. `create_fixed_length_windows(window_size, stride=window_size, on_last_window='drop', preload=True)`.
6. `windows.save(subject_path, offset=record_idx)` writes to folder `<record_idx>/`. Each recording has its own folder, so workers never write to the same one.
7. Returns `True` (processed or skipped) or `False` (dropped / error).
8. `finally`: deletes local objects, closes figures, sets `record.raw = None` and calls `gc.collect()` to keep RAM use flat over long runs.

---

## 7. Resuming, skipping and dropped recordings

Long runs can be interrupted and restarted without redoing work. Everything is tracked **per recording** (`record_idx`):

| Situation | `overwrite=False` | `overwrite=True` |
|---|---|---|
| Recording already saved (`PreProcessed-EEG/<record_idx>/description.json` exists) | skipped | reprocessed and saved again |
| `recording_<record_idx>` in `params.json["dropped"]` | skipped | retried |
| `params.json` exists, same settings (`n_jobs` ignored) | `dropped` kept | same |
| `params.json` exists, different settings | asks: **y** = run with the saved settings (`dropped` kept); **n** = run with the new settings, `overwrite` forced to `True` and `dropped` cleared | same |
| `params.json` exists but is corrupted | run stops with a JSON error (so `dropped` isn't silently lost) | same |

A recording is added to `dropped` when:
- it has **too many bad channels** (stores the list and counts); or
- **any exception** happens while processing it (stores the error message).

To retry only one recording, delete its entry from `params.json["dropped"]` and run again with `overwrite=False`. After fixing a bug that made many recordings fail, remove every entry with that `error` (keep the "Too many bad channels" ones).

> The settings comparison protects against mixing parameters in one output folder: answering **n** reprocesses everything with the new settings. The prompt uses `input()`, so unattended runs with changed settings block there; use a new `output_dir` for them instead.

---

## 8. Writing a `channels_treatment` function

The function receives an `mne.io.Raw` and **must return it**. Use it to:

- rename channels so they match an MNE montage (see [MONTAGES.md](MONTAGES.md));
- set channel types so EOG/ECG channels are recognised for step 7;
- drop channels that can't be placed on the scalp.

Define it at module level (e.g. in `main.py`) so it can be pickled and sent to the workers.

```python
def ds_treatment(raw):
    # "EEG_Fp1" -> "Fp1"
    raw.rename_channels({ch: ch.split('_')[-1] for ch in raw.ch_names})
    # Tell MNE which channels are EOG / ECG (needed for artifact removal)
    raw.set_channel_types({'LOC': 'eog', 'ROC': 'eog', 'EKG': 'ecg'}, on_unit_change='ignore')
    # Remove channels with no standard position
    raw.drop_channels(['Status'], on_missing='ignore')
    return raw
```

If the dataset already uses standard names, pass `channels_treatment=None`.

---

## 9. Known limitations and caveats

| # | Issue | Where |
|---|---|---|
| 1 | Auto-detection with several matching montages calls `input()`, which blocks unattended runs (cluster / `nohup`). Datasets with standard 10-20 names match several montages (`colin27_*`, `fsaverage_*`, `spherical_*`), so they always ask. Pass `montage=` explicitly in those cases. The settings-changed prompt for `params.json` also uses `input()` | [preprocessing.py:556](preprocessing.py#L556), [preprocessing.py:856](preprocessing.py#L856) |
| 2 | The montage is detected from the **first** recording only and reused for all others. Montage auto-detection also loads that first recording in the main process | [preprocessing.py:815](preprocessing.py#L815) |
| 3 | `self.montage` is kept between calls. Reusing one `DatasetManager` for several datasets without passing `montage=` skips auto-detection for the later ones and applies the first dataset's montage. Use a new `DatasetManager` per dataset, or pass `montage=` | [preprocessing.py:811](preprocessing.py#L811) |
| 4 | Recordings are identified by their **position** (`record_idx`) in the loaded dataset. Changing `subject_limit` is safe (it keeps a prefix), but if the dataset's order or content changes (new dataset version, different loader filters), saved folders and `dropped` keys point to different recordings. Use a new `output_dir` (or `overwrite=True`) in that case | [preprocessing.py:639](preprocessing.py#L639) |
| 5 | `subject_limit` counts recordings, not subjects | [preprocessing.py:320](preprocessing.py#L320) |
| 6 | Any exception marks the recording as dropped, including transient ones (I/O, interrupted download). Retry with `overwrite=True` or edit `params.json` | [preprocessing.py:760](preprocessing.py#L760) |
| 7 | The final "Succesfully processed subjects" count includes skipped recordings, and counts recordings, not subjects | [preprocessing.py:917](preprocessing.py#L917) |
| 8 | The `montage` argument of `_process_single_subject` is unused (`self.montage` is used instead) | [preprocessing.py:628](preprocessing.py#L628) |
| 9 | Artifact removal only works if EOG/ECG channels have the right MNE **type**. If they are typed `eeg`, step 7 is skipped and the montage check may fail on them | `__eliminate_artifacts` |
| 10 | `value_multiplier` must be chosen per dataset; nothing checks the unit during preprocessing. `models_bench.py` only warns afterwards (median \|x\| > 1e-3 V) | `__apply_multiplier` |

---

# Part 2: `models_bench.py`

## 10. `models_bench` quick start

```python
from models_bench import BenchBrainomni, BenchLabram, BenchCBraMod

windows_path = "PreProcessed-Datasets/DS006576/PreProcessed-EEG"   # output of preprocess_and_window

bench = BenchBrainomni(windows_path, "DS006576", batch_size=16)  # verifies the windows, then loads the model

bench.use_latent_space("windows")      # one point per window (encodes once, cached)
bench.umap()                           # one PNG per metadata label + event UMAPs (all / by recording / by subject)

for group_by in ["recording", "subject"]:
    for mode in ["avg", "median", "std"]:
        bench.use_latent_space(mode, group_by=group_by)  # reuses the cached window embeddings
        bench.umap()                                     # one PNG per metadata label
```

The constructor checks the windows against the model's requirements **before** loading or downloading the model, so an incompatible dataset fails fast with a report. See [main.py](main.py) (`bench`) for the loop over all datasets and models used in this project.

---

## 11. How a benchmark runs

```
BenchModel(windows_path, ds_name)
 │
 ├─ load_concat_dataset(windows_path, preload=False)   lazy: one entry per recording
 ├─ verify_windows()                                   errors ─► ValueError, before loading the model
 └─ load_model() ─► .to(device) ─► .eval()

use_latent_space(mode, limit=0, z_score=None, group_by="recording")
 │
 ├─ encode_windows(limit)          cached in latent_space_collection/ (runs the model only once)
 │   └─ for each recording:
 │        select_channels(ch_names)                    model specific (default: all channels)
 │        __prepare(window_ds, picks, sfreq)           all windows of the recording:
 │          ├─ WINDOW_SIZE is None:  resample each window to SFREQ
 │          └─ WINDOW_SIZE is set:   join the windows ─► resample once ─► split in WINDOW_SIZE
 │          ├─ × V_TO_MODEL                            Volts ─► model units
 │          └─ z-score per channel (if z_score)
 │        embed(batch) in batches of batch_size        model specific ─► [batch, latent_dim]
 │   returns (window embeddings, recording index of each window)
 │
 └─ mode == "windows" ─► one point per window
    otherwise         ─► avg / median / std over the windows of each group ─► one point per group
                             group_by="recording": windows of one recording (subject/session/run)
                             group_by="subject":   windows of every recording of the subject

umap()
 ├─ labels = metadata (description) of each point's recording; all-N/A columns skipped
 │     group_by="subject": only columns constant inside each subject (session, run... are dropped)
 ├─ UMAP(cosine) of the points ─► one scatter plot per label
 └─ mode == "windows" ─► __events_umap()
       __window_events(): event (annotation) under the center of each window, "misc" if none
       misc windows dropped, then one UMAP labelled by event:
         ├─ all recordings together            umap_plots/<DS>/..._label_event.png
         ├─ one per recording                  umap_plots/<DS>/recordings_events/
         └─ one per subject (all its recordings) umap_plots/<DS>/subjects_events/
```

### How aggregation works (example with `group_by="subject"`)

Aggregation happens **before** UMAP, in `use_latent_space()`. UMAP never sees the stacked windows of a group, only its single aggregated vector. Example with `latent_dim = 1000`, where sub-01 has two recordings:

```
Recording A (sub-01): 300 windows ─► [300, 1000] ┐
                                                 ├─ stack ─► [400, 1000] ─ mean over windows (dim=0) ─► [1000]   ← sub-01's single point
Recording B (sub-01): 100 windows ─► [100, 1000] ┘

sub-01 ─► [1000]
sub-02 ─► [1000]
 ...
sub-50 ─► [1000]
           ↓
UMAP input: [50, 1000] ─► 50 dots in the plot
```

- Only the **windows axis** collapses; the feature dimension (`latent_dim`) stays the same.
- All windows of the group are **pooled first and aggregated once** ([models_bench.py:367](models_bench.py#L367)), so each **window** has equal weight, not each recording. Above, the result is `0.75 · mean_A + 0.25 · mean_B`, not `(mean_A + mean_B) / 2`. Longer recordings pull the subject's vector toward themselves.
- `median` and `std` are taken per feature over the pooled windows. The pooled `std` also includes the shift between the subject's recordings, not only the variability inside each one.
- `group_by="recording"` does the same with one group per recording (one dot per subject/session/run). `mode="windows"` skips aggregation: every window is its own dot (sub-01 would be 400 dots).

---

## 12. Model input requirements

Each model declares what it needs as **class attributes**. `__prepare` adapts the windows to them, and `verify_windows` checks what can't be adapted.

| Attribute | Default | Meaning |
|---|---|---|
| `SFREQ` | `None` | Sampling rate the model expects. Windows at another rate are resampled with `scipy.signal.resample_poly` (polyphase FIR with anti-aliasing). `None` = keep the preprocessed rate |
| `WINDOW_SIZE` | `None` | Samples per model window, after resampling. Windows are **regrouped**: the recording's windows are joined back into one continuous signal (they are contiguous, since preprocessing uses stride = window size), resampled once, and split into `WINDOW_SIZE` chunks. Only the incomplete last chunk of each recording is dropped. `None` = keep the preprocessed windows |
| `MIN_CHANNELS` | `1` | Minimum usable channels after `select_channels` |
| `REQUIRES_POSITIONS` | `False` | Every used channel needs a valid 3D position (montage) |
| `V_TO_MODEL` | `1.0` | Multiplier from Volts to the units the model was pre-trained with (`1e6` = µV, `1e4` = 100 µV) |
| `Z_SCORE` | `False` | Default for the per-channel, per-window z-score. `use_latent_space(z_score=...)` overrides it |

### Regrouping examples (`WINDOW_SIZE`)

For a recording with 3 preprocessed windows of 50 samples (150 samples in total):

| `WINDOW_SIZE` | Model windows | Dropped |
|---|---|---|
| `25` | 6 (each window split in two) | 0 |
| `75` | 2 (0–75, 75–150) | 0 |
| `100` | 1 (two windows joined) | last 50 samples |

So the preprocessing `window_size` doesn't have to match any model. It only has to give each recording at least one full `WINDOW_SIZE` after resampling.

### What `verify_windows` checks

| Check | Result |
|---|---|
| Recording with no windows | error |
| Fewer usable channels than `MIN_CHANNELS` | error |
| Whole recording shorter than one `WINDOW_SIZE` (after resampling) | error |
| `REQUIRES_POSITIONS` and a channel without valid position (NaN / zeros) | error |
| Sampling rate different from `SFREQ` | warning (will be resampled) |
| Window size different from `WINDOW_SIZE` | warning (will be regrouped) |
| Recordings with different sampling rates / window sizes | warning |
| First window has median \|x\| > 1e-3 (doesn't look like Volts) | warning: preprocess with `value_multiplier=1e-6` if the data is in µV |

It prints a report (recordings, sampling rates, window sizes, usable channels, model specs, warnings, errors) and raises `ValueError` if there is any error.

### Units and z-score

Preprocessing saves the windows in **Volts** (the MNE convention; see `value_multiplier`). `__prepare` multiplies by `V_TO_MODEL` and then, if `z_score` is on, normalizes each channel of each window to zero mean and unit variance (`eps = 1e-12`, small enough for data in Volts). With the z-score on, the physical unit cancels out.

---

## 13. Supported models

| Class | `MODEL_NAME` | `SFREQ` | `WINDOW_SIZE` | `V_TO_MODEL` | `Z_SCORE` | Channels | Weights |
|---|---|---|---|---|---|---|---|
| `BenchBrainomni` | `brainomni_base` | 256 | 512 (2 s) | `1e6` (µV) | `True` | all; needs 3D positions | Hugging Face `OpenTSLab/BrainOmni` (`base/`), downloaded to `ckpt_collection/` |
| `BenchLabram` | `labram` | 200 | 3000 (15 s) | `1e6` (µV) | `False` | only names known by LaBraM (others dropped, printed) | `braindecode/labram-pretrained` |
| `BenchCBraMod` | `cbramod` | 200 | `None` | `1e4` (100 µV) | `False` | all | `braindecode/cbramod-pretrained` |

Each class docstring lists the preprocessing used in the model's paper.

### BrainOmni

- **Paper preprocessing:** 0.1–96 Hz band-pass, 50/60 Hz notch, 256 Hz, PSD-based bad channel interpolation, average reference, and **per-channel z-score per sample**. The paper states no physical unit, because the z-score removes it. Reference and filtering come from `preprocessing.py`; the z-score is on by default here (`use_latent_space(z_score=False)` turns it off).
- **Window length:** the tokenizer splits the signal in `window_length = 512` samples (2 s at 256 Hz) with 25% overlap and **zero-pads** anything shorter. `WINDOW_SIZE = 512` avoids the padding. Longer windows without padding: `512 + k × 384` (896, 1280, …).
- **Embedding:** `model.encode(x, pos, sensor_type)` with the 3D channel positions (plus zeros for the orientation) and sensor type `0` (EEG), averaged over tokens.

### LaBraM

- **Paper preprocessing:** 0.1–75 Hz band-pass, 50 Hz notch, 200 Hz, µV.
- **Window length:** patches of 200 samples (1 s). The pre-trained position/temporal embeddings are fixed to 15 patches, so the model only accepts exactly **3000 samples** (15 s): `WINDOW_SIZE = 3000`, and the preprocessed windows are regrouped into 15 s windows. Each recording therefore yields far fewer windows than for BrainOmni (e.g. 900 vs 6948 on BNCI2014_001), and one window can span several trials/events.
- **Channels:** positional embeddings are chosen by channel name, so channels not in `LABRAM_CHANNEL_ORDER` are dropped.
- **Embedding:** `return_features=True` features, averaged over channel/patch tokens.

### CBraMod

- **Paper preprocessing:** 0.3–75 Hz band-pass, 60 Hz notch, 200 Hz, 30 s non-overlapping windows, units of 100 µV.
- **Window length:** patches of 200 samples, so a **multiple of 200**. Its positional encoding is convolutional and works with any number of patches. `WINDOW_SIZE = 6000` would match the 30 s pre-training windows.
- **Embedding:** `return_features=True` features `[batch, n_chans, n_patches, emb_dim]`, averaged over channel/patch tokens.

---

## 14. Benchmark public API

### `BenchModel(windows_path, ds_name, batch_size=16)`

| Param | Description |
|---|---|
| `windows_path` | Folder with the saved windows (`PreProcessed-Datasets/<DS>/PreProcessed-EEG`) |
| `ds_name` | Dataset name used in output file and folder names (spaces removed) |
| `batch_size` | Model windows encoded at once. Lower it if the GPU runs out of memory |

Uses CUDA when available, otherwise CPU.

### `use_latent_space(mode="avg", limit=0, z_score=None, group_by="recording")`

| Param | Description |
|---|---|
| `mode` | `"avg"`, `"median"`, `"std"`: one vector per group (aggregated over its windows). `"windows"`: one vector per model window |
| `limit` | Encode only the first *N* recordings (`0` = all). Cached separately |
| `z_score` | `None` = the model's `Z_SCORE`; `True` / `False` forces it |
| `group_by` | `"recording"`: one point per recording (subject/session/run). `"subject"`: one point per subject, aggregating the windows of all its recordings (uses the `subject` column of the description). Ignored for `"windows"` |

Sets `final_features` (`[N_points, latent_dim]`) and `subject_ids`, and saves aggregated features to disk. Despite its name, `subject_ids` holds a **recording index** per point: the window's recording in `"windows"` mode, or one representative recording of the group otherwise (its metadata labels the point).

### `encode_windows(limit=0)`

Runs the model over every window and returns `(embeddings [N_windows, latent_dim], recording_ids [N_windows])`. Loaded from cache if it exists. Called by `use_latent_space`; call it directly only if you need the raw window embeddings.

### `umap()`

Fits `umap.UMAP(min_dist=0.1, metric="cosine", random_state=42, init="random")` on `final_features`, with `n_neighbors = min(15, N-1)` for `"windows"` (many points) and `min(5, N-1)` for aggregated points. Skipped if there are fewer than 3 points. Then it saves one plot per column of the dataset `description` (columns that are all N/A are skipped; with `group_by="subject"`, columns that vary inside a subject, like `session` or `run`, are skipped too):

- columns containing `age` → continuous colormap with a colorbar;
- `subject` → each point labelled with its subject id;
- anything else → categorical legend.

In `"windows"` mode it also draws the **event UMAPs** (see below). Call `use_latent_space` first.

The UMAP is fitted in a separate (spawned) process: numba, which UMAP uses, can segfault while compiling on some machines. A crash is retried up to 5 times; `random_state=42` makes every attempt give the same embedding.

### Event UMAPs (`"windows"` mode)

Each window gets the description of the annotation under its **center** (e.g. `left_hand`, `feet` in BNCI2014_001), or `"misc"` if it falls outside every annotation. Window times are rebuilt from the first window's start and the model window duration, so they stay correct after resampling/regrouping. Windows labelled `"misc"` are dropped, and a new UMAP is fitted for each of:

| Plot | Where |
|---|---|
| Every recording together | `umap_plots/<DS>/<plot name>_label_event.png` |
| One per recording | `umap_plots/<DS>/recordings_events/<plot name>_recording_<idx>_label_event.png` |
| One per subject (windows of all its recordings) | `umap_plots/<DS>/subjects_events/<plot name>_subject_<id>_label_event.png` |

All of them use the same event → color mapping. Groups with fewer than 3 event windows are skipped. Nothing is drawn if the windows have no annotations (e.g. resting-state datasets).

### `verify_windows()`

Called by the constructor (see [§12](#what-verify_windows-checks)). Can be called again manually.

### `window_info(window_ds)` (static)

Returns the `mne.Info` of one recording's windows, for both raw-based and epochs-based Braindecode window datasets.

---

## 15. Benchmark output layout and caching

Relative to the **current working directory**:

```
./Model-Benchmarks/<MODEL_NAME>/
├── ckpt_collection/                          ← downloaded weights (BrainOmni only; braindecode models use the HF cache)
├── latent_space_collection/
│   ├── <MODEL>_<DS>_latent_space_windows[_with_z_score][_limit_N].pt              ← window embeddings (the cache)
│   ├── <MODEL>_<DS>_latent_space_windows[_with_z_score][_limit_N]_subject_ids.pt  ← recording index of each window
│   └── <MODEL>_<DS>_latent_space_{avg,median,std}_by_{recording,subject}[_with_z_score][_limit_N].pt  ← aggregated features
└── umap_plots/<DS>/
    ├── <MODEL>_<DS>_windows_umap[_with_z_score]_label_<column>.png
    ├── <MODEL>_<DS>_windows_umap[_with_z_score]_label_event.png                   ← events, every recording
    ├── <MODEL>_<DS>_{avg,median,std}_by_{recording,subject}_umap[_with_z_score]_label_<column>.png
    ├── recordings_events/
    │   └── <MODEL>_<DS>_windows_umap[_with_z_score]_recording_<idx>_label_event.png
    └── subjects_events/
        └── <MODEL>_<DS>_windows_umap[_with_z_score]_subject_<id>_label_event.png
```

`encode_windows` runs the model **once** per dataset / z-score / limit combination; every `mode` and `group_by` reuses the cached window embeddings. Aggregation is cheap and recomputed on every call (the aggregated `.pt` files are outputs, not a cache).

Files named `..._latent_space_{avg,median,std}.pt` or `..._{avg,median,std}_umap...png` without `_by_<group>` come from older runs (before `group_by` existed; they were per recording).

> **Stale cache:** the cache file name only records the dataset, z-score and limit, and the cache isn't checked against the number of recordings. After changing anything else (re-running the preprocessing, `SFREQ`, `WINDOW_SIZE`, `V_TO_MODEL`, `embed`, model weights), **delete the model's `latent_space_collection/` files** for that dataset, or the old embeddings are loaded silently.

---

## 16. Adding a new model

Subclass `BenchModel`, implement `load_model` and `embed`, and set the attributes the model needs:

```python
class BenchMyModel(BenchModel):
    """
    Bench MyModel. Paper Preprocessing Setup:

    - Bandpass filter: ...
    - Resample to ... Hz
    - Set unit to ...

    Font: <paper / repository link>
    """
    MODEL_NAME = "mymodel"     # output folder name
    SFREQ = 250                # Hz expected by the model (None = keep)
    WINDOW_SIZE = 500          # samples per model window (None = keep the preprocessed windows)
    V_TO_MODEL = 1e6           # pre-trained in µV
    Z_SCORE = False
    REQUIRES_POSITIONS = False

    def load_model(self):
        # Return the pre-trained torch.nn.Module (download weights into self.CHECKPOINTS_DIR if needed)
        ...

    def select_channels(self, ch_names):
        # Optional: indices of the channels the model accepts (default: all)
        return list(range(len(ch_names)))

    def embed(self, x, window_ds):
        # x: [batch, n_chans, n_times] float32 tensor, already on self.device, in model units
        # self.current_picks / self.current_ch_names: channels in x (after select_channels)
        # window_ds.ch_pos[self.current_picks]: 3D positions, if the model needs them
        # Return one vector per window: [batch, latent_dim]
        ...
```

Everything else (verification, resampling, regrouping, units, z-score, batching, caching, aggregation, UMAP) is inherited.

---

## 17. Benchmark limitations and caveats

| # | Issue | Where |
|---|---|---|
| 1 | The cache isn't invalidated when model settings or preprocessing change, and isn't checked against the number of recordings (see [§15](#15-benchmark-output-layout-and-caching)) | [models_bench.py:277](models_bench.py#L277) |
| 2 | `__prepare` loads all windows of one recording into RAM at once (needed to rebuild the continuous signal). Long recordings with many channels use more memory; GPU memory is still limited by `batch_size` | [models_bench.py:208](models_bench.py#L208) |
| 3 | The unit check looks only at the **first window** of each recording | [models_bench.py:166](models_bench.py#L166) |
| 4 | With `WINDOW_SIZE = None`, each window is resampled on its own, which adds small edge effects at every window boundary. With `WINDOW_SIZE` set, the recording is resampled once | [models_bench.py:212](models_bench.py#L212) |
| 5 | CBraMod has `WINDOW_SIZE = None`: the preprocessed `window_size` should give a multiple of 200 samples at 200 Hz, otherwise the window doesn't split into whole patches. Nothing checks this before encoding | `BenchCBraMod` |
| 6 | Window embeddings are averaged over all tokens (channels and patches); no model-specific pooling (e.g. CLS token) is used | `embed` methods |
| 7 | `BenchBrainomni` treats every channel as EEG (`sensor_type = 0`) and passes zeros as sensor orientation | [models_bench.py:577](models_bench.py#L577) |
| 8 | With `group_by="subject"`, each point is labelled with the metadata of its subject's first recording; columns that vary inside a subject are skipped, so only subject-level labels (subject, sex, group...) are plotted | [models_bench.py:393](models_bench.py#L393) |
| 9 | Event labels use only the annotation under the window **center**. Windows longer than a trial (e.g. LaBraM's 15 s) span several events but get one label; event UMAPs need raw-based windows whose saved raw keeps its annotations | [models_bench.py:462](models_bench.py#L462) |
| 10 | Few aggregated points (e.g. 9 subjects) give a UMAP with `n_neighbors` ≤ 5 that mostly reflects noise; read those plots with care | [models_bench.py:415](models_bench.py#L415) |
| 11 | With `group_by="subject"`, windows of all the subject's recordings are pooled before aggregating, so recordings with more windows weigh more in the subject's point (see [How aggregation works](#how-aggregation-works-example-with-group_bysubject)) | [models_bench.py:367](models_bench.py#L367) |
