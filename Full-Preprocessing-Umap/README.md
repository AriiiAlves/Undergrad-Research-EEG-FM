# Full Preprocessing + UMAP

Two modules, used one after the other:

1. **[`preprocessing.py`](preprocessing.py)**: `DatasetManager` loads an EEG dataset from **EEGDash** or **MOABB**, cleans every recording (one subject per worker, optionally in parallel), cuts it into fixed-length windows, and saves the windows to disk.
2. **[`models_bench.py`](models_bench.py)**: `BenchModel` and its subclasses (`BenchBrainomni`, `BenchLabram`, `BenchCBraMod`) adapt those windows to each pre-trained EEG foundation model, encode them into latent vectors, and plot UMAPs of the latent space.

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
7. [Resuming, skipping and dropped subjects](#7-resuming-skipping-and-dropped-subjects)
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
    bad_ch_tolerance=0.2,        # drop subject if >20% of EEG channels are bad
    ransac=True,                 # RANSAC bad-channel detection (slow, better)
    ica=True,                    # ICA for EOG/ECG removal (else SSP)
    window_size=512,             # samples per window (512 / 256 Hz = 2 s)
    n_jobs=4,                    # subjects processed at the same time
    ref_channels=[],             # [] = Common Average Reference
    overwrite=False,             # skip subjects already processed / dropped
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
 ├─ write PreProcessed-Datasets/<output_dir>/params.json  (keeps previous "dropped" entries)
 │
 └─ joblib.Parallel(n_jobs) over every recording:         (worker processes)
      _process_single_subject()
       │
       ├─ already saved? (description.json with same subject) ──► skip unless overwrite
       ├─ listed in params.json["dropped"]?                  ──► skip unless overwrite
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
       ├─ windows.save(...)                  written straight to disk
       │
       ├─ on any exception ─► __record_dropped("Processing error") ─► return False
       └─ finally: free memory (del, record.raw = None, gc.collect())
 │
 └─ print "Finished Processing. Succesfully processed subjects: X/N"
```

### Why two stages?

Bad-channel detection (step 6) is cheap compared with ICA, filtering and resampling. Checking `bad_ch_tolerance` between the stages means unusable subjects are dropped before the expensive work runs.

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
- **`params.json`:** workers read and write it with a `FileLock` (`params.json.lock`), so two subjects dropped at the same time don't overwrite each other. If two workers need it at once, the second **waits** until the first finishes its read/write. Only that short block is serialized; the rest of the processing stays parallel. `params.json.lock` is created before the workers start and deleted when processing finishes (even after an error).
- **Dataset not copied per task:** `self.dataset` is set to `None` while the workers run (and restored afterwards). Otherwise the whole dataset would be pickled and sent with every subject.
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
├── params.json                              ← parameters + dropped subjects
├── params.json.lock                         ← only while running: lock file for the workers (deleted at the end)
├── ds006576_inspection.txt                  ← only if inspect_dataset() was called
├── PreProcessed-EEG/
│   ├── 0/                                   ← one folder per recording (folder name = record index)
│   │   ├── 0-raw.fif                        ← the preprocessed signal
│   │   ├── metadata_df.pkl                  ← window start/stop indices
│   │   ├── description.json                 ← includes "subject"; used to detect already-processed subjects
│   │   └── *_kwargs.json                    ← Braindecode window/preprocessing kwargs
│   ├── 1/
│   └── ...
└── Recordings-PSDs/
    ├── before/
    │   ├── psd_<sub>_before.png
    │   └── psd_topomap_<sub>_before.png
    └── after/
        ├── psd_<sub>_after.png
        └── psd_topomap_<sub>_after.png
```

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
        "105": {
            "reason": "Too many bad channels",
            "bad_channels": ["PO4", "TP8", "..."],
            "n_bad_channels": 13,
            "n_total_channels": 62
        },
        "142": {
            "reason": "Processing error",
            "error": "..."
        }
    }
}
```

`"montage"` is the montage that was actually used, including one that was auto-detected.

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
Same idea for MOABB. `dataset` is the MOABB class name (e.g. `"BNCI2014_001"`) and is checked against `get_moabb_datasets()`. It sets the MOABB download dir to `DATA_DIR` and builds a Braindecode `MOABBDataset`.

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
The main entry point. It chooses the montage, writes `params.json`, then runs `_process_single_subject` on every recording in `self.dataset.datasets`, in parallel when `n_jobs > 1`.

| Param | Default | Description |
|---|---|---|
| `output_dir` | — | Name of the output folder inside `./PreProcessed-Datasets/` |
| `channels_treatment` | `None` | `Callable[[mne.io.Raw], mne.io.Raw]`, applied first to each recording. `None` = keep channels unchanged. See [§8](#8-writing-a-channels_treatment-function) |
| `l_freq` | `1` | High-pass cutoff (Hz) |
| `h_freq` | `40` | Low-pass cutoff (Hz) |
| `down_freq` | `200` | Target sampling rate after resampling (Hz) |
| `bad_ch_tolerance` | `0.15` | Max fraction (0–1) of bad EEG channels. A subject is dropped if `n_bad > int(n_eeg × tolerance)` |
| `ransac` | `True` | Include RANSAC in pyprep's bad-channel detection (more accurate, much slower) |
| `ica` | `True` | `True` = ICA (15 components, FastICA) for EOG/ECG removal; `False` = SSP (faster) |
| `window_size` | `512` | Window length in **samples** at `down_freq`. Stride = `window_size` (no overlap) |
| `n_jobs` | `4` | Number of subjects processed at the same time (worker processes). `1` = sequential. See [§3](#3-parallel-processing) |
| `ref_channels` | `[]` | `[]` = Common Average Reference; otherwise these channels are the reference and are **dropped** afterwards |
| `overwrite` | `False` | `False`: skip subjects that are already saved or dropped. `True`: process everything again |
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

#### `__psd_gen(raw, sub_id, group, output_dir, sub_dir)`: steps 5 and 14
Computes the power spectral density (`raw.compute_psd()`) and saves two figures at 150 dpi in `PreProcessed-Datasets/<output_dir>/Recordings-PSDs/<sub_dir>/`:
- `psd_<sub_id>_<sub_dir>.png`: PSD per channel;
- `psd_topomap_<sub_id>_<sub_dir>.png`: topographic map per frequency band.

`sub_dir` is `"before"` or `"after"`. Bad channels are kept in the plot so problems stay visible. Use these figures to judge recording quality (and to check that the notch removed the line noise).

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

#### `__record_dropped(output_dir, sub_id, info)`
Under a `FileLock` (`params.json.lock`), reads `PreProcessed-Datasets/<output_dir>/params.json`, sets `params["dropped"][sub_id] = info`, and writes the file back.

#### `_process_single_subject(record, output_dir, ..., record_idx)`
Runs the whole pipeline for one recording (see the diagram in §2). It runs inside a worker when `n_jobs > 1`.

1. Gets the subject id from `record.description['subject']`.
2. Skip checks: an existing output folder whose `description.json` has the same subject, then `params.json["dropped"]` (read under the lock). Both apply only when `overwrite=False`.
3. Wraps the record in its own `BaseConcatDataset` so preprocessing touches only that recording.
4. Stage 1 → tolerance check → Stage 2.
5. `create_fixed_length_windows(window_size, stride=window_size, on_last_window='drop', preload=True)`.
6. `windows.save(subject_path, offset=record_idx)` writes to folder `<record_idx>/`. Each recording has its own folder, so workers never write to the same one.
7. Returns `True` (processed or skipped) or `False` (dropped / error).
8. `finally`: deletes local objects, closes figures, sets `record.raw = None` and calls `gc.collect()` to keep RAM use flat over long runs.

---

## 7. Resuming, skipping and dropped subjects

Long runs can be interrupted and restarted without redoing work:

| Situation | `overwrite=False` | `overwrite=True` |
|---|---|---|
| Subject already saved | skipped | reprocessed and saved again |
| Subject in `params.json["dropped"]` | skipped | retried |
| `params.json` exists | parameters replaced, `dropped` kept | same |
| `params.json` exists but is corrupted | run stops with a JSON error (so `dropped` isn't silently lost) | same |

A subject is added to `dropped` when:
- it has **too many bad channels** (stores the list and counts); or
- **any exception** happens while processing it (stores the error message).

To retry only one subject, delete its entry from `params.json["dropped"]` and run again with `overwrite=False`.

> Changing preprocessing parameters with `overwrite=False` does **not** reprocess subjects that are already saved, so the output folder can end up with a mix of old and new parameters. Use a new `output_dir` or `overwrite=True` after changing parameters (e.g. adding `notch_freqs` or changing `value_multiplier`).

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
| 1 | Auto-detection with several matching montages calls `input()`, which blocks unattended runs (cluster / `nohup`). Datasets with standard 10-20 names match several montages (`colin27_*`, `fsaverage_*`, `spherical_*`), so they always ask. Pass `montage=` explicitly in those cases | [preprocessing.py:504](preprocessing.py#L504) |
| 2 | The montage is detected from the **first** recording only and reused for all others. Montage auto-detection also loads that first recording in the main process | [preprocessing.py:769](preprocessing.py#L769) |
| 3 | `self.montage` is kept between calls. Reusing one `DatasetManager` for several datasets without passing `montage=` skips auto-detection for the later ones and applies the first dataset's montage. Use a new `DatasetManager` per dataset, or pass `montage=` | [preprocessing.py:766](preprocessing.py#L766) |
| 4 | Recordings are identified by **subject** only. If a subject has several recordings (sessions / tasks / runs), the later ones are skipped as "already saved", their PSD images overwrite each other, and dropping one drops the whole subject. Fine for datasets with one recording per subject (e.g. DS006576) | [preprocessing.py:579](preprocessing.py#L579) |
| 5 | `subject_limit` counts recordings, not subjects | [preprocessing.py:326](preprocessing.py#L326) |
| 6 | Any exception marks the subject as dropped, including transient ones (I/O, interrupted download). Retry with `overwrite=True` or edit `params.json` | [preprocessing.py:716](preprocessing.py#L716) |
| 7 | The final "Succesfully processed" count includes skipped subjects | [preprocessing.py:856](preprocessing.py#L856) |
| 8 | The `montage` argument of `_process_single_subject` is unused (`self.montage` is used instead) | [preprocessing.py:575](preprocessing.py#L575) |
| 9 | Artifact removal only works if EOG/ECG channels have the right MNE **type**. If they are typed `eeg`, step 7 is skipped and the montage check may fail on them | `__eliminate_artifacts` |
| 10 | `value_multiplier` must be chosen per dataset; nothing checks the unit during preprocessing. `models_bench.py` only warns afterwards (median \|x\| > 1e-3 V) | `__apply_multiplier` |

---

# Part 2: `models_bench.py`

## 10. `models_bench` quick start

```python
from models_bench import BenchBrainomni, BenchLabram, BenchCBraMod

windows_path = "PreProcessed-Datasets/DS006576/PreProcessed-EEG"   # output of preprocess_and_window

bench = BenchBrainomni(windows_path, "DS006576", batch_size=16)  # verifies the windows, then loads the model
for mode in ["avg", "median", "std", "windows"]:
    bench.use_latent_space(mode)       # encodes (once, cached) and aggregates
    bench.umap()                       # one PNG per metadata label
```

The constructor checks the windows against the model's requirements **before** loading or downloading the model, so an incompatible dataset fails fast with a report. See [main.py](main.py) (`bench`) for the loop over all datasets and models used in this project.

---

## 11. How a benchmark runs

```
BenchModel(windows_path, ds_name)
 │
 ├─ load_concat_dataset(windows_path, preload=False)   lazy: one entry per subject/recording
 ├─ verify_windows()                                   errors ─► ValueError, before loading the model
 └─ load_model() ─► .to(device) ─► .eval()

use_latent_space(mode, limit=0, z_score=None)
 │
 ├─ encode_windows(limit)          cached in latent_space_collection/ (runs the model only once)
 │   └─ for each subject:
 │        select_channels(ch_names)                    model specific (default: all channels)
 │        __prepare(window_ds, picks, sfreq)           all windows of the subject:
 │          ├─ WINDOW_SIZE is None:  resample each window to SFREQ
 │          └─ WINDOW_SIZE is set:   join the windows ─► resample once ─► split in WINDOW_SIZE
 │          ├─ × V_TO_MODEL                            Volts ─► model units
 │          └─ z-score per channel (if z_score)
 │        embed(batch) in batches of batch_size        model specific ─► [batch, latent_dim]
 │
 └─ mode == "windows" ─► one point per window
    otherwise         ─► avg / median / std over the windows of each subject ─► one point per subject

umap()
 └─ UMAP(cosine) of the points ─► one scatter plot per metadata column of the dataset description
```

---

## 12. Model input requirements

Each model declares what it needs as **class attributes**. `__prepare` adapts the windows to them, and `verify_windows` checks what can't be adapted.

| Attribute | Default | Meaning |
|---|---|---|
| `SFREQ` | `None` | Sampling rate the model expects. Windows at another rate are resampled with `scipy.signal.resample_poly` (polyphase FIR with anti-aliasing). `None` = keep the preprocessed rate |
| `WINDOW_SIZE` | `None` | Samples per model window, after resampling. Windows are **regrouped**: the subject's windows are joined back into one continuous signal (they are contiguous, since preprocessing uses stride = window size), resampled once, and split into `WINDOW_SIZE` chunks. Only the incomplete last chunk of each subject is dropped. `None` = keep the preprocessed windows |
| `MIN_CHANNELS` | `1` | Minimum usable channels after `select_channels` |
| `REQUIRES_POSITIONS` | `False` | Every used channel needs a valid 3D position (montage) |
| `V_TO_MODEL` | `1.0` | Multiplier from Volts to the units the model was pre-trained with (`1e6` = µV, `1e4` = 100 µV) |
| `Z_SCORE` | `False` | Default for the per-channel, per-window z-score. `use_latent_space(z_score=...)` overrides it |

### Regrouping examples (`WINDOW_SIZE`)

For a subject with 3 preprocessed windows of 50 samples (150 samples in total):

| `WINDOW_SIZE` | Model windows | Dropped |
|---|---|---|
| `25` | 6 (each window split in two) | 0 |
| `75` | 2 (0–75, 75–150) | 0 |
| `100` | 1 (two windows joined) | last 50 samples |

So the preprocessing `window_size` doesn't have to match any model. It only has to give each subject at least one full `WINDOW_SIZE` after resampling.

### What `verify_windows` checks

| Check | Result |
|---|---|
| Subject with no windows | error |
| Fewer usable channels than `MIN_CHANNELS` | error |
| Whole recording shorter than one `WINDOW_SIZE` (after resampling) | error |
| `REQUIRES_POSITIONS` and a channel without valid position (NaN / zeros) | error |
| Sampling rate different from `SFREQ` | warning (will be resampled) |
| Window size different from `WINDOW_SIZE` | warning (will be regrouped) |
| Subjects with different sampling rates / window sizes | warning |
| First window has median \|x\| > 1e-3 (doesn't look like Volts) | warning: preprocess with `value_multiplier=1e-6` if the data is in µV |

It prints a report (subjects, sampling rates, window sizes, usable channels, model specs, warnings, errors) and raises `ValueError` if there is any error.

### Units and z-score

Preprocessing saves the windows in **Volts** (the MNE convention; see `value_multiplier`). `__prepare` multiplies by `V_TO_MODEL` and then, if `z_score` is on, normalizes each channel of each window to zero mean and unit variance (`eps = 1e-12`, small enough for data in Volts). With the z-score on, the physical unit cancels out.

---

## 13. Supported models

| Class | `MODEL_NAME` | `SFREQ` | `WINDOW_SIZE` | `V_TO_MODEL` | `Z_SCORE` | Channels | Weights |
|---|---|---|---|---|---|---|---|
| `BenchBrainomni` | `brainomni_base` | 256 | 512 (2 s) | `1e6` (µV) | `True` | all; needs 3D positions | Hugging Face `OpenTSLab/BrainOmni` (`base/`), downloaded to `ckpt_collection/` |
| `BenchLabram` | `labram` | 200 | `None` | `1e6` (µV) | `False` | only names known by LaBraM (others dropped, printed) | `braindecode/labram-pretrained` |
| `BenchCBraMod` | `cbramod` | 200 | `None` | `1e4` (100 µV) | `False` | all | `braindecode/cbramod-pretrained` |

Each class docstring lists the preprocessing used in the model's paper.

### BrainOmni

- **Paper preprocessing:** 0.1–96 Hz band-pass, 50/60 Hz notch, 256 Hz, PSD-based bad channel interpolation, average reference, and **per-channel z-score per sample**. The paper states no physical unit, because the z-score removes it. Reference and filtering come from `preprocessing.py`; the z-score is on by default here (`use_latent_space(z_score=False)` turns it off).
- **Window length:** the tokenizer splits the signal in `window_length = 512` samples (2 s at 256 Hz) with 25% overlap and **zero-pads** anything shorter. `WINDOW_SIZE = 512` avoids the padding. Longer windows without padding: `512 + k × 384` (896, 1280, …).
- **Embedding:** `model.encode(x, pos, sensor_type)` with the 3D channel positions (plus zeros for the orientation) and sensor type `0` (EEG), averaged over tokens.

### LaBraM

- **Paper preprocessing:** 0.1–75 Hz band-pass, 50 Hz notch, 200 Hz, µV.
- **Window length:** patches of 200 samples (1 s), so the model window must be a **multiple of 200**, up to 3000 samples (15 patches) for the pre-trained weights. With `WINDOW_SIZE = None`, the preprocessed window after resampling must already satisfy this (e.g. 512 samples at 256 Hz → 400 at 200 Hz). Set `WINDOW_SIZE = 200` (or another multiple) to make it independent of the preprocessing.
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

### `use_latent_space(mode="avg", limit=0, z_score=None)`

| Param | Description |
|---|---|
| `mode` | `"avg"`, `"median"`, `"std"`: one vector per subject (aggregated over its windows). `"windows"`: one vector per model window |
| `limit` | Encode only the first *N* subjects/recordings (`0` = all). Cached separately |
| `z_score` | `None` = the model's `Z_SCORE`; `True` / `False` forces it |

Sets `final_features` (`[N_points, latent_dim]`) and `subject_ids`, and saves aggregated features to disk.

### `encode_windows(limit=0)`

Runs the model over every window and returns `(embeddings [N_windows, latent_dim], subject_ids [N_windows])`. Loaded from cache if it exists. Called by `use_latent_space`; call it directly only if you need the raw window embeddings.

### `umap()`

Fits `umap.UMAP(n_neighbors=min(5, N-1), min_dist=0.1, metric="cosine", random_state=42)` on `final_features` and saves one plot per column of the dataset `description` (columns that are all N/A are skipped):

- columns containing `age` → continuous colormap with a colorbar;
- `subject` → each point labelled with its subject id;
- anything else → categorical legend.

Call `use_latent_space` first.

### `verify_windows()`

Called by the constructor (see [§12](#what-verify_windows-checks)). Can be called again manually.

### `window_info(window_ds)` (static)

Returns the `mne.Info` of one subject's windows, for both raw-based and epochs-based Braindecode window datasets.

---

## 15. Benchmark output layout and caching

Relative to the **current working directory**:

```
./Model-Benchmarks/<MODEL_NAME>/
├── ckpt_collection/                          ← downloaded weights (BrainOmni only; braindecode models use the HF cache)
├── latent_space_collection/
│   ├── <MODEL>_<DS>_latent_space_windows[_with_z_score][_limit_N].pt              ← window embeddings (the cache)
│   ├── <MODEL>_<DS>_latent_space_windows[_with_z_score][_limit_N]_subject_ids.pt  ← subject of each window
│   └── <MODEL>_<DS>_latent_space_{avg,median,std}[_with_z_score][_limit_N].pt     ← per-subject features
└── umap_plots/<DS>/
    └── <MODEL>_<DS>_<mode>_umap[_with_z_score]_label_<column>.png
```

`encode_windows` runs the model **once** per dataset / z-score / limit combination; every `mode` reuses the cached window embeddings.

> **Stale cache:** the cache file name only records the dataset, z-score and limit. After changing anything else (preprocessing, `SFREQ`, `WINDOW_SIZE`, `V_TO_MODEL`, `embed`, model weights), **delete the model's `latent_space_collection/` files** for that dataset, or the old embeddings are loaded silently.

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
| 1 | The cache isn't invalidated when model settings or preprocessing change (see [§15](#15-benchmark-output-layout-and-caching)) | [models_bench.py:229](models_bench.py#L229) |
| 2 | `__prepare` loads all windows of one subject into RAM at once (needed to rebuild the continuous signal). Long recordings with many channels use more memory; GPU memory is still limited by `batch_size` | [models_bench.py:194](models_bench.py#L194) |
| 3 | The unit check looks only at the **first window** of each subject | [models_bench.py:157](models_bench.py#L157) |
| 4 | With `WINDOW_SIZE = None`, each window is resampled on its own, which adds small edge effects at every window boundary. With `WINDOW_SIZE` set, the subject is resampled once | [models_bench.py:202](models_bench.py#L202) |
| 5 | LaBraM and CBraMod have `WINDOW_SIZE = None`: the preprocessed `window_size` should give a multiple of 200 samples at 200 Hz (≤ 3000 for LaBraM), otherwise the window doesn't split into whole patches. Nothing checks this before encoding | `BenchLabram`, `BenchCBraMod` |
| 6 | Window embeddings are averaged over all tokens (channels and patches); no model-specific pooling (e.g. CLS token) is used | `embed` methods |
| 7 | `BenchBrainomni` treats every channel as EEG (`sensor_type = 0`) and passes zeros as sensor orientation | [models_bench.py:408](models_bench.py#L408) |
