# `preprocessing.py` — Documentation

`preprocessing.py` defines one class, `DatasetManager`. It loads an EEG dataset from **EEGDash** or **MOABB**, cleans every recording (one subject per worker, optionally in parallel), cuts it into fixed-length windows, and saves the windows to disk. Each step is a Braindecode `Preprocessor` applied to an MNE `Raw` object.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Pipeline overview](#2-pipeline-overview)
3. [Parallel processing](#3-parallel-processing)
4. [Output layout](#4-output-layout)
5. [Public API](#5-public-api)
6. [Internal (private) methods](#6-internal-private-methods)
7. [Resuming, skipping and dropped subjects](#7-resuming-skipping-and-dropped-subjects)
8. [Writing a `channels_treatment` function](#8-writing-a-channels_treatment-function)
9. [Known limitations and caveats](#9-known-limitations-and-caveats)

---

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
    l_freq=0.5, h_freq=40,       # band-pass (Hz)
    down_freq=200,               # resample to 200 Hz
    bad_ch_tolerance=0.2,        # drop subject if >20% of EEG channels are bad
    ransac=True,                 # RANSAC bad-channel detection (slow, better)
    ica=True,                    # ICA for EOG/ECG removal (else SSP)
    window_size=512,             # samples per window (512 / 200 Hz = 2.56 s)
    n_jobs=4,                    # subjects processed at the same time
    ref_channels=[],             # [] = Common Average Reference
    overwrite=False,             # skip subjects already processed / dropped
    montage="colin27_1020",      # "" = auto-detect from channel names
    treat_artifacts=True,
)
```

See [main.py](main.py) (`test_preprocessing`) for the call used in this project.

---

## 2. Pipeline overview

```
preprocess_and_window()                                   (main process)
 │
 ├─ channels_treatment is None? ──► use __default_ch_treatment (returns raw unchanged)
 ├─ montage == ""? ──► __investigate_montage on the first recording (may ask with input())
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
       │   4. __psd_gen (before)             PSD + topomap PNGs of the raw signal
       │   5. __auto_detect_bads             pyprep NoisyChannels -> raw.info["bads"]
       │
       ├─ bad channels > bad_ch_tolerance × n_EEG?
       │      yes ─► __record_dropped("Too many bad channels") ─► return False
       │
       ├─ STAGE 2 ──────────────────────────────────────────────────────────────
       │   6. __eliminate_artifacts          ICA or SSP using EOG/ECG channels
       │   7. __drop_aux_channels            keep only EEG channels
       │   8. __interpolate_bads_if_any      spherical-spline interpolation of bads
       │   9. __define_reference             CAR or custom reference channels
       │  10. filter(l_freq, h_freq)         FIR band-pass (firwin)
       │  11. resample(down_freq)
       │  12. __psd_gen (after)              PSD + topomap PNGs of the clean signal
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

Bad-channel detection (step 5) is cheap compared with ICA, filtering and resampling. Checking `bad_ch_tolerance` between the stages means unusable subjects are dropped before the expensive work runs.

### Why artifacts before interpolation?

ICA is fitted on the good channels. Interpolated channels are linear combinations of their neighbours, so they add no independent information and would distort the decomposition. Bad channels are rebuilt after the artifacts have been removed from their neighbours.

---

## 3. Parallel processing

`preprocess_and_window` runs `_process_single_subject` once per recording with `joblib.Parallel(n_jobs=n_jobs, backend='loky')`. Each worker is a separate Python process, so it gets its own **copy** of the `DatasetManager` and of its arguments.

- **`n_jobs=1`** runs everything sequentially in the main process (no workers).
- **Memory:** each worker loads one full recording and runs ICA/RANSAC on it, so RAM grows roughly linearly with `n_jobs`. Start low (e.g. 4) and watch memory before going higher.
- **Montage:** workers can't ask for `input()`, and a montage found in one worker isn't seen by the others. So when `montage=""`, the montage is detected **once in the main process**, using the first recording, before the workers start. If several montages match, you're asked to choose at that point. For unattended runs (`nohup`, cluster jobs), pass `montage=` explicitly.
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

The saved windows can be loaded back with Braindecode:

```python
from braindecode.datautil import load_concat_dataset
windows = load_concat_dataset("PreProcessed-Datasets/DS006576/PreProcessed-EEG", preload=False)
```

### `params.json` example

```json
{
    "l_freq": 0.5,
    "h_freq": 40,
    "down_freq": 200,
    "bad_ch_tolerance": 0.2,
    "ransac": true,
    "ica": true,
    "window_size": 512,
    "n_jobs": 4,
    "ref_channels": [],
    "montage": "colin27_1020",
    "treat_artifacts": true,
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
| `montage` | `str` | Montage in use (`""` until fixed or auto-detected) |
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
| `montage` | `""` | MNE built-in montage name (e.g. `"colin27_1020"`). `""` = auto-detect from the first recording |
| `treat_artifacts` | `True` | `False` skips step 6 entirely |

Returns nothing. Progress and errors are printed to stdout. At the end it prints how many recordings returned `True`; skipped recordings count as successful.

> **Montage names:** recent MNE versions deprecate `standard_1020` / `standard_1005` in favour of `colin27_1020` / `colin27_1005` (the old names still work for now, with a warning). Run `mne.channels.get_builtin_montages()` to list the names your MNE version provides. [NOTES.md](NOTES.md) lists the channels of each montage.

---

## 6. Internal (private) methods

Most of these use Python name mangling (`__name`), so they can't be called from outside the class. The exception is `_process_single_subject` (one underscore), because it must be picklable for the workers (see [§3](#3-parallel-processing)). Steps 1–12 refer to the pipeline in [§2](#2-pipeline-overview).

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

`preprocess_and_window` calls it once in the main process when `montage=""`, so inside the workers `self.montage` is always set and the montage is simply applied.

#### `__verify_montage(raw)`: step 3
Checks that each EEG channel's position (`info['chs'][i]['loc'][:3]`) is not NaN and not all zeros. If one is, it stops with an `AssertionError` naming the channel. This usually means `channels_treatment` left a name that the montage doesn't know.

#### `__psd_gen(raw, sub_id, group, output_dir, sub_dir)`: steps 4 and 12
Computes the power spectral density (`raw.compute_psd()`) and saves two figures at 150 dpi in `PreProcessed-Datasets/<output_dir>/Recordings-PSDs/<sub_dir>/`:
- `psd_<sub_id>_<sub_dir>.png`: PSD per channel;
- `psd_topomap_<sub_id>_<sub_dir>.png`: topographic map per frequency band.

`sub_dir` is `"before"` or `"after"`. Bad channels are kept in the plot so problems stay visible. Use these figures to judge recording quality.

#### `__auto_detect_bads(raw, ransac=True)`: step 5
Runs `pyprep.NoisyChannels(raw, random_state=42).find_all_bads(ransac=ransac)` and stores the result in `raw.info['bads']`. pyprep always runs its NaN/flat, deviation, high-frequency noise, correlation and SNR checks. `ransac=True` adds RANSAC, which predicts each channel from a random subset of the others and flags channels that are predicted poorly.

#### `__eliminate_artifacts(raw, run=True, ica=True)`: step 6
Removes ocular (EOG) and cardiac (ECG) artifacts. It needs channels **typed** `eog`/`ecg` in MNE.

- `run=False` (`treat_artifacts=False`) → returns immediately.
- No EOG/ECG channels → skipped (avoids an unnecessary ICA fit).
- `ica=True`:
  1. fits FastICA (15 components) on a **1 Hz high-passed copy** (slow drifts hurt ICA);
  2. finds components correlated with EOG (`find_bads_eog`) and/or ECG (`find_bads_ecg`);
  3. excludes them and applies the ICA to the original `raw`.
- `ica=False` → SSP: computes one EEG projector each for EOG and ECG and applies them.

#### `__drop_aux_channels(raw)`: step 7
Drops every non-EEG channel (EOG, ECG, EMG, stim, misc…). This saves memory and makes the windows contain EEG only.

#### `__interpolate_bads_if_any(raw)`: step 8
If `raw.info['bads']` is not empty, rebuilds those channels from their neighbours (spherical spline). `reset_bads=False` keeps them marked as bad, so they remain visible in the "after" PSD and can be traced later.

#### `__define_reference(raw, ref_channels=[])`: step 9
- `[]` → `set_eeg_reference('average')` (Common Average Reference, CAR).
- otherwise → checks that each reference channel exists and is EEG (`AssertionError` if not), re-references to their mean, then **drops** them.

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

> Changing preprocessing parameters with `overwrite=False` does **not** reprocess subjects that are already saved, so the output folder can end up with a mix of old and new parameters. Use a new `output_dir` or `overwrite=True` after changing parameters.

---

## 8. Writing a `channels_treatment` function

The function receives an `mne.io.Raw` and **must return it**. Use it to:

- rename channels so they match an MNE montage (see the montage list in [NOTES.md](NOTES.md));
- set channel types so EOG/ECG channels are recognised for step 6;
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
| 1 | Auto-detection with several matching montages calls `input()`, which blocks unattended runs (cluster / `nohup`). Pass `montage=` explicitly in those cases | [preprocessing.py:506](preprocessing.py#L506) |
| 2 | The montage is detected from the **first** recording only and reused for all others. Montage auto-detection also loads that first recording in the main process | [preprocessing.py:749](preprocessing.py#L749) |
| 3 | Recordings are identified by **subject** only. If a subject has several recordings (sessions / tasks / runs), the later ones are skipped as "already saved", their PSD images overwrite each other, and dropping one drops the whole subject. Fine for datasets with one recording per subject (e.g. DS006576) | [preprocessing.py:562](preprocessing.py#L562) |
| 4 | `subject_limit` counts recordings, not subjects | [preprocessing.py:319](preprocessing.py#L319) |
| 5 | Any exception marks the subject as dropped, including transient ones (I/O, interrupted download). Retry with `overwrite=True` or edit `params.json` | [preprocessing.py:699](preprocessing.py#L699) |
| 6 | The final "Succesfully processed" count includes skipped subjects | [preprocessing.py:814](preprocessing.py#L814) |
| 7 | The `montage` argument of `_process_single_subject` is unused (`self.montage` is used instead) | [preprocessing.py:562](preprocessing.py#L562) |
| 8 | Artifact removal only works if EOG/ECG channels have the right MNE **type**. If they are typed `eeg`, step 6 is skipped and the montage check may fail on them | `__eliminate_artifacts` |
