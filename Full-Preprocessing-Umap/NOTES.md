# Coisas a se fazer

# Treating query

Básicos
- Expandir classes de models_bench para mais modelos
Importantes
- Testar benchmark de outros modelos e expandir para mais modelos
- Pesquisar datasets de treino (usados por artigos)

# Notas

- Harmonics don't matter here. 100 Hz and 120 Hz are above your h_freq = 75, so the band-pass already removes them.

# Available Montages

Defining the montage is very important. If the dataset does not contain exaclty the same channel names of selected montage, the preprocessing is not going to work.

The available montages is defined by `mne` (it is possible to see with `mne.channels.get_builtin_montages(descriptions=True)`)

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

# Outros

Startup takes several minutes. The run started at 17:56, but params.json was only written at 18:02.

The .venv is on the network drive, so every import reads thousands of small files over the network. A plain import check I ran earlier didn't finish within 120 s.
main.py also imports models_bench (torch + BrainOmni) at the top, even though only test_preprocessing() runs.
Each worker re-imports preprocessing.py's libraries (mne, braindecode, eegdash, moabb) when it starts.
Choosing the montage also reads the first recording before any worker starts.