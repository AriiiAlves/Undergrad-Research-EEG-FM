import mne
import contextlib

montages = mne.channels.get_builtin_montages(descriptions=True)

all = {}

for mnt, description in montages:
    channels = mne.channels.make_standard_montage(mnt).ch_names
    print(mnt)
    all[mnt] = channels

    with open("./DEMETRIO.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            print(all)