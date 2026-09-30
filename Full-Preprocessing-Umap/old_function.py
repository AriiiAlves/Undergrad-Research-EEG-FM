# Custom function: rename EEG channels
# def rename_channels(raw):
#         rename_dict = {ch: ch.split('_')[1] for ch in raw.ch_names if '_' in ch}
#         raw.rename_channels(rename_dict)
#         return raw

# def preprocess(dataset, output_dir, l_freq = 1, h_freq = 40, down_freq = 200, bad_ch_tolerance = 0.15, ransac = True, ica = True, window_size = 512):
#     """
#     General function for EEG preprocessing. It requires a braindecode EEG dataset,
#     """
#     #Triggers EEG download
#     #for record in dataset.datasets:
#     #    try:
#     #        _ = record.raw
#     #    except Exception as e:
#     #        print(f"Skipping failed subject: {e}")

#     os.makedirs(output_dir, exist_ok=True) # Ensure that the folder already exists
#     os.makedirs(f"{output_dir}/subjects-analysis", exist_ok=True) # Ensure that the folder already exists
#     os.makedirs(f"{output_dir}/subjects-analysis/after", exist_ok=True) # Ensure that the folder already exists

#     valid_datasets = []
#     for record in dataset.datasets:
#         try:
#             # Attempt to check file existence via raw filenames
#             if record.raw.filenames and os.path.exists(record.raw.filenames[0]):
#                 valid_datasets.append(record)
#             else:
#                 print(f"Skipping: Path missing for subject {record.description['subject']}")
#         except (FileNotFoundError, OSError, Exception) as e:
#             print(f"Skipping subject {record.description["subject"]} due to missing data file: {e}")

#     # Re-wrap valid list into the container
#     valid_datasets = BaseConcatDataset(valid_datasets)

#     # Process all the records
#     preprocess(
#         valid_datasets,
#         [
#             # Note: if apply_on_array=False, passes raw object. If not, passes raw underlying NumPy array.
            
#             # 1. Rename channels
#             Preprocessor(rename_channels, apply_on_array=False),
#             # 2. Set montage
#             Preprocessor("set_montage", montage="colin27_1020", on_missing="ignore"),
#             # 3. Detect bad channels
#             Preprocessor(auto_detect_bads, ransac=ransac, apply_on_array=False),
#             # 4. Interpolate bad channels
#             Preprocessor(interpolate_bads_if_any, apply_on_array=False),
#             # 5. Eliminate EOG/ECG artifacts
#             Preprocessor(eliminate_artifacts, ica=ica, apply_on_array=False),
#             # 6. Set A1/A2 reference
#             Preprocessor(define_reference, apply_on_array=False),
#             # 7. Band-pass filter
#             Preprocessor(
#                 "filter",
#                 l_freq=l_freq,
#                 h_freq=h_freq,
#                 method="fir",
#                 fir_design="firwin",
#             ),
#             # 8. Notch filter (eliminates isolated frequencies)
#             # Preprocessor(
#             #     "notch_filter",
#             #     freqs=[50.0],  # Use [60.0] if electrical network of 60Hz
#             #     method="fir",
#             #     fir_design="firwin"
#             # )
#             # 9. Downsampling
#             Preprocessor("resample", sfreq=down_freq),
#         ],
#     )

#     for record in valid_datasets.datasets:
#         print(f"Processing subject {record.description["subject"]}...")
#         # Generate PSD and PSD Topomap for each subject
#         fig1_path = f"./{output_dir}/subjects-analysis/after/psd_{record.description["subject"]}_after.png"
#         fig2_path = f"./{output_dir}/subjects-analysis/after/psd_topomap_{record.description["subject"]}_after.png"

#         psd = record.raw.compute_psd(verbose=False)
#         fig1 = psd.plot(show=False);
#         fig2 = psd.plot_topomap(show=False);

#         fig1.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD")
#         fig2.suptitle(f"Subject {record.description["subject"]} ({record.description["group"]}) PSD Topomap")
#         fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
#         fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")

#     filtered_dataset = []
#     for record in valid_datasets.datasets:
#         # Bad subject dropping
#         ch_bads = len(record.raw.info["bads"])
#         ch_total = len(record.raw.ch_names)
#         if ch_bads <= int(ch_total * bad_ch_tolerance):
#             filtered_dataset.append(record)
#         else:
#             print(f"-> Dropping subject {record.description["subject"]} (Too many bad channels : {ch_bads})")

#     filtered_dataset = BaseConcatDataset(filtered_dataset)

#     # Create windows for this single record
#     windows = create_fixed_length_windows(
#         filtered_dataset,
#         start_offset_samples=0,
#         stop_offset_samples=None,
#         window_size_samples=window_size,
#         window_stride_samples=window_size,
#         on_last_window='drop',
#         preload=True
#     )

#     # Save this individual subject's windows to disk
#     subject_path = f"{output_dir}/pre-processed-eeg"
#     os.makedirs(subject_path, exist_ok=True)
#     windows.save(subject_path, overwrite=True)