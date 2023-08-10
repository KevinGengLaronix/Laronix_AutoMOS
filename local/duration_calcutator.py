import os
import librosa

folder_path = "/home/kevingeng/Disk2/laronix/laronix_automos/data/Patient_sil_trim_16k_normed_5_snr_40/Sentences"  # Replace with the path to your folder
total_duration = 0.0

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        try:
            # Load the audio file and get its duration
            audio_data, _ = librosa.load(file_path)
            duration = librosa.get_duration(audio_data)
            total_duration += duration
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

print(f"Total duration of audio files in the folder: {total_duration} seconds.")
