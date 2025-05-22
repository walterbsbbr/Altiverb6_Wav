import os
import numpy as np
import soundfile as sf

def bytes_to_int24(byte_data):
    """Convert a byte array to int24 PCM format."""
    int24_data = np.frombuffer(byte_data, dtype=np.uint8)
    int24_data = int24_data[: (len(int24_data) // 3) * 3]  # Ensure proper multiple of 3 bytes
    int24_data = int24_data.reshape(-1, 3)  # Each sample is 3 bytes
    int24_values = (int24_data[:, 0].astype(np.int32) |
                    (int24_data[:, 1].astype(np.int32) << 8) |
                    (int24_data[:, 2].astype(np.int32) << 16))
    int24_values[int24_values >= (1 << 23)] -= (1 << 24)  # Convert from unsigned to signed
    return int24_values

def process_int24_pcm_file(file_path):
    """Process an int24 PCM file and convert it to WAV."""
    with open(file_path, "rb") as f:
        raw_data = f.read()
    
    # Extract raw audio data, skipping metadata (first 1024 bytes assumed)
    audio_data = raw_data[1024:]
    
    # Ensure data length is a multiple of 3 for int24 conversion
    valid_length = (len(audio_data) // 3) * 3
    trimmed_data = audio_data[:valid_length]
    
    # Convert to int24 PCM
    extracted_int24_data = bytes_to_int24(trimmed_data)
    
    # Normalize to float32
    normalized_audio = extracted_int24_data / (2**23)
    
    # Define output WAV file path (same directory as input file)
    output_wav_path = file_path + ".wav"
    
    # Save as WAV with 48kHz sample rate
    sf.write(output_wav_path, normalized_audio, samplerate=48000, subtype="PCM_24")
    
    return output_wav_path

def convert_all_files(root_folder):
    """Convert all .1, .2, .3, .4, .R, and .L int24 PCM files in root folder and subdirectories to WAV."""
    for root, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith((".1", ".2", ".3", ".4", ".R", ".L")):
                file_path = os.path.join(root, file_name)
                try:
                    process_int24_pcm_file(file_path)
                    print(f"Converted: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
    
    print("Processing complete. All converted files are saved in their respective directories.")

if __name__ == "__main__":
    root_folder = os.getcwd()  # Run from the current working directory
    convert_all_files(root_folder)
