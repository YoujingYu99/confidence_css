import soundfile as sf
import os
import librosa

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
audio_path = os.path.join(
    home_dir,
    "extracted_audio",
    "0",
    "I_show_0ICYmXCSBEtC1zPv7jHTyA_16eJN8HS7dHgcfN2YpMGKC_3477.8.mp3",
)

data, rate = sf.read(audio_path, dtype="float32")

