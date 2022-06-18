"""
Test whether the mp3 files extracted sound correct.
"""

import vlc
import os
import pyglet
import playsound

# home_dir is the location of script
home_dir = os.path.join("/home", "yyu")
audio_path = os.path.join(home_dir, "extracted_audio", "0", 'I_show_0ICYmXCSBEtC1zPv7jHTyA_6jOfa6I0Gq4U4vdcA9xAIS_2470.2.mp3')
# audio = pyglet.media.load(audio_path)
# audio = vlc.MediaPlayer(audio_path)
playsound.playsound(audio_path)