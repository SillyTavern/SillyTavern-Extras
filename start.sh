#!/bin/bash

# Environment Variables
miniconda_path="$HOME/miniconda"

# Activate the Conda environment (sillytavernextras)
source $miniconda_path/etc/profile.d/conda.sh
conda config --set auto_activate_base false
conda init bash
conda activate sillytavernextras

# Start the Python server for SillyTavern-extras. You can modify the flags below
python server.py --coqui-gpu --rvc-save-file --cuda-device=0 --max-content-length=1000 --enable-modules=talkinghead,chromadb,caption,summarize,rvc,coqui-tts
read -p 'PRESS ENTER TO CLOSE'

# Coqui --enable-modules=coqui-tts --coqui-gpu --cuda-device=0
# RVC --enable-modules=rvc --rvc-save-file --max-content-length=1000
# Talkinghead --enable-modules=talkinghead
# Caption --enable-modules=caption
# Summarize --enable-modules=summarize

# To enable multiple flags for example Coqui and RVC do this: 
# --enable-modules=coqui-tts,rvc --coqui-gpu --cuda-device=0 --rvc-save-file --max-content-length=1000