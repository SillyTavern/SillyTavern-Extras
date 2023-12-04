#!/bin/bash

# Environment Variables

function find-conda {
    local paths=(
        "$HOME/miniconda3"
        "$HOME/miniconda"
        "$HOME/opt/miniconda3"
        "$HOME/opt/miniconda"
        "/opt/miniconda3"
        "/opt/miniconda"
        "/usr/local/miniconda3"
        "/usr/local/miniconda"
        "/usr/miniconda3"
        "/usr/miniconda"
        "$HOME/anaconda3"
        "$HOME/anaconda"
        "$HOME/opt/anaconda3"
        "$HOME/opt/anaconda"
        "/opt/anaconda3"
        "/opt/anaconda"
        "/usr/local/anaconda3"
        "/usr/local/anaconda"
        "/usr/anaconda3"
        "/usr/anaconda"
    )

    local conda-path="$(which conda)"
    if [ $? -eq 0 ]; then
        paths+=("$(dirname "$(dirname "$conda-path")")")
    fi


    if [ "$(uname)" == "Darwin" ]; then
        paths+=("/opt/homebrew-cask/Caskroom/miniconda")
        paths+=("/usr/local/Caskroom/miniconda/base")
    fi

    for path in "${paths[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    echo "ERROR: Could not find miniconda installation" >&2
    return 1
}

#if not set, try to detect miniconda
if [ -z "$CONDA_PATH" ]; then
    echo "CONDA_PATH not set, trying to detect miniconda"
    CONDA_PATH=$(find-conda)
fi
# Activate the Conda environment (extras)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
conda init bash
conda activate extras

# Start the Python server for SillyTavern-extras. You can modify the flags below
python server.py --rvc-save-file --cuda-device=0 --max-content-length=1000 --enable-modules=summarize,rvc,

# https://docs.sillytavern.app/extras/installation/#decide-which-module-to-use

# Coqui --enable-modules=coqui-tts --coqui-gpu --cuda-device=0
# RVC --enable-modules=rvc --rvc-save-file --max-content-length=1000
# Talkinghead --enable-modules=talkinghead
# Caption --enable-modules=caption
# Summarize --enable-modules=summarize

# To enable multiple flags for example Coqui and RVC do this:
# --enable-modules=coqui-tts,rvc --coqui-gpu --cuda-device=0 --rvc-save-file --max-content-length=1000
