#!/bin/bash
echo -e "\033]0;Extras\007"

# ANSI Escape Code for Colors
reset="\033[0m"
white_fg_strong="\033[90m"
red_fg_strong="\033[91m"
green_fg_strong="\033[92m"
yellow_fg_strong="\033[93m"
blue_fg_strong="\033[94m"
magenta_fg_strong="\033[95m"
cyan_fg_strong="\033[96m"

# Normal Background Colors
red_bg="\033[41m"
blue_bg="\033[44m"

# Default arguments file
arguments_file="modules.txt"

# Function to log messages with timestamps and colors
log_message() {
    # This is only time
    current_time=$(date +'%H:%M:%S')
    # This is with date and time 
    # current_time=$(date +'%Y-%m-%d %H:%M:%S')
    case "$1" in
        "INFO")
            echo -e "${blue_bg}[$current_time]${reset} ${blue_fg_strong}[INFO]${reset} $2"
            ;;
        "WARN")
            echo -e "${yellow_bg}[$current_time]${reset} ${yellow_fg_strong}[WARN]${reset} $2"
            ;;
        "ERROR")
            echo -e "${red_bg}[$current_time]${reset} ${red_fg_strong}[ERROR]${reset} $2"
            ;;
        *)
            echo -e "${blue_bg}[$current_time]${reset} ${blue_fg_strong}[DEBUG]${reset} $2"
            ;;
    esac
}

# Log your messages test window
#log_message "INFO" "Something has been launched."
#log_message "WARN" "${yellow_fg_strong}Something is not installed on this system.${reset}"
#log_message "ERROR" "${red_fg_strong}An error occurred during the process.${reset}"
#log_message "DEBUG" "This is a debug message."
#read -p "Press Enter to continue..."

set -e

# Function to find Miniconda or Anaconda installation
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

    log_message "ERROR" "Could not find Miniconda or Anaconda installation" >&2
    return 1
}

# If not set, try to detect Miniconda or Anaconda
if [ -z "$CONDA_PATH" ]; then
    echo "CONDA_PATH not set, trying to detect Miniconda or Anaconda"
    CONDA_PATH=$(find-conda)
fi

if [ -n "$CONDA_PATH" ]; then
    echo "Using Conda at $CONDA_PATH"
    # Activate the Conda environment (extras)
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda config --set auto_activate_base false
    conda init bash
    conda activate extras
fi

# Check if the arguments file exists
if [ ! -f "$arguments_file" ]; then
    log_message "ERROR" "Arguments file '$arguments_file' not found."
    read -p "Press Enter to continue..."
    exit 1
fi

# Read and sanitize the arguments from the file
arguments=$(cat "$arguments_file" | tr -d '\n')

# Start the Python script with the arguments
python server.py $arguments


# XTTS --cuda-device=0
# RVC --enable-modules=rvc --rvc-save-file --max-content-length=1000
# Talkinghead --enable-modules=talkinghead
# Caption --enable-modules=caption
# Summarize --enable-modules=summarize
# To enable multiple flags for example Coqui and RVC do this:
# --enable-modules=coqui-tts,rvc --coqui-gpu --cuda-device=0 --rvc-save-file --max-content-length=1000