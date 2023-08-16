"""
RVC module for SillyTavern Extras

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models used by RVC are saved into local data folder: "data/models/"
User RVC model are expected be in "data/models/rvc", one folder per model contraining a pth file and optional index file

References:
    - Code adapted from:
        - RVC-webui: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
        - Audio-webui: https://github.com/gitmylo/audio-webui
"""
from flask import abort, request, send_file, jsonify
import json
import modules.voice_conversion.rvc.rvc as rvc
from scipy.io import wavfile
import os
import io

DEBUG_PREFIX = "<RVC module>"
RVC_MODELS_PATH = "data/models/rvc/"
IGNORED_FILES = [".placeholder"]

def rvc_get_models_list():
    """
    Return the list of RVC model in the expected folder
    """
    try:
        print(DEBUG_PREFIX, "Received request for list of RVC models")

        folder_names = os.listdir(RVC_MODELS_PATH)

        print(DEBUG_PREFIX,"Searching model in",RVC_MODELS_PATH)

        model_list = []
        for folder_name in folder_names:
            folder_path = RVC_MODELS_PATH+folder_name

            if folder_name in IGNORED_FILES:
                continue

            # Must be a folder
            if not os.path.isdir(folder_path):
                print("> WARNING:",folder_name,"is not a folder, ignored")
                continue
            
            print("> Found model folder",folder_name)

            # Check pth
            valid_folder = False
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".pth"):
                    print(" > pth:",file_name)
                    valid_folder = True
                if file_name.endswith(".index"):
                    print(" > index:",file_name)
                
            if valid_folder:
                print(" > Valid folder added to list")
                model_list.append(folder_name)
            else:
                print(" > WARNING: Missing pth, ignored folder")

        # Return the output_audio_path object as a response
        response = json.dumps({"models_list":model_list})
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while searching for RVC models.")

def rvc_process_audio():
    """
    Process request audio file with the loaded RVC model
    Expected request format:
        modelName: string,
        pitchExtraction: string,
        pitchOffset: int,
        indexRate: float [0,1],
        filterRadius: int [0,7],
        rmsMixRate: rmsMixRate,
        protect: float [0,1]
    """
    try:
        file = request.files.get('AudioFile')
        print(DEBUG_PREFIX, "received:", file)
        
        # Create new instances of io.BytesIO() for each request
        input_audio_path = io.BytesIO()
        output_audio_path = io.BytesIO()
        
        file.save(input_audio_path)
        input_audio_path.seek(0)
        
        parameters = json.loads(request.form["json"])
        
        print(DEBUG_PREFIX, "Received audio conversion request with model", parameters)

        folder_path = RVC_MODELS_PATH+parameters["modelName"]+"/"
        model_path = None
        index_path = None
        
        print(DEBUG_PREFIX, "Check for pth file in ", folder_path)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pth"):
                print(" > set pth as ",file_name)
                model_path = folder_path+file_name
                break

        if model_path is None:
            abort(500, DEBUG_PREFIX + " No pth file found.")

        print(DEBUG_PREFIX, "loading", model_path)
        rvc.load_rvc(model_path)
        
        print(DEBUG_PREFIX, "Check for index file", folder_path)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".index"):
                print(" > set index as ",file_name)
                index_path = folder_path+file_name
                break

        if index_path is None:
            index_path = ""
            print(DEBUG_PREFIX, "no index file found, proceeding without index")
        
        info, (tgt_sr, wav_opt) = rvc.vc_single(
            sid=0,
            input_audio_path=input_audio_path,
            f0_up_key=int(parameters["pitchOffset"]),
            f0_file=None,
            f0_method=parameters["pitchExtraction"],
            file_index=index_path,
            file_index2="",
            index_rate=float(parameters["indexRate"]),
            filter_radius=int(parameters["filterRadius"]) // 2 * 2 + 1, # Need to be odd number
            resample_sr=0,
            rms_mix_rate=float(parameters["rmsMixRate"]),
            protect=float(parameters["protect"]),
            crepe_hop_length=128)
        
        #print(info) # DBG

        #out_path = os.path.join("data/", "rvc_output.wav")
        wavfile.write(output_audio_path, tgt_sr, wav_opt)
        output_audio_path.seek(0)  # Reset cursor position
        
        print(DEBUG_PREFIX, "Audio converted using RVC model:", rvc.rvc_model_name)
        
        # Return the output_audio_path object as a response
        response = send_file(output_audio_path, mimetype="audio/x-wav")
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while processing audio.")

def fix_model_install():
    """
    Fix RVC model organisation, move found pth/index file into 
    """
    print(DEBUG_PREFIX,"Checking RVC models folder:",RVC_MODELS_PATH)

    # 1) Search for pth and create corresponding folder
    file_names = os.listdir(RVC_MODELS_PATH)
    print("> Searching for pth files")
    for file_name in file_names:
        file_path = os.path.join(RVC_MODELS_PATH,file_name)

        if file_name in IGNORED_FILES:
            continue
        
        # Must be a folder
        if not os.path.isdir(file_path):
            new_folder_path, file_extension = os.path.splitext(file_path)
            if file_extension != ".pth":
                continue

            print(" > WARNING: pth file found!",file_path)
            print(" > Attempting to create a folder", new_folder_path)

            if os.path.exists(new_folder_path):
                print("  > Folder already exists")
            else:
                os.mkdir(new_folder_path)
                print("  > New model folder created:",new_folder_path)

            new_file_path = os.path.join(new_folder_path,file_name)
            print(" > attempting to move",file_name,"to",new_file_path)

            if os.path.exists(new_file_path):
                print("  > WARNING, file already exists in folder")
                print("   > Model should work.")
                print("   > Clean",RVC_MODELS_PATH,"to stop warnings (all pth/index file must be together in a folder).")
                print("  > File",file_name,"ignored")
                continue
            else:
                os.rename(file_path, new_file_path)
                print("  > File moved, new path:",new_file_path)

    # 2) search for index file and put in corresponding folder   
    file_names = os.listdir(RVC_MODELS_PATH)
    print("> Searching for index files")    
    for file_name in file_names:
        file_path = os.path.join(RVC_MODELS_PATH,file_name)

        if file_name in IGNORED_FILES:
            continue
        
        # Must be a folder
        if not os.path.isdir(file_path):
            new_folder_path, file_extension = os.path.splitext(file_path)
            if file_extension != ".index":
                continue
            
            print(" > WARNING: index file found!",file_path)
            print(" > Searching for possible model folder")

            found = False
            for folder_candidate in file_names:
                folder_candidate_path = os.path.join(RVC_MODELS_PATH,folder_candidate)
                if os.path.isdir(folder_candidate_path):
                    if folder_candidate in file_name:
                        print("  > Found corresponding model folder:",folder_candidate_path)

                        new_file_path = os.path.join(folder_candidate_path,file_name)
                        print("  > attempting to move",file_name,"to",new_file_path)

                        if os.path.exists(new_file_path):
                            print("   > WARNING: file already exists in folder")
                            print("    > Model should work.")
                            print("    > Clean",RVC_MODELS_PATH,"to stop warnings (all pth/index file must be together in a folder).")
                            print("   > File",file_name,"ignored")
                        else:
                            os.rename(file_path, new_file_path)
                            print("   > File moved, new path:",new_file_path)
                        
                        found = True
                        break
            if not found:
                print("  > WARNING: no corresponding folder found, move or delete the file manually to stop warnings.")

    print(DEBUG_PREFIX,"RVC model folder checked.")