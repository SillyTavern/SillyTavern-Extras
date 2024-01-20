# Docker With ROCm Usage

## Building the image

*This is assuming you have ROCm, docker and docker compose installed and running.*

1. Open a terminal and set your current directory to the "docker-rocm" directory in your clone of this repo.
2. Adjust the "docker-compose.yml" file to match your needs. The default selection and the selection with all modules are provided as examples.
3. Un-comment environment in "docker-compose.yml" file if you have Navi 22 GPU - 6700XT / 6750XT / 6800M XT / 6850M XT. It may even work for RDNA 2 APUs.
4. Once ready, run the command "docker compose build" to build the "cohee1207/sillytavern-extras" docker image.
