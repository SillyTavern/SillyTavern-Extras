#!/bin/bash
#
# Launch the THA3 manual poser app.
#
# This app can be used to generate static expression images, given just
# one static input image in the appropriate format.
#
# This app is standalone, and does not interact with SillyTavern.
#
# This must run in the "extras" conda venv!
# Do this first:
#   conda activate extras
#
python -m tha3.app.manual_poser $@
