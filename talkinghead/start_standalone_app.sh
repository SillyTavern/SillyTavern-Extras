#!/bin/bash
#
# Launch THA3 in standalone app mode.
#
# This standalone app mode does not interact with SillyTavern.
#
# The usual way to run this fork of THA3 is as a SillyTavern-extras plugin.
# The standalone app mode comes from the original THA3 code, and is included
# for testing and debugging.
#
# If you want to manually pose a character (to generate static expression images),
# use `start_manual_poser.sh` instead.
#
# This must run in the "extras" conda venv!
# Do this first:
#   conda activate extras
#
# The `--char=...` flag can be used to specify which image to load under "tha3/images".
#
python -m tha3.app.app --char=example.png $@
