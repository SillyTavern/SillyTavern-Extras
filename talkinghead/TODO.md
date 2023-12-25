## Talkinghead TODO

### Live mode

- Improve frame timing
  - Try to keep the output FPS constant
    - Use a queue instead of a polling loop if Python can efficiently I/O wait on them?
    - Render one frame at startup
    - Make the network streamer send the available frame and then request a new frame immediately
    - Then calculate how much time is left until the next send deadline, and sleep for that, then repeat
  - Decouple animation speed from render FPS; need to calibrate against wall time.
    - OTOH, do we need to do this? Only needed for slow renderers, because if render FPS > network FPS,
      the rate limiter already makes the animation run at a constant FPS.
- Make cool-looking optional output filters:
  - Static scanlines
  - Dynamic scanlines (odd/even lines every other frame)
    - Like Phosphor deinterlacer in VLC.
    - To look good, requires a steady output FPS, and either sync to display refresh, or high enough
      display refresh that syncing doesn't matter (could work for 24 FPS stream on a 144Hz panel).
  - Luma noise
  - Needs torch kernels to do these on the GPU?
- Make the various hyperparameters user-configurable (ideally per character, but let's make a global version first):
  - Blink timing: `blink_interval` min/max
  - Blink probability per frame
  - "confusion" emotion initial segment duration (where blinking quickly in succession is allowed)
  - Sway timing: `sway_interval` min/max
  - Sway strength (`max_random`, `max_noise`)
  - Breathing cycle duration
  - Output target FPS
  - Separate animation target FPS?
- PNG sending efficiency? Look into encoding the stream into YUVA420 using `ffmpeg`.
- Investigate if some particular emotions could use a small random per-frame oscillation applied to "iris_small",
  for that anime "intense emotion" effect (since THA3 doesn't have a morph specifically for the specular reflections in the eyes).
- The "eye_unimpressed" morph has just one key in the emotion JSON, although the model has two morphs (left and right) for this.
  - We should fix this, but it will break backward compatibility for old emotion JSON files.
  - OTOH, maybe not much of an issue, because in all versions prior to this one being developed, the emotion JSON system
    was underutilized anyway (only a bunch of pre-made presets, only used by the live plugin).
  - All the more important to fix this now, before the next release, because the improved manual poser makes it easy to
    generate new emotion JSON files, so from the next release on we can assume those to exist in the wild.

### Client-side bugs / missing features:

- Talking animation is broken, seems the client isn't sending us a request to start/stop talking.
- Add `/emote xxx` support for talkinghead, would make testing much easier.
  - Needs a new API endpoint ("emote"?) in `server.py`, and making the client call that when `/emote xxx` is used.
- If `classify` is enabled, emotion state could be updated from the latest AI-generated text
  when switching chat files, to resume in the same state where the chat left off.
  - Either call the "classify" endpoint (which will re-analyze), or if the client stores the emotion,
    then the new "emote" endpoint.
- When a new talkinghead sprite is uploaded:
  - The preview thumbnail doesn't update.
  - Talkinghead must be switched off and back on to actually send the new image to the backend.

### Common

- Add pictures to the README.
  - Screenshot of the manual poser. Anything else that the user needs to know about it?
  - Examples of generated poses, highlighting both success and failure cases. How it looks in the actual GUI.
- Merge appropriate material from old user manual into the new README.
- Update the user manual.
- Far future: lip-sync talking animation to TTS output (need realtime data from client)
