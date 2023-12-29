## Talkinghead TODO

### Live mode

- Add configuration support to the client
  - Pass the data all the way here (from ST client, to ST server, to ST-extras server, to talkinghead)
  - Configuration:
    - Output target FPS
    - Postprocessor effect chain (including settings)
    - Animation parameters (ideally per character, but let's make a global version first)
      - Blink timing: `blink_interval` min/max
      - Blink probability per frame
      - "confusion" emotion initial segment duration (where blinking quickly in succession is allowed)
      - Sway timing: `sway_interval` min/max
     - Sway strength (`max_random`, `max_noise`)
      - Breathing cycle duration
- Improve frame timing
  - Try to keep the output FPS constant
    - Use a queue or an event instead of a polling loop? Difficult to get this working at plugin startup time.
  - Decouple animation speed from render FPS; need to calibrate against wall time.
    - OTOH, do we need to do this? Only needed for slow renderers, because if render FPS > network FPS,
      the network rate limiter already makes the animation run at a constant FPS (since we produce only
      as many frames as are consumed).
- Add more postprocessing filters. Possible ideas, no guarantee I'll ever get around to them:
  - Pixelize, posterize (8-bit look)
  - Analog video glitches
    - Partition image into bands, move some left/right temporarily
  - Digital data connection glitches
    - Apply to random rectangles; may need to persist for a few frames to animate and/or make them more noticeable
    - May need to protect important regions like the character's head (approximately, from the template); we're after "Hollywood glitchy", not actually glitchy
    - Types:
      - Constant-color rectangle
      - Missing data (zero out the alpha?)
      - Blur (leads to replacing by average color, with controllable sigma)
      - Zigzag deformation
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
- Add `/emote xxx` support for talkinghead to make testing much easier.
  - Needs a new API endpoint ("emote"?) in `server.py`, and making the client call that when `/emote xxx` is used.
- If `classify` is enabled, emotion state should be updated from the latest AI-generated text
  when switching chat files, to resume in the same emotion state where the chat left off.
  - Either call the "classify" endpoint (which will re-analyze), or if the client stores the emotion,
    then the new "emote" endpoint.
- When a new talkinghead sprite is uploaded:
  - The preview thumbnail in the client doesn't update.
  - Talkinghead must be switched off and back on to actually send the new image to the backend.
    We have the `/api/talkinghead/load` endpoint already, so just call it in the client.

### Common

- Add pictures to the README.
  - Screenshot of the manual poser. Anything else the user needs to know about it?
  - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI.
- Merge appropriate material from old user manual into the new README.
- Update the user manual.
- Far future: lip-sync talking animation to TTS output (need realtime data from client)
  - THA3 has morphs for A, I, U, E, O, and the "mouth delta" shape Δ.
