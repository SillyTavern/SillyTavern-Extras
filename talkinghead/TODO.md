## Talkinghead TODO

### Live mode

- Make animation speed independent of target FPS (choppy animation is better than running slower than realtime in a realtime application)
  - Currently animation works per-frame, so it looks natural only at its design target FPS (25...30)
  - But we should also allow higher-FPS, smoother animation for users who prefer that and have the hardware to support it
  - Scale the interpolation step so that higher FPS -> smaller step (and vice versa)
    - Note the saturating exponential behavior (when target pose held constant); the step size scaling needs to be nonlinear to account for this.
      Work out the math, should be rather simple:
      - The pose interpolator is essentially an ODE solver for Newton's law of cooling, with piecewise constant loading.
      - But instead of numerical integration, we're essentially reading off points from the analytical solution curve, so it's stable regardless of step size.
      - However, the step is the distance from the current state to the final state (along the "temperature" axis in the "cooling" law), not the time.
      - Invert this relationship to find out how to scale step size to make the result behave linearly in time.
      - Then scale the "linear step" by `target_sec / reference_sec`, where `target_sec = 1 / target_fps`, and `reference_sec = 1 / reference_fps = 1 / 25`
        (or `1 / 30`, whichever looks better in practice).
- Add optional per-character configuration
  - At client end, JSON files in `SillyTavern/public/characters/characternamehere/`
  - Pass the data all the way here (from ST client, to ST server, to ST-extras server, to talkinghead module)
  - Configuration:
    - Target FPS (default 25.0)
    - Postprocessor effect chain (including settings)
    - Animation parameters (ideally per character)
      - Blink timing: `blink_interval` min/max (when randomizing the next blink timing)
      - Blink probability per frame
      - "confusion" emotion initial segment duration (where blinking quickly in succession is allowed)
      - Sway timing: `sway_interval` min/max (when randomizing the next sway timing)
     - Sway strength (`max_random`, `max_noise`)
      - Breathing cycle duration
    - Emotion templates
      - One JSON file per emotion, like for the server default templates? This format is easily produced by the manual poser GUI tool.
      - Could be collected by the client into a single JSON for sending.
  - Need also global defaults
    - These could live at the SillyTavern-extras server end
    - Still, don't hardcode, but read from JSON file, to keep easily configurable
- Add live-modifiable configuration for animation and postprocessor settings?
  - Add a new control panel to SillyTavern client extension settings
  - Send new configs to backend whenever anything changes
- Small performance optimization: see if we could use more in-place updates in the postprocessor, to reduce allocation of temporary tensors.
  - The effect on speed will be small; the compute-heaviest part is the inference of the THA3 deep-learning model.
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
- Investigate if some particular emotions could use a small random per-frame oscillation applied to "iris_small",
  for that anime "intense emotion" effect (since THA3 doesn't have a morph specifically for the specular reflections in the eyes).
- The "eye_unimpressed" morph has just one key in the emotion JSON, although the model has two morphs (left and right) for this.
  - We should fix this, but it will break backward compatibility for old emotion JSON files.
  - OTOH, maybe not much of an issue, because in all versions prior to this one being developed, the emotion JSON system
    was underutilized anyway (only a bunch of pre-made presets, only used by the live plugin).
  - All the more important to fix this now, before the next release, because the improved manual poser makes it easy to
    generate new emotion JSON files, so from the next release on we can assume those to exist in the wild.

### Client-side bugs / missing features:

- If `classify` is enabled, emotion state should be updated from the latest AI-generated text
  when switching chat files, to resume in the same emotion state where the chat left off.
  - Either call the "classify" endpoint (which will re-analyze), or if the client stores the emotion,
    then the "set_emotion" endpoint.
- When a new talkinghead sprite is uploaded:
  - The preview thumbnail in the client doesn't update.

### Common

- Add pictures to the README.
  - Screenshot of the manual poser. Anything else the user needs to know about it?
  - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI.
- Document postprocessor filters and their settings in the README, with example pictures.
- Merge appropriate material from old user manual into the new README.
- Update the user manual.
- Far future:
  - Lip-sync talking animation to TTS output (need realtime data from client)
    - THA3 has morphs for A, I, U, E, O, and the "mouth delta" shape Δ.
  - Fast, high-quality scaling mechanism.
    - On a 4k display, the character becomes rather small, which looks jarring on the default backgrounds.
    - The algorithm should be cartoon-aware, some modern-day equivalent of waifu2x. A GAN such as 4x-AnimeSharp or Remacri would be nice, but too slow.
    - Maybe the scaler should run at the client side to avoid the need to stream 1024x1024 PNGs.
      - What JavaScript anime scalers are there, or which algorithms are simple enough for a small custom implementation?
