## Talkinghead TODO

### Live mode

- Fix timing of microsway based on 25 FPS reference
- Add optional per-character configuration
  - At client end, JSON files in `SillyTavern/public/characters/characternamehere/`
  - Pass the data all the way here (from ST client, to ST-extras server, to talkinghead module)
    - New API endpoints: `/api/talkinghead/load_emotion_templates`, `/api/talkinghead/load_animator_settings`
  - Configuration (per-character):
    - Target FPS (default 25.0)
    - Postprocessor effect chain (including settings)
    - Animation parameters
      - Blink timing: `blink_interval` min/max (when randomizing the next blink timing)
      - Blink probability per frame
      - "confusion" emotion initial segment duration (where blinking quickly in succession is allowed)
      - Sway timing: `sway_interval` min/max (when randomizing the next sway timing)
     - Sway strength (`max_random`, `max_noise`)
      - Breathing cycle duration
    - Emotion templates
      - One JSON with all emotions, easier for sending from the client.
      - The manual poser currently produces individual emotion `.json` files only.
      - When batch-exporting from the manual poser, also automatically produce a combined `_emotions.json`.
        - This also makes it easier to maintain `talkinghead/emotions/_defaults.json`, because the batch export then generates all necessary files.
        - Optimize the JSON export to drop zeroes, since that is the default - at least in `_emotions.json`.
          The individual emotion files could retain the zeros, to help discoverability.
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

### Client side

- If `classify` is enabled, emotion state should be updated from the latest AI-generated text
  when switching chat files, to resume in the same emotion state where the chat left off.
  - Either call the "classify" endpoint (which will re-analyze), or if the client stores the emotion,
    then the "set_emotion" endpoint.
- When a new talkinghead sprite is uploaded:
  - The preview thumbnail in the client doesn't update.
- Are there other places in *Character Expressions* (`SillyTavern/public/scripts/extensions/expressions/index.js`)
  where we need to check whether the `talkinghead` module is enabled? `(!isTalkingHeadEnabled() || !modules.includes('talkinghead'))`
- Check zip upload whether it refreshes the talkinghead character (it should).

### Common

- Add pictures to the talkinghead README.
  - Screenshot of the manual poser. Anything else the user needs to know about it?
  - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI.
- Document the per-character configuration in the README:
  - Animator settings,
  - Emotion templates,
  - Postprocessor filters (with example pictures).
- Merge appropriate material from old user manual into the new README.
- Update the user manual.
  - This extension really has nothing to do with VTubing, except that this uses a (different!) character animation technology that produces
    output similar to VTubing software such as Live2D.
  - Emphasize that `talkinghead` is an AI-powered character animation technology that animates *the AI character's* avatar
    (cf. VTubing where the idea is to animate the *user's* avatar). Current focus is on 1-on-1 interactions. Group chats
    and visual novel mode are not supported.
- Far future:
  - Fast, high-quality scaling mechanism.
    - On a 4k display, the character becomes rather small, which looks jarring on the default backgrounds.
    - The algorithm should be cartoon-aware, some modern-day equivalent of waifu2x. A GAN such as 4x-AnimeSharp or Remacri would be nice, but too slow.
    - Maybe the scaler should run at the client side to avoid the need to stream 1024x1024 PNGs.
      - What JavaScript anime scalers are there, or which algorithms are simple enough for a small custom implementation?
  - Lip-sync talking animation to TTS output.
    - THA3 has morphs for A, I, U, E, O, and the "mouth delta" shape Δ.
    - This needs either:
      - Realtime data from client
      - Or if ST-extras generates the TTS output, then at least a start timestamp for the playback of a given TTS output audio file,
        and a possibility to stop animating if the user stops the audio.
  - Postprocessor for static character expression sprites.
    - This would need reimplementing the static sprite system at the `talkinghead` end (so that we can apply per-frame dynamic postprocessing),
      and then serving that as `result_feed`.
  - Group chats / visual novel mode / several talkingheads running simultaneously.
