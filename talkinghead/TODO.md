## Talkinghead TODO

### Live mode

- Add a server-side config for animator and postprocessor settings.
  - For symmetry with emotion handling; but also foreseeable that target FPS is an installation-wide thing instead of a character-wide thing.
    Currently we don't have a way to set it installation-wide.
- Fix timing of microsway based on 25 FPS reference.
- Fix timing of dynamic postprocessor effects, these should also use a 25 FPS reference.
- Postprocessor for static character expression sprites.
  - This would need reimplementing the static sprite system at the `talkinghead` end (so that we can apply per-frame dynamic postprocessing),
    and then serving that as `result_feed`.
  - Easier solution is to fake it: just invoke the poser once for the target pose of each expression (lazily, as each expression is first seen),
    and cache the results. The engine should actually already do this, it seems to use some sort of cached policy. Disable just the animation parts.
    This would still use THA3, but without animation, and would likely be usable in CPU mode, possibly even with some light postprocessing.
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

- Switching `talkinghead` mode on/off in Character Expressions should set the expression to the current emotion.
  - The client *does* store the emotion, as evidenced by this quick reply STScript:
      /lastsprite {{char}} | /echo Current sprite of {{char}}: {{pipe}}
    So we should find what implements the slash command `/lastsprite`, to find where the emotion is stored.
- If `classify` is enabled, emotion state should be updated from the latest AI-generated text
  when switching chat files, to resume in the same emotion state where the chat left off.
  - Use the expression setting mechanism to set the emotion.
  - Investigate what calls `/api/classify` (other than the expression setting code in Character Expressions); classifying updates the talkinghead state.
    We should make the same code (at the client end) also update the sprite if Character Expressions is enabled, and call that code after switching to a different chat.
- When a new talkinghead sprite is uploaded:
  - The preview thumbnail in the client doesn't update.
- Not related to talkinghead, but client bug, came up during testing: in *Manage chat files*, when using the search feature,
  clicking on a search result either does nothing, or opens the wrong chat. When not searching, clicking on a previous chat
  correctly opens that specific chat.
- Are there other places in *Character Expressions* (`SillyTavern/public/scripts/extensions/expressions/index.js`)
  where we need to check whether the `talkinghead` module is enabled? `(!isTalkingHeadEnabled() || !modules.includes('talkinghead'))`
- Check zip upload whether it refreshes the talkinghead character (it should).

### Common

- Add pictures to the talkinghead README.
  - Screenshot of the manual poser. Anything else the user needs to know about it?
  - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI.
  - Examples of postprocessor filter results.
- Merge appropriate material from old user manual into the new README.
- Update/rewrite the user manual, based on the new README.
- Far future:
  - To save GPU resources, automatically pause animation when the web browser window with SillyTavern is not in focus. Resume when it regains focus.
    - Needs a new API endpoint for pause/resume. Note the current `/api/talkinghead/unload` is actually a pause function (the client pauses, and
      then just hides the live image), but there is currently no resume function (except `/api/talkinghead/load`, which requires sending an image file).
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
  - Group chats / visual novel mode / several talkingheads running simultaneously.
