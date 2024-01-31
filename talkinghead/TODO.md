## Talkinghead TODO


### High priority

As of January 2024, preferably to be completed before the next release.

#### Frontend

- Add a slash command to enable/disable `talkinghead` mode for *Character Expressions*. This could be bound to a Quick Reply for a single-button toggle.
- See if we can get this working also with local classify now that we have a `set_emotion` API endpoint.
  - Responsibilities: the client end should set the emotion when it calls classify, instead of relying on the extras server doing it internally when extras classify is called.
- Figure out why the crop filter doesn't help in positioning the `talkinghead` sprite in *MovingUI* mode.
  - There must be some logic at the frontend side that reserves a square shape for the talkinghead sprite output,
    regardless of the image dimensions or aspect ratio of the actual `result_feed`.
- Check zip upload whether it refreshes the talkinghead character (it should).
- Switching `talkinghead` mode on/off in Character Expressions should set the expression to the current emotion.
  - The client *does* store the emotion, as evidenced by this quick reply STScript:
    `/lastsprite {{char}} | /echo Current sprite of {{char}}: {{pipe}}`
    So we should find what implements the slash command `/lastsprite`, to find where the emotion is stored.
- If `classify` is enabled, emotion state should be updated from the latest AI-generated text
  when switching chat files, to resume in the same emotion state where the chat left off.
  - Use the expression setting mechanism to set the emotion.
  - Investigate what calls `/api/classify` (other than the expression setting code in Character Expressions); classifying updates the talkinghead state.
    We should make the same code (at the client end) also update the sprite if Character Expressions is enabled, and call that code after switching to a different chat.
- Are there other places in *Character Expressions* (`SillyTavern/public/scripts/extensions/expressions/index.js`)
  where we need to check whether the `talkinghead` module is enabled? `(!isTalkingHeadEnabled() || !modules.includes('talkinghead'))`

#### Documentation

- Polish up the documentation for release:
  - Add pictures to the talkinghead README.
    - Screenshot of the manual poser. Anything else we should say about it?
    - Examples of generated poses, highlighting both success and failure cases. How the live talking head looks in the actual SillyTavern GUI. Link the original THA tech reports.
    - Examples of postprocessor filter results.
    - How each postprocessor example config looks when rendering the example character.
  - Merge appropriate material from old user manual into the new README.
  - Update/rewrite the user manual, based on the new README.
    - This should replace the old manual at https://docs.sillytavern.app/extras/extensions/talkinghead/

#### Examples

- Add some example characters created in Stable Diffusion.
  - Original characters only, as per ST content policy.
  - Maybe we should do Seraphina, since she's part of a default SillyTavern install?


### Low priority

Not scheduled for now.

#### Backend

- Low compute mode: static poses + postprocessor.
  - Poses would be generated from `talkinghead.png` using THA3, as usual, but only once per session. Each pose would be cached.
  - To prevent postproc hiccups (in dynamic effects such as CRT TV simulation) during static pose generation in CPU mode, there are at least two possible approaches.
    - Generate all poses when the plugin starts. At 2 FPS and 28 poses, this would lead to a 14-second delay. Not good.
    - Run the postprocessor in a yet different thread, and postproc the most recent poser output available.
      - This would introduce one more frame of buffering, and split the render thread into two: the poser (which is 99% of the current `Animator`),
        and the postprocessor (which is invoked by `Animator`, but implemented in a separate class).
  - This *might* make it feasible to use CPU mode for static poses with postprocessing.
    - But I'll need to benchmark the postproc code first, whether it's fast enough to run on CPU in realtime.
  - Alpha-blending between the static poses would need to be implemented in the `talkinghead` module, similarly to how the frontend switches between static expression sprites.
    - Maybe a clean way would be to provide different posing strategies (alternative poser classes): realtime posing, or static posing with alpha-blending.
- Small performance optimization: see if we could use more in-place updates in the postprocessor, to reduce allocation of temporary tensors.
  - The effect on speed will be small; the compute-heaviest part is the inference of the THA3 deep-learning model.
- Add more postprocessing filters. Possible ideas, no guarantee I'll ever get around to them:
  - Pixelize, posterize (8-bit look)
  - Digital data connection glitches
    - Apply to random rectangles; may need to persist for a few frames to animate and/or make them more noticeable
    - Types:
      - Constant-color rectangle
      - Missing data (zero out the alpha?)
      - Blur (leads to replacing by average color, with controllable sigma)
      - Zigzag deformation (perhaps not needed now that we have `shift_distort`, which is similar, but with a rectangular shape, and applied to full lines of video)
- Investigate if some particular emotions could use a small random per-frame oscillation applied to "iris_small",
  for that anime "intense emotion" effect (since THA3 doesn't have a morph specifically for the specular reflections in the eyes).

#### Frontend

- Add a way to upload new JSON configs (`_animator.json`, `_emotions.json`), because ST could be running on a remote machine somewhere.
  - Send new uploaded config to backend.
- Add live-modifiable configuration for animation and postprocessor settings.
  - Add a new control panel to SillyTavern client extension settings.
  - Send new configs to backend whenever anything changes.

#### Both frontend and backend

- To save GPU resources, automatically pause animation when the web browser window with SillyTavern is not in focus. Resume when it regains focus.
  - Needs a new API endpoint for pause/resume. Note the current `/api/talkinghead/unload` is actually a pause function (the client pauses, and
    then just hides the live image), but there is currently no resume function (except `/api/talkinghead/load`, which requires sending an image file).
- Lip-sync talking animation to TTS output.
  - THA3 has morphs for A, I, U, E, O, and the "mouth delta" shape Δ.
  - This needs either:
    - Realtime data from client
      - Exists already! See `SillyTavern/public/scripts/extensions/tts/index.js`, function `playAudioData`. There's lip sync for VRM (VRoid).
        Still need to investigate how the VRM plugin extracts phonemes from the audio data.
    - Or if ST-extras generates the TTS output, then at least a start timestamp for the playback of a given TTS output audio file,
      and a possibility to stop animating if the user stops the audio.

### Far future

Definitely not scheduled. Ideas for future enhancements.

- Fast, high-quality output scaling mechanism.
  - On a 4k display, the character becomes rather small, which looks jarring on the default backgrounds.
  - The algorithm should be cartoon-aware, some modern-day equivalent of waifu2x. A GAN such as 4x-AnimeSharp or Remacri would be nice, but too slow.
  - Maybe the scaler should run at the client side to avoid the need to stream 1024x1024 PNGs.
    - What JavaScript anime scalers are there, or which algorithms are simple enough for a small custom implementation?
- Group chats / visual novel mode / several talkingheads running simultaneously.
