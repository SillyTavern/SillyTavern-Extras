## Talkinghead TODO


### High priority

As of January 2024, preferably to be completed before the next release.

#### Backend

- Add configurable crop filter (after posing, before postproc) to trim unused space around the sides of the character, to allow better positioning of the character in **MovingUI** mode.
- Postprocessor: make real brightness filters, to decouple translucency from all other filters.
  - Currently many of the filters abuse the alpha channel as a luma substitute, which looks fine for a scifi hologram, but not for some other use cases.
  - Need to convert between RGB and some other color space. Preferably not YUV, since that doesn't map so well to RGB and back.
      https://stackoverflow.com/questions/17892346/how-to-convert-rgb-yuv-rgb-both-ways
      https://www.cs.sfu.ca/mmbook/programming_assignments/additional_notes/rgb_yuv_note/RGB-YUV.pdf
  - Maybe HSL, or HCL, or a combined strategy from both, like in this R package:
      https://colorspace.r-forge.r-project.org/articles/manipulation_utilities.html
- Add a server-side config for animator and postprocessor settings.
  - For symmetry with emotion handling; but also foreseeable that target FPS is an installation-wide thing instead of a character-wide thing.
    Currently we don't have a way to set it installation-wide.

#### Frontend

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


### Medium priority

Maybe some time in the near-ish future. Would be nice to have in the next release.

#### Frontend

- When a new talkinghead sprite is uploaded:
  - The preview thumbnail in the client doesn't update. The same goes for the other sprites, so this is a general bug in *Character Expressions*.

- Not related to talkinghead, but since I have a TODO list here, I'm dumping notes on some potentially easily fixable things here instead of opening a ticket for each one:
  - In *Manage chat files*, when using the search feature, clicking on a search result either does nothing,
    or opens the wrong chat (often the latest one, whether or not it matched the search terms). When not searching,
    clicking on a previous chat correctly opens that specific chat.
  - *Render Formulas* shows both the rendered formula and its plaintext. Would look better to show only the rendered formula, unless the user wants to edit it
    (like the inline LaTeX equation renderer in Emacs).
  - Missing tooltips:
    - **MovingUI** (*User Settings ⊳ Advanced*): "Allow repositioning certain UI elements by dragging them."
      - **MUI Preset** = ??? Is this a theme selector for MovingUI, affecting how the dragging GUI looks, or something else?
    - **No WI/AN** (Extensions ⊳ Vector Storage ⊳ Chat vectorization settings): "Do not vectorize World Info and Author's Note."
    - **Depth** (appears in many places): "How many messages before the current end of the chat."
      - I think this is important to clarify, because at least to a programmer, "depth" first brings to mind nested brackets; and brackets are actually used in ST,
        to make parenthetical remarks to the AI (such as for summarization: "[Pause your roleplay. Summarize...]").
    - **AI Response Configuration**:
      - **Top P**: Otherwise fine, but maybe mention that Top P is also known as nucleus sampling.
      - **Top A**: Relative of Min P, but operates on squared probabilities.
        - See https://www.reddit.com/r/KoboldAI/comments/vcgsu1/comment/icrp0n1
      - **Tail Free Sampling**: "Estimates where the 'knee' of the next-token probability distribution is, and cuts the tail off at that point."
        - I would assume the slider controls the `z` value, but this should be confirmed from the source code.
        - See https://www.trentonbricken.com/Tail-Free-Sampling/
      - **Typical P** = ???
      - **Epsilon Cutoff** = ???
      - **Eta Cutoff** = ???
      - **Mirostat**: "Thermostat for output perplexity. Controls the output perplexity directly, to match the perplexity of the input. This avoids the
        repetition trap (where, as the autoregressive inference produces text, the perplexity of the output tends toward zero) and the confusion
        trap (where the perplexity diverges)."
        - See https://arxiv.org/abs/2007.14966
        - In practice, Min P can lead to similarly good results, while being simpler and faster. Should we mention this?
      - **Beam Search** = ???
        - At least it's the name of a classical optimization method in numerics. Also, in LLM sampling, beam search is infamous for its bad performance;
          easily gets stuck in a repetition loop (which hints that it always picks tokens that are too probable, decreasing output perplexity).
          I think this was mentioned in one of the Contrastive Search papers.
      - **Contrast Search**: "The representation space of most LLMs is isotropic, and this sampler exploits that in order to encourage diversity while maintaining coherence."
        - Name should be "Contrastive Search"
        - In math terms, this is a minor modification to an older, standard sampling strategy. Have to re-read the paper to check details.
          In any case, the penalty alpha controls the relative strength of the regularization term.
        - See https://arxiv.org/abs/2202.06417 , https://arxiv.org/abs/2210.14140
        - In practice this method produces pretty good results, just like Min P does.
      - **Temperature Last**: We should probably emphasize that Temperature Last is the sensible thing to do: pick the set of plausible tokens first, then tweak their
        relative probabilities (actually logits). Don't tweak the full distribution first, and then pick the token set from that, because this tends to amplify
        the probability of an incoherent response too much (which is what happens if Temperature Last is off).
      - **CFG**: Context Free Guidance.
        - Should also explain what it does... at least ooba uses CFG to control the strength of the negative prompt?
    - *User Settings ⊳ Advanced*:
      - **No Text Shadows**: obvious, but missing a tooltip
      - **Visual Novel Mode**: what exactly does VN mode do, and how does it relate to group chats? What does it do in a 1-on-1 chat? Maybe needs a link to the manual, or something.
      - **Expand Message Actions** = ??? What are message actions?
      - **Zen Sliders** = ???
      - **Mad Lab Mode** = ???
      - **Message Timer**: "Time the AI's message generation, and show the duration in the chat log."
      - **Chat Timestamps**: obvious, but missing a tooltip
      - **Model Icons** = ???
      - **Message IDs**: "Show message numbers in the chat log."
      - **Message Token Count**: "Show number of tokens in each message in the chat log."
      - **Compact Input Area** = ??? Nothing happens when toggling this on PC.
      - **Characters Hotswap**: "In the Character Management panel, show quick selection buttons for favorited characters."
      - **Tags as Folders** = ??? What are tags? How to use them? Link to manual?
      - **Message Sound** = ??? Has a link to the manual, could extract a one-line summary from there.
      - **Background Sound Only** = ???
      - **Custom CSS** = ??? What is the scope where the custom style applies? Just MovingUI, or the whole ST GUI? Where to get an example style to learn how to make new ones?
      - **Example Messages Behavior**: obvious, but missing a tooltip
      - **Advanced Character Search** = ???
      - **Never resize avatars** = ???
      - **Show avatar filenames** = ??? This seems to affect the *Character Management* panel only, not *Character Expressions* sprites?
      - **Import Card Tags** = ??? Something to do with the PNG character card thing?
      - **Spoiler Free Mode** = ???
      - **"Send" to Continue** = ??? Sending the message to the AI continues the last message instead of generating a new one? How do you generate a new one, then?
      - **Quick "Continue" button**: "Show a button in the input area to ask the AI to continue (extend) its last message."
      - **Swipes**: "Generate alternative responses before choosing which one to commit. Shows arrow buttons next to the AI's last message."
      - **Gestures** = ???
      - **Auto-load Last Chat**: obvious, but missing a tooltip
      - **Auto-scroll Chat**: obvious, but missing a tooltip
      - **Auto-save Message Edits** = ??? When does the autosave happen?
      - **Confirm Message Deletion**: obvious, but missing a tooltip
      - **Auto-fix Markdown** = ??? What exactly does it fix in Markdown, and using what algorithm?
      - **Render Formulas**: "Render LaTeX and JSMath equation notation in chat messages."
      - **Show {{char}}: in responses**: obvious, but missing a tooltip
      - **Show {{user}}: in responses**: obvious, but missing a tooltip
      - **Show tags in responses** = ???
      - **Log prompts to console**: obvious, but missing a tooltip
      - **Auto-swipe**: obvious once you expand the panel and look at the available settings, but missing a tooltip.
        "Automatically reject and re-generate AI message based on configurable criteria."
      - **Reload Chat** = ??? What exactly gets reloaded?
    - Probably lots more. Maybe open a ticket and start fixing these?


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
  - Analog video glitches
    - Partition image into bands, move some left/right temporarily (for a few frames now that we can do that)
    - Another effect of bad VHS hsync: dynamic "bending" effect near top edge:
      - Distortion by horizontal movement
      - Topmost row of pixels moves the most, then a smoothly decaying offset profile as a function of height (decaying to zero at maybe 20% of image height, measured from the top)
      - The maximum offset flutters dynamically in a semi-regular, semi-unpredictable manner (use a superposition of three sine waves at different frequencies, as functions of time)
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

#### Frontend

- Add live-modifiable configuration for animation and postprocessor settings?
  - Add a new control panel to SillyTavern client extension settings
  - Send new configs to backend whenever anything changes

#### Both frontend and backend

- To save GPU resources, automatically pause animation when the web browser window with SillyTavern is not in focus. Resume when it regains focus.
  - Needs a new API endpoint for pause/resume. Note the current `/api/talkinghead/unload` is actually a pause function (the client pauses, and
    then just hides the live image), but there is currently no resume function (except `/api/talkinghead/load`, which requires sending an image file).


### Far future

Definitely not scheduled. Ideas for future enhancements.

- Fast, high-quality output scaling mechanism.
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
