import textwrap


VOCODER_INFO_FORK = textwrap.dedent("""\
    **Vocoder for audio synthesis:**
    
    **RefineGAN:**
    - **Arch overview:** ParallelResBlocks + AdaIN + Hn-NSF
    - **COMPATIBILITY:** Offline: This Fork or Applio, Streaming: not 100% sure.
    
    **RingFormer:**
    - **Arch overview:** A hybrid Resblocks+Conformer-Based Vocoder + RingAttention + Hn-NSF.
    - **COMPATIBILITY:** Offline: This Fork. Streaming: None.
    ⚠ This architecture is under a big question mark atm. ⚠
    
    **APEX-GAN:**
    - **Arch overview:** Pitch-Conditioned Snake ResBlocks + Conv-iSTFT Hybrid architecture.
    - **COMPATIBILITY:** Offline: This Fork. Streaming: None atm.
    
    **NOTES:**
    **( Offline = Static inference/Covers, Streaming = Real-Time voice changers )**
    **( Each Vocoder and its supported sample rates require appropriate pretrained models )**
""")


VOCODER_INFO_RVC = textwrap.dedent("""\
    **Vocoder for audio synthesis:**
    
    **HiFi-GAN:**
    - **Arch overview:** HiFi-GAN + Hn-NSF. ( RVC's og vocoder )
    - **COMPATIBILITY:** Offline: RVC, Fork, Applio. Streaming: W-okada, Vonovox
    
    **NOTES:**
    **( Offline = Static inference/Covers, Streaming = Real-Time voice changers )**
    **( Each Vocoder and its supported sample rates require appropriate pretrained models )**
""")


DATASET_TRUNCATION_INFO = textwrap.dedent("""\
<br/>

<span style="font-size: 25px; font-weight: bold;">1.  Codename;0's recommended approach:</span>
- Set Normalization to: **post_rms**
- Set Audio cutting to: **Simple**

<span style="font-size: 14px; font-weight: bold;">Dataset requirements:</span>
- **1 continuous / concatenated audio file** instead of many independent short segments/files.
- Apply **silence truncation** ( keeping short silence gaps around **80–120 ms** is fine ).
- Ensure your dataset isn't busted in terms of volume / dynamic range or peaks. ( What can help: Peak / RMS compression or "Leveler" if you have iZotope RX )

<span style="font-size: 14px; font-weight: bold;">EXTRA INFO:</span>
- **Audacity** is a great free tool for the first 2 things I mention above if you don't have another alternative.
- For quick concatenation, drag and drop a folder containing multiple audio files onto **run_concat.bat** located in the **EXTRAS** folder (**found in the root directory of this fork**).

<br/>

<span style="font-size: 25px; font-weight: bold;">2. Lazy / Low effort approach ( Not recommended.):</span>
- Set Normalization to: **post_peak**
- Set Audio cutting to: **Automatic**
<br/> ⚠ ( Yet I'd still recommend to go with the 1st approach if you want training stability and more consistent results.

<br/>

<span style="font-size: 25px; font-weight: bold;">3. Experimental alternative approach:</span>
- Enable "SmartCutter" by ticking the checkbox
- Set Normalization to: **post_rms**
- Set Audio cutting to: **Simple**

<span style="font-size: 14px; font-weight: bold;">Requirements:</span>
- **1 continuous / concatenated audio file** instead of many independent short segments/files.
<br/> ⚠ ( **There is a chance** SmartCutter might not work well on your dataset.. In that case just go for one of the above approaches. )

<span style="font-size: 14px; font-weight: bold;">EXTRA INFO:</span>
- SmartCutter was made with base/pretrain model creation in mind.
- The model is meant to detect silent/low-noise gaps that are +100ms long, replace them with digital-silence and trim to 100ms.
- By design should respect zero-crossings and avoid cutting into breaths and organic sounds but might make mistakes.

<br/>

` NOTES ` <br/>
1. Remember.. Dataset is the very foundation of your model. If you use awful datasets, don't expect miraculous results.. AI is not a magical tool like that.
2. Generally.. you shouldn't tweak these default settings unless you know you're doing it.
""")


PREPROCESS_RMS_VALUE_INFO = textwrap.dedent("""\
Set your RMS target for 'post_rms' normalization mode. If your dataset isn't suitable for a given dBFS ( clipping occurs), **it'll get auto-adjusted automatically to whatever is safe**.
If you want to squeeze out more volume out of your dataset without clipping, consider performing dynamic-range compression or peak-compression on your dataset beforehand.
""")


DATASET_FORMAT_INFO = textwrap.dedent("""\
    **Sliced audio storage format**

    Controls which format is used for storing sliced audio files.

    **WAV:** Uncompressed PCM audio. Much larger files but preserves the exact waveform.
    - Recommended if your source files are 32-bit float (e.g., preprocessed audio).

    **FLAC:** Lossless compressed audio codec. Much smaller files and safe for training.
    - Useful if storage space is limited.
""")


RESAMPLER_INFO = textwrap.dedent("""\
- **librosa:** Uses SoX resampler
( SoXr set to VHQ by default. )

- **ffmpeg:** Uses SW resampler
( Windowed Sinc filter with Blackman-Nuttall window. ) 

**Both are fine, but SoXr has better anti-aliasing.**
""")


SMARTCUTTER_INFO = textwrap.dedent("""\
**Machine-Learning solution to silence truncation**
Created especially with this fork in mind.
- Automatically truncates the silences ( Ensuring min. 100ms spacings )
- Doesn't damage word-tails or inter-phonetic gaps ( unlike gating )
- Truncated areas are automatically replaced by pure silence
( in case of noise-contamination between words or sentences. )
- Tries to heavily respect breathing.

⚠ **Due to technical limitations, multi-spk processing will be slower.**
""")


NORMALIZATION_INFO = textwrap.dedent("""\
- **none:** Disabled
( Select this if the files are already normalized. )
- **post_peak:** Peak post-norm,
( Peak [ * 0.95] norm of each slice. )
- **post_peak_rvc:** Peak post-norm with alpha blend
( Peak [ max amp * alpha] norm of each slice. )
- **post_rms:** RMS-based post-norm
( Configurable RMS target (dBFS) for each slice. )
""")


AUDIO_FILE_SLICING_INFO = textwrap.dedent("""\
**Audio file slicing-method selection:**
- **Skip:** if the files are already pre-sliced and properly normalized.
- **Simple:** If your dataset is already silence-truncated or well behaving in terms of spaces / gaps.
- **Automatic:** for automatic silence detection and slicing around it.
**It is advised to go for SmartCutter or Universal approach.**
**( PS. Automatic is pretty crap. I advise against it unless your set's clean and you can't bother truncating it. )**
""")


PITCH_EXTRACTION_INFO = textwrap.dedent("""\
**Pitch extraction algorithm to use for the audio conversion:**

**RMVPE:** The default algorithm, recommended for most cases.
- The fastest, very robust to noise. Can tolerate harmonies / layered vocals to some degree.

**CREPE:** Better suited for truly clean audio.
- Is slower and way worse in handling noise. Can provide different / softer-ish results.

**CREPE-TINY:** Smaller / lighter variant of CREPE.
- Performs worse than 'full' ( standard crepe ) but is way lighter on hardware.
""")


BATCH_SIZE_INFO = textwrap.dedent("""\
**[ TOO LARGE BATCH SIZE CAN LEAD TO VRAM 'OOM' ISSUES. ]**

Bigger batch size:
- Promotes smoother, more stable gradients.
- Can be beneficial in cases where your dataset is big and diverse.
- Can lead to early overtraining or flat / ' stuck ' graphs on small datasets.
- Generalization might be worsened.
- **Favors faster learning rate.**

Smaller batch size:
- Promotes noisier, less stable gradients.
- More suitable when your dataset is small, less diverse or repetitive.
- Can lead to instability / divergence or noisy as hell graphs.
- Generalization / High-pitch performance might be improved.
- **Favors slower learning rate.**
""")


SPECTRAL_LOSS_INFO = textwrap.dedent("""\
- **L1 Mel Loss:** L1 loss using mel spec - **Safe default.**
- **Multi-Scale Mel Loss:** Multi-scale L1 mel spec loss.
- **Hybrid:** L1 or Multi-Scale Mel + Multi-res. A-Weighted log-STFT spec loss.

**NOTE:** Hybrid in "L1" version is likely more stable but it varies per-case.
""")


LR_SCHEDULER_INFO = textwrap.dedent("""\
- **exp decay:** Decays the lr exponentially - **Safe default.**
**( 'step' variant decays per step, 'epoch' per epoch. )**
- **cosine annealing:** Cosine annealing schedule - **Optional alternative.**
- **none:** No scheduler - **For debugging or developing.**
""")


KL_ANNEALING_INFO = textwrap.dedent("""\
**Enables cyclic KL loss annealing for training.**
- Might potentially mitigate overfitting on smaller datasets.
- Generally should help with convergence.

**(EXPERIMENTAL)**
""")

KL_ANNEALING_CYCLE_INFO = textwrap.dedent("""\
Determines the duration of each repeating annealing cycle.
Limited testing showed 3 epochs is the most optimal,
but you can experiment for yourself.
**( Duration in epochs )**
""")

OPTIMIZER_INFO = textwrap.dedent("""\
Choose an optimizer used in training:
( If unsure, just leave it as it is or try these in this order: AdamW -> AdaBelief -> RAdam. )

- **AdamW:** Default; Safe and reliable.
- **AdaBelief:** Adapts step size by "belief" in the gradient direction. ( **Likely more stable than AdamW in GANs** )
- **RAdam:** Rectified Adam. ( **Can help** with early instability - **Most likely slower convergence** )
- **Ranger21:** AdamW + LookAhead and few more extras. ( **Most likely unstable** )
""")

