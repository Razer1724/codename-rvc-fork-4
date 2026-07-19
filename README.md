# <p align="center">` Codename-RVC-Fork 🍇 4 ` </p>
<div align="center">
  
` " The spiritual successor to Mangio-RVC-Fork "`

</div>

#### <p align="center"> ✨ Originally based on Applio, evolved into its own independent project ✨</p>

<p align="center">
  ㅤㅤTo stay up-to-date with advancements, hang out or get supportㅤㅤ<br>
  ㅤㅤyou can join my 👇 discord server 👇 ( RVC / AI Audio friendly )ㅤㅤ
</p>

<p align="center">
  <!-- GitHub Release Version -->
  <a href="https://github.com/codename0og/codename-rvc-fork-4/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/codename0og/codename-rvc-fork-4?include_prereleases&style=flat-for-the-badge&color=8a2be2" alt="Latest Release">
  </a>
  
  <!-- Discord Invite -->
  <a href="https://discord.gg/ymfdwx5jwZ" target="_blank">
    <img src="https://img.shields.io/badge/Discord-Join_Sanctuary-7289da?style=flat-for-the-badge&logo=discord&logoColor=white" alt="Discord Online">
  </a>
  
  <!-- GitHub Repo Size -->
  <img src="https://img.shields.io/github/repo-size/codename0og/codename-rvc-fork-4?style=flat-for-the-badge&color=333" alt="Repository Size">

  <!-- License -->
  <img src="https://img.shields.io/github/license/codename0og/codename-rvc-fork-4?style=flat-for-the-badge&color=success" alt="License">
</p>
ㅤ

# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
<br/>
 
`1. Datasets must be processed properly:`
- In case of wild dynamic range or inconsistent recording sessions, peak / rms compression is advised.
- silence-truncating your dataset ( Or at least ensure the gaps / silences aren't too crazy or inconsistent. ) 

`2. Experimental things are experimental for a reason:`
- If you don't understand what it does, what it brings or how it works? preferably don't use it or ask on my server.
- Certain features / currently chosen params can be potentially unstable or broken and are a subject to change.
- Some experimental things can get removed at any point if deemed too unstable / not worth the risk.

`3. Clarification on pretrained models, architectures & vocoders:`
- **Each Architecture/Vocoder requires own dedicated pretrains.**
##### 1. HiFi-GAN ( RVC architecture ):
- The original architecture. ( HiFi-GAN + MPD, MSD )
- Its pretrained models are auto-downloaded during the first launch.
- Available for sample rates: 48, 40 and 32khz. <br/><br/>`Models made with this arch ARE cross-compatible: RVC, Applio and codename-rvc-fork-4.` 
##### 2. RefineGAN ( Fork architecture ):
- Custom architecture. ( RefineGAN + MPD, MSD, MRD )
- **There are no available pretrained models for it yet. ( Applio's one is incompatible. )**<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4`
##### 3. RingFormer ( Fork architecture ):
- This architecture remains in question ~ Might get removed, might get updated.
##### 4. APEX-GAN ( Fork architecture ):
- Custom architecture. ( APEX-GAN + MPD, SBD, MRD )
- **There are no available pretrained models for it yet. Currently in "trials+polishing" phase.**
- Supported sample rates: 24, 32, 40 and 48khz.<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4` 
<br/>

# **Things exclusive to my fork:**
 
- F0 / Pitch curve editor for inference integrated in the UI.
 
- Many available optimizers.  ` ( AdamW, AdaBelief, RAdam, Ranger21, Schedule-Free AdamW/RAdam ) `
 
- Decoupled G/D Tweaking: Schedulers, Optims, Learning Rates etc.
 
- Support for Multi-scale L1 Mel, classic L1 mel and Hybrid ( L1 Mel + MS-STFT ) spectral losses.
 
- Extras such as: Kl loss annealing, 2-sample KL loss calculation, Double-Update for Discriminator and more..
 
- Support for the following vocoders: HiFi-GAN-NSF, Refine-GAN, RingFormer, APEX-GAN.<br/>
`( And potentially more in future ..)`
 
- Support for many discriminator stacks: Avocodo's, mps/msd/mrd, mpd/sbd/mrd, hmddd and more..<br/>
`( Naturally they require pretrained models. )`
 
- Much better loss logging handling.<br/>
`( Per-epoch-avg loss as the main one, rolling avg as the long-term one. )`
 
- More dataset-preprocessing options and generally simplified workflow ( RMS norm has dbFS auto-correction. ).
 
- Lots of deeper training-related tweaks directly in the ui.<br/>
` ( lr for g/d, schedulers, linear warmup, kl loss annealing and much more .. )`
 
- Direct integration of `SmartCutter` - My own ml-based silence-truncation approach.
<br/>[More info](https://github.com/codename0og/SmartCutter)
 
- Various speed, performance and QOL improvements.
 
- A much cleaner, continuously evolving codebase compared to existing alternatives.
 
**Any new / experimental features are always described in releases so, feel free to check it out there.**
  
 
 
 <br/>
 
 
✨ to-do list ✨
> - Need to figure out a better / more appropriate validation..
 
💡 Ideas / concepts 💡
> - Upscaling / Refinement for Inference output and for datasets.
> - If you have some nice ideas, feel free to share 'em or PR!
 
 
### ❗ For contact, please join my discord server ❗
 <br/>
 <br/>

## Getting Started:

### 1. Installation of the Fork

Run the installation script:

- Double-click `run-install.bat`.

### 2. Running the Fork

Use the run script:

- Double-click `run-fork.bat`.
 
This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring
 
To monitor training or visualize data:
- Drag the 'eval' folder onto "run_tensorboard_in_model_folder.bat" ( you can copy it from logs dir -> your model's dir ).
</br></br>If it doesn't work for you due to blocked port, open up CMD with admin rights and use this command:</br>`` netsh advfirewall firewall add rule name="Open Port 25565" dir=in action=allow protocol=TCP localport=25565 ``</br></br>
- Alternatively if the above method fails, run the tensorboard manually in cmd:</br> ``tensorboard --logdir="path/to/your/model/folder" --bind_all``</br>
(PS. Make sure you have tensorboard installed. ( in cmd:  pip install tensorboard )
 
## Referenced projects
+ [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
+ [Applio](https://github.com/IAHispano/Applio)
+ [RingFormer](https://github.com/seongho608/RingFormer)
+ [RiFornet](https://github.com/Respaired/RiFornet_Vocoder)
+ [BigVGAN](https://github.com/NVIDIA/BigVGAN/tree/main)
+ [Pytorch-Snake](https://github.com/falkaer/pytorch-snake)
+ [wavehax](https://github.com/chomeyama/wavehax)
+ [auraloss](https://github.com/csteinmetz1/auraloss/tree/main)
+ [Avocodo](https://github.com/ncsoft/avocodo)
+ [HiFTNet](https://github.com/yl4579/HiFTNet)
 
## Disclaimer
``The creators, maintainers, and contributors of the original Applio repository, as well as the creator of this fork (Codename;0), which is based on Applio, and the contributors of this fork, are not liable for any legal issues, damages, or consequences arising from the use of this repository or any content generated from it. By using this fork, you acknowledge and accept the following terms:``
 
- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-4/blob/main/TERMS_OF_USE.md)
