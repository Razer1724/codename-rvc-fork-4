# <p align="center">` Codename-RVC-Fork 🍇 4 ` </p>
## <p align="center">Based on Applio</p>

<p align="center"> ㅤㅤ👇 You can join my discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/ymfdwx5jwZ" target="_blank"> Codename's Sanctuary</a>
</p>

<p align="center"> ㅤㅤ👆 To stay up-to-date with advancements, hang out or get support 👆ㅤㅤ </p>


## A lil bit more about the project:

### This fork is pretty much my personal take on Applio. ✨
``You could say.. A more advanced features-rich Applio ~ With my lil twist.``
<br/>
``But If you have any ideas, want to pr or collaborate, feel free to do so!``
<br/>
ㅤ
<br/>
# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
`1. Datasets must be processed properly:`
- In case of wild dynamic range or inconsistent recording sessions, peak / rms compression is advised.
- silence-truncating your dataset ( Or at least ensure the gaps / silences aren't too crazy or inconsistent. ) 

`2. Experimental things are experimental for a reason:`
- If you don't understand what it does, what it brings or how it works? preferably don't use it.
- Certain features / currently chosen params can be potentially unstable or broken and are a subject to change.
- Some experimental things might disappear at some point if deemed too unstable / not worth the risk.

`3. Clarification on pretrained models, architectures & vocoders:`
- **Each Architecture/Vocoder requires own dedicated pretrains.**
##### 1. HiFi-GAN ( RVC architecture ):
- The original architecture. ( HiFi-GAN + MPD, MSD )
- It's pretrained models are auto-downloaded during the first launch.
- Available for sample rates: 48, 40 and 32khz. <br/><br/>`Models made with this arch are cross-compatible: RVC, Applio and codename-rvc-fork-4.` 
##### 2. RefineGAN ( Fork / Applio architecture ):
- Custom architecture. ( RefineGAN + MPD, MSD )
- **Pretrains available. For more info, visit my discord server.** <br/><br/>`Models made with this arch are LIMITED cross-compatible: codename-rvc-fork-4 and Applio`
##### 3. RingFormer ( Fork architecture ):
- As it is right now, this architecture remains in question.
##### 4. APEX-GAN ( Fork architecture ):
- A direct continuation of the previous vocoder, known under the alias "PCPH-GAN".
- Custom architecture. ( APEX-GAN + CoMBD, MSB, UnivHD )
- **There are no available pretrained models for it yet. Currently in "trials+polishing" phase.**
- Supported sample rates: 24, 32, 40 and 48khz.<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4` 
<br/>

# **Fork's exclusive features:**
 
- My own ml-based silence-truncation approach.
<br/>[More info](https://github.com/codename0og/SmartCutter)
 
- F0 / Pitch curve editor for inference integrated in the UI.
 
- Support for 'Spin' embedder. ` ( and perhaps more in future. ) `
 
- Many available optimizers.  ` ( AdamW, RAdam, AdamSPD, Ranger21, DiffGrad ) `
 
- Different adversarial losses to try. ` ( Available: lsgan, hinge, tprls. [ lsgan is the safe / rvc's default one. ] ) `
 
- Support for Multi-scale, classic L1 mel and (EXP) multi-resolution stft spectral losses.
 
- Support for some of VITS2 enhancements.
`( Transformer-enhanced normalizing flow + spk conditioned text encoder. )`<br/>
`( Requires pretrains that were trained with it. )`
 
- Support for the following vocoders: HiFi-GAN-NSF, Refine-GAN, RingFormer, APEX-GAN.<br/>
 
- Much better loss logging handling.
`( Per-epoch-avg loss as the main one, rolling avg as the long-term one )`
 
- More sophisticated dataset-preprocessing approach.
 
- Lots of deeper training-related tweaks directly in the ui. ` ( lr for g/d, schedulers, linear warmup, kl loss annealing and much more .. )`
 
- Direct integration of [SmartCutter](https://github.com/codename0og/SmartCutter)
 
- Various speed and performance improvements.

**Any new / experimental features are always described in releases so, feel free to check it out there.**
  
 
 
 <br/>
 
 
✨ to-do list ✨
> - Better long-term logging for pretrained / base models training.
> - Some additional feedback during training in terms of model's performance.
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

 
## Disclaimer
``The creators, maintainers, and contributors of the original Applio repository, as well as the creator of this fork (Codename;0), which is based on Applio, and the contributors of this fork, are not liable for any legal issues, damages, or consequences arising from the use of this repository or any content generated from it. By using this fork, you acknowledge and accept the following terms:``
 
- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md)
