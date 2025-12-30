# <p align="center">` Codename-RVC-Fork 🍇 4 ` </p>
## <p align="center">Based on Applio</p>

<p align="center"> ㅤㅤ👇 You can join my discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/nQFpNBvvd3" target="_blank"> Codename's Sanctuary</a>
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
- Peak or RMS compression if necessary! ( This step isn't covered by the fork's preprocessing btw.)
- Silence-truncation ( Absolutely necessary. )
- 'simple' method chosen for preprocessing ( Even 3 sec segments. )
- Enable Loudness Normalization in the ui.
- Enable automatic LUFS range finder for Loudness Normalization. <br/>
``Expect issues with PESQ and data alignment If the following requirements are not met.``


`2. Experimental things are experimental for a reason:`
- If you don't understand what it does, what it brings or how it works? preferably don't use it.
- Certain features / currently chosen params can be potentially unstable or broken and are a subject to change.
- Not all experimental features gonna reach "stable" status ( There's only as much I can test/ablation study on my own. )
- Some experimental things might disappear at some point if deemed too unstable / not worth it.

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
- Custom architecture. ( RingFormer + MPD, MSD, MRD )
- **There are no available pretrained models for it. atm it's unsure if there will be any.**
- Supported sample rates: 24, 32, 40 and 48khz.<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4` 
##### 4. PCPH-GAN ( Fork architecture ):
- Custom architecture. ( PCPH-GAN + MPD, MSD, MRD )
- **There are no available pretrained models for it yet. Currently in test/prototyping phase.**
- Supported sample rates: 32, 40 and 48khz.<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4` 
<br/>

# **Fork's exclusive features:**
 
- Hold-Out type validation mechanism during training. `( L1 MEL, mrSTFT, PESQ, SI-SDR )`
 
- Support for 'Spin' embedder. ` ( and perhaps more in future. ) `
 
- Many available optimizers.  ` ( AdamW [and optimi variant for bf16), RAdam, AdamSPD, Ranger21, DiffGrad, Prodigy ) `
 
- Different adversarial losses to try. ` ( Available: lsgan, hinge, tprls. [ lsgan is the safe / rvc's default one. ] ) `
 
- Support for Multi-scale, classic L1 mel and (EXP) multi-resolution stft spectral losses.
 
- Support for some of VITS2 enhancements.
`( Transformer-enhanced normalizing flow + spk conditioned text encoder. )`<br/>
`( Requires pretrains that were trained with it enabled. )`
 
- Support for the following vocoders: HiFi-GAN-NSF, Refine-GAN, RingFormer, PCPH-GAN.<br/>
` RingFormer and PCPH-GAN architectures utilize MPD, MSD and MRD Discs combo.`
 
- Much better loss logging handling.
`( Per-epoch-avg loss as the main one, over-50-steps rolling avg as the long-term one )`
 
- More sophisticated dataset-preprocessing approach.
 
- Quick from-ui tweaks ` ( lr for g/d, lr schedulers, linear warmup, kl loss annealing and much more .. )`
 
- Various speed and performance improvements.

**Any new / experimental features are always described in releases so, feel free to check it out there.**
  
 
 
 <br/>
 
 
✨ to-do list ✨
> - Better long-term logging for pretrained / base models training.
> - Some additional feedback during training in terms of model's performance. 
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join my discord server ❗
 <br/>
 <br/>

## Getting Started:

### 1. Installation of the Fork

Run the installation script:

- Double-click `run-install.bat`.

### 2. Running the Fork

Start Applio using:

- Double-click `run-fork.bat`.
 
This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring
 
To monitor training or visualize data:
- Run the " run_tensorboard_in_model_folder.bat " file from logs folder and paste in there path to your model's folder </br>( containing 'eval' folder or tfevents file/s. )</br></br>If it doesn't work for you due to blocked port, open up CMD with admin rights and use this command:</br>`` netsh advfirewall firewall add rule name="Open Port 25565" dir=in action=allow protocol=TCP localport=25565 ``</br></br>
- Alternatively if the above method fails, run the tensorboard manually in cmd:</br> ``tensorboard --logdir="path/to/your/model/folder" --bind_all``</br>
(PS. Make sure you have tensorboard installed. ( in cmd:  pip install tensorboard )
 
## Referenced projects
+ [RingFormer](https://github.com/seongho608/RingFormer)
+ [RiFornet](https://github.com/Respaired/RiFornet_Vocoder)
+ [BigVGAN](https://github.com/NVIDIA/BigVGAN/tree/main)
+ [Pytorch-Snake](https://github.com/falkaer/pytorch-snake)
+ [wavehax](https://github.com/chomeyama/wavehax)

 
## Disclaimer
``The creators, maintainers, and contributors of the original Applio repository, as well as the creator of this fork (Codename;0), which is based on Applio, and the contributors of this fork, are not liable for any legal issues, damages, or consequences arising from the use of this repository or any content generated from it. By using this fork, you acknowledge and accept the following terms:``
 
- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md)
