import math
import torch
from typing import Optional, List
import random

from rvc.lib.algorithm import generators
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments

# Normalizing Flow
from rvc.lib.algorithm.normalizing_flow import ResidualCouplingBlock
# Text Encoder
from rvc.lib.algorithm.text_encoder import TextEncoder
# Posterior Encoder
from rvc.lib.algorithm.posterior_encoder import PosteriorEncoder


debug_shapes = False

class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        checkpointing: bool = False,
        # Other
        use_2_sample_kl: bool = False,
        # RingFormer
        gen_istft_n_fft: int = 120,
        gen_istft_hop_size: int = 30,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.vocoder = vocoder
        self.sr = sr
        self.use_2_sample_kl = use_2_sample_kl


        # ------   [ Decoder / Vocoder ] Reconstructs audio from latents (z)   ------------------------------------------------------
        vocoder_map = {
            "RefineGAN": generators.RefineGANGenerator,
            "RingFormer_v1": generators.RingFormerGenerator,
            "RingFormer_v2": generators.RingFormerGenerator,
            "APEX-GAN": generators.APEX_GAN_Generator,
            "HiFi-GAN": generators.HiFiGANNSFGenerator if use_f0 else generators.HiFiGANGenerator
        }

        dec_kwargs = {
            "initial_channel": inter_channels,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "sr": sr,
            "checkpointing": checkpointing,
        }

        if not use_f0 and vocoder != "HiFi-GAN":
            print(f"!!! Warning: {vocoder} does not support training without pitch guidance.")
            self.dec = None
        else:
            GeneratorClass = vocoder_map.get(vocoder, generators.HiFiGANNSFGenerator)

            if vocoder in ["RingFormer_v1", "RingFormer_v2"]: # RingFormer needs inverse stft params
                dec_kwargs.update({"gen_istft_n_fft": gen_istft_n_fft, "gen_istft_hop_size": gen_istft_hop_size})

            self.dec = GeneratorClass(**dec_kwargs)

            if use_f0 and vocoder == "HiFi-GAN":
                print("    ██████  Vocoder: NSF-HiFi-GAN")
            else:
                print(f"    ██████  Vocoder: {vocoder}")



        # ------   [ TextEncoder ] Maps extracted features to latent space (p)   ----------------------------------------------------
        self.enc_p = TextEncoder(
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            embedding_dim=text_enc_hidden_dim,
            f0=use_f0,
        )


        # ------   [ Posterior Encoder ] Extracts latents (z) from target audio (training only)   -----------------------------------
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            gin_channels=gin_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
        )


        # ------   [ Flow ] Reversible transformation between content priors (p) and speaker-conditioned latents (z)   --------------
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            n_flows=4,
            n_layers=3,
            kernel_size=5,
            dilation_rate=1,
            gin_channels=gin_channels,
        )


        # ------   [ Speaker Embedding ] Maps identity to global conditioning (g)   -------------------------------------------------
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)


    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None,
        spec_lengths: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Contentvec features.
            phone_lengths (torch.Tensor): Lengths of the contentvec features.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            spec (torch.Tensor, optional): Target spectrogram.
            spec_lengths (torch.Tensor, optional): Lengths of the target spectrograms ( linear specs ).
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)

        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        if spec is not None:
            # Posterior
            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
            # Flow
            z_p = self.flow(z, spec_mask, g=g)

            # 2nd sample for KL variance reduction
            z_p2 = None
            if self.use_2_sample_kl:
                z2 = (m_q + torch.randn_like(m_q) * torch.exp(logs_q)) * spec_mask
                z_p2 = self.flow(z2, spec_mask, g=g)

            # Slicing operations
            z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
            if self.use_f0:
                pitchf_slice = slice_segments(pitchf, ids_slice, self.segment_size, 2)


            # Decoder forward
            if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                o, spec, phase = self.dec(z_slice, pitchf_slice, g=g)
                return o, ids_slice, x_mask, spec_mask, (z, z_p, z_p2, m_p, logs_p, m_q, logs_q), (spec, phase)
            elif self.vocoder == "APEX-GAN":
                o = self.dec(z_slice, pitchf_slice, g=g)
                return o, ids_slice, x_mask, spec_mask, (z, z_p, z_p2, m_p, logs_p, m_q, logs_q)
            elif self.vocoder == "RefineGAN":
                o = self.dec(z_slice, pitchf_slice, g=g)
                return o, ids_slice, x_mask, spec_mask, (z, z_p, z_p2, m_p, logs_p, m_q, logs_q)
            else: # For HiFi-Gan
                if self.use_f0:
                    o = self.dec(z_slice, pitchf_slice, g=g)
                else:
                    o = self.dec(z_slice, g=g)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, z_p2, m_p, logs_p, m_q, logs_q)
        else:
            print(" NONE SPEC ")
            return None, None, x_mask, None, (None, None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        seed: int = 0,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Contentvec features.
            phone_lengths (torch.Tensor): Lengths of the contentvec features.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            seed (int, optional): Seed for randomization of noise.
            
        """

        # Seed handler
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Embedding
        g = self.emb_g(sid).unsqueeze(-1)

        # TextEncoder
        m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        # Flow
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decoder
        if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            o, _, _ = self.dec(z * x_mask, nsff0, g)
        elif self.vocoder == "APEX-GAN":
            o = (self.dec(z * x_mask, nsff0, g) if self.use_f0 else self.dec(z * x_mask, g=g))
        elif self.vocoder == "RefineGAN":
            o = (self.dec(z * x_mask, nsff0, g) if self.use_f0 else self.dec(z * x_mask, g=g))
        else: # HiFi-GAN
            o = (self.dec(z * x_mask, nsff0, g) if self.use_f0 else self.dec(z * x_mask, g=g))

        return o, x_mask, (z, z_p, m_p, logs_p)
