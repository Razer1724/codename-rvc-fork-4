import torch
from torch.nn import functional as F

def phase_loss(x_fft: torch.Tensor, g_fft: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    x_norm = x_fft / (x_fft.abs() + 1e-9)
    g_norm = g_fft / (g_fft.abs() + 1e-9)

    phase_similarity = (x_norm * g_norm.conj()).real
    loss = 1.0 - phase_similarity

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return 2 * sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def feature_loss_mask(fmap_r, fmap_g, silence_mask=None, reduce=True):
    """
    Silence-aware feature matching loss.
    If silence_mask is provided, applies it per sample to reduce loss contribution.

    Args:
        fmap_r (List[List[Tensor]]): Feature maps from real audio
        fmap_g (List[List[Tensor]]): Feature maps from generated audio
        silence_mask (Tensor or None): Tensor of shape [B], 1 for voiced, 0 for silence
        reduce (bool): Whether to return mean or per-sample loss
    Returns:
        Scalar loss or per-sample tensor
    """
    losses = []

    for dr, dg in zip(fmap_r, fmap_g):  # across discriminators
        for rl, gl in zip(dr, dg):      # across layers
            diff = torch.abs(rl - gl)
            per_sample = diff.view(diff.shape[0], -1).mean(dim=1)  # [B]
            losses.append(per_sample)

    total = torch.stack(losses, dim=0).mean(dim=0)  # mean over layers → [B]

    if silence_mask is not None:
        total = total * silence_mask  # scale loss per sample

    if reduce:
        return total.sum() / (silence_mask.sum() + 1e-6 if silence_mask is not None else total.numel())
    else:
        return total  # shape [B]


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss # , r_losses, g_losses


def generator_loss(disc_outputs):
    """
    LSGAN Generator Loss:
    """
    loss = 0
    #gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss #, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()

    return loss

def kl_loss_clamped(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.
    Variant with non-negativity clamp.

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    loss = torch.clamp(loss, min=0.0)

    return loss


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()

        # Median centering
        m_DG = torch.median((dr - dg))

        # Relative difference
        # We only penalize when the Real sample is NOT sufficiently better than Fake
        # Condition: Real < Fake + Margin
        diff = (dr - dg) - m_DG
        mask = dr < (dg + m_DG)

        # Calculate Squared Error on the masked (hard) examples
        # We use empty-safe mean calculation
        masked_diff = diff[mask]
        if masked_diff.numel() > 0:
            L_rel = torch.mean(masked_diff ** 2)
        else:
            L_rel = torch.tensor(0.0, device=dr.device)

        # Truncate the loss (clamp)
        loss += tau - F.relu(tau - L_rel)

    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        
        # Median centering (Fake - Real)
        diff = dg - dr 
        m_DG = torch.median(diff)

        # Relative difference
        # Generator wants Fake > Real. 
        # We penalize when Fake is NOT sufficiently better than Real
        # Condition: Fake < Real + Margin
        rel_diff = diff - m_DG
        mask = diff < m_DG

        masked_diff = rel_diff[mask]
        if masked_diff.numel() > 0:
            L_rel = torch.mean(masked_diff ** 2)
        else:
            L_rel = torch.tensor(0.0, device=dg.device)

        # Truncate the loss (clamp)
        loss += tau - F.relu(tau - L_rel)

    return loss

def discriminator_loss_v2(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.
    """
    loss = 0

    # LSGAN Loss
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss

    # TPRLS Loss
    loss_rel = discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs)
    loss += loss_rel

    return loss


def generator_loss_v2(disc_outputs, disc_real_outputs):
    """
    LSGAN Generator Loss + TPRLS
    """
    loss = 0

    # Existing LSGAN Loss
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        loss += l

    # TPRLS Loss
    loss_rel = generator_TPRLS_loss(disc_real_outputs, disc_outputs)
    loss += loss_rel

    return loss

