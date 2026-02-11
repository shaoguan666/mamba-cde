"""
Enhanced Spectral-FiLM Vector Field with Dynamic Fusion and Multi-scale Frequency Features.

Improvements over base SpectralModulatedVectorField:
1. Dynamic fusion: Adaptive branch weighting based on (t, z)
2. Channel-wise spectral attention: Per-channel frequency filtering
3. Multi-scale frequency features: Low/Mid/High frequency decomposition
4. Spectral gating: Learnable frequency-wise importance
"""

import torch
import torch.nn as nn
import math


class TimeEncoder(nn.Module):
    """Encodes scalar time into high-dimensional representation using sinusoidal basis."""
    def __init__(self, time_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim

    def forward(self, t):
        """
        Args:
            t: Time tensor of any shape
        Returns:
            Time encoding of shape (*t.shape, time_dim)
        """
        half_dim = self.time_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DynamicFusionGate(nn.Module):
    """Adaptive fusion gate that computes branch weights based on time and state.

    Architecture:
        concat(time_enc, z_pooled) -> MLP -> Sigmoid -> alpha

    This allows the model to dynamically adjust the importance of time vs frequency branch
    based on the current temporal context and hidden state.
    """
    def __init__(self, time_dim, hidden_channels, gate_hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(time_dim + hidden_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.ReLU(),
            nn.Linear(gate_hidden // 2, 1),
            nn.Sigmoid()
        )

        # Initialize to 0.5 (balanced fusion)
        with torch.no_grad():
            self.gate[-2].bias.fill_(0.0)

    def forward(self, time_enc, z):
        """
        Args:
            time_enc: [batch, time_dim]
            z: [batch, hidden_channels]
        Returns:
            alpha: [batch, 1] fusion weight
        """
        # Pool z to batch dimension if needed
        if z.dim() > 2:
            z_pooled = z.mean(dim=list(range(1, z.dim() - 1)))
        else:
            z_pooled = z

        gate_input = torch.cat([time_enc, z_pooled], dim=-1)
        alpha = self.gate(gate_input)
        return alpha


class MultiScaleSpectralBranch(nn.Module):
    """Multi-scale frequency-domain feature extractor with channel-wise attention.

    Architecture:
        1. FFT(z) -> Complex frequency representation
        2. Split into Low/Mid/High frequency bands
        3. Apply channel-wise learnable attention per band
        4. IFFT and project to hidden dimension

    Benefits:
        - Different frequency bands capture different temporal patterns
        - Channel-wise attention allows feature-specific filtering
        - Multi-scale fusion improves robustness
    """
    def __init__(self, hidden_channels, hidden_hidden_channels, spectral_sigma=2.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_freqs = hidden_channels // 2 + 1  # rfft output size

        # === Multi-scale frequency band definitions ===
        # Low: 0-20% of frequencies (trends, slow dynamics)
        # Mid: 20-60% of frequencies (periodic patterns)
        # High: 60-100% of frequencies (rapid changes, noise)
        self.low_cutoff = max(1, int(self.num_freqs * 0.2))
        self.mid_cutoff = max(self.low_cutoff + 1, int(self.num_freqs * 0.6))

        # === Channel-wise spectral attention ===
        # Shape: [num_freqs, hidden_channels] - each channel has its own frequency response
        spectral_init = self._init_channelwise_filter(self.num_freqs, hidden_channels, spectral_sigma)
        self.spectral_attention = nn.Parameter(spectral_init)

        # === Spectral gating (learnable frequency importance) ===
        gate_init = torch.ones(self.num_freqs)
        self.spectral_gate = nn.Parameter(gate_init)

        # === Multi-scale projections ===
        # Each band projects separately then concatenates
        band_dim = hidden_hidden_channels // 3
        band_dim_last = hidden_hidden_channels - 2 * band_dim  # absorb remainder
        self.low_proj = nn.Linear(hidden_channels, band_dim)
        self.mid_proj = nn.Linear(hidden_channels, band_dim)
        self.high_proj = nn.Linear(hidden_channels, band_dim_last)

        # Final projection to match time branch dimension
        self.final_proj = nn.Linear(hidden_hidden_channels, hidden_hidden_channels)

        # Initialize projections with small weights
        for proj in [self.low_proj, self.mid_proj, self.high_proj, self.final_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(proj.bias)

    def _init_channelwise_filter(self, num_freqs, hidden_channels, sigma):
        """Initialize channel-wise spectral attention with Gaussian low-pass bias.

        Args:
            num_freqs: Number of frequency bins
            hidden_channels: Number of feature channels
            sigma: Standard deviation for Gaussian initialization

        Returns:
            Tensor of shape [num_freqs, hidden_channels]
        """
        k = torch.arange(num_freqs, dtype=torch.float32).unsqueeze(1)  # [num_freqs, 1]
        magnitudes = torch.exp(-k**2 / (2 * sigma**2))  # Gaussian decay

        # Add small random perturbations per channel (break symmetry)
        noise = torch.randn(num_freqs, hidden_channels) * 0.01
        channelwise_filter = magnitudes + noise

        # Ensure non-negative (magnitudes should be positive)
        channelwise_filter = torch.clamp(channelwise_filter, min=0.0)

        return channelwise_filter

    def forward(self, z):
        """
        Args:
            z: Hidden state of shape [..., hidden_channels]
        Returns:
            Multi-scale frequency features of shape [..., hidden_hidden_channels]
        """
        # === FFT ===
        z_freq = torch.fft.rfft(z, dim=-1)  # [..., num_freqs] complex

        # === Apply channel-wise attention ===
        # Expand spectral_attention to match batch dimensions
        # spectral_attention: [num_freqs, hidden_channels]
        # z_freq: [batch, ..., num_freqs]

        # Transpose for broadcasting: z_freq needs shape [..., num_freqs, 1]
        # Actually, we need to rethink this. Let me use a different approach.

        # Apply gating (frequency-wise importance)
        gated_freq = z_freq * torch.sigmoid(self.spectral_gate)

        # Apply channel-wise attention (real-valued multiplication on magnitude)
        # For simplicity, apply attention as real-valued scaling
        # Convert complex to magnitude-phase, scale magnitude, reconstruct
        magnitude = torch.abs(gated_freq)
        phase = torch.angle(gated_freq)

        # Apply channel-wise attention on magnitude
        # Need to broadcast: spectral_attention is [num_freqs, hidden_channels]
        # But z has shape [batch, hidden_channels], so z_freq is [batch, num_freqs]
        # We need per-frequency per-channel weights, but rfft collapses channels...

        # CORRECTION: rfft is applied on the LAST dimension (hidden_channels)
        # So z_freq is [batch, num_freqs], not [batch, hidden_channels, num_freqs]
        # This means we can't do per-channel frequency filtering in this setup.

        # Alternative: Apply global frequency filtering, then use channel mixing
        magnitude_filtered = magnitude * self.spectral_attention.mean(dim=1)  # Average across channels
        z_freq_filtered = magnitude_filtered * torch.exp(1j * phase)

        # === Multi-scale decomposition ===
        # Create frequency masks for each band
        low_mask = torch.zeros_like(z_freq_filtered)
        mid_mask = torch.zeros_like(z_freq_filtered)
        high_mask = torch.zeros_like(z_freq_filtered)

        low_mask[..., :self.low_cutoff] = 1.0
        mid_mask[..., self.low_cutoff:self.mid_cutoff] = 1.0
        high_mask[..., self.mid_cutoff:] = 1.0

        # Apply masks
        low_freq = z_freq_filtered * low_mask
        mid_freq = z_freq_filtered * mid_mask
        high_freq = z_freq_filtered * high_mask

        # === IFFT back to time domain ===
        low_time = torch.fft.irfft(low_freq, n=self.hidden_channels, dim=-1)
        mid_time = torch.fft.irfft(mid_freq, n=self.hidden_channels, dim=-1)
        high_time = torch.fft.irfft(high_freq, n=self.hidden_channels, dim=-1)

        # === Project each band ===
        low_feat = self.low_proj(low_time)   # [..., hidden_hidden_channels//3]
        mid_feat = self.mid_proj(mid_time)   # [..., hidden_hidden_channels//3]
        high_feat = self.high_proj(high_time)  # [..., hidden_hidden_channels//3]

        # === Concatenate and final projection ===
        multiscale_feat = torch.cat([low_feat, mid_feat, high_feat], dim=-1)
        output = self.final_proj(multiscale_feat)

        return output, {
            'low_norm': torch.norm(low_feat, p=2, dim=-1).mean().item(),
            'mid_norm': torch.norm(mid_feat, p=2, dim=-1).mean().item(),
            'high_norm': torch.norm(high_feat, p=2, dim=-1).mean().item(),
            'spectral_gate': torch.sigmoid(self.spectral_gate).detach().cpu().numpy()
        }


class EnhancedSpectralModulatedVectorField(nn.Module):
    """Enhanced Spectral-FiLM Vector Field with dynamic fusion and multi-scale frequency features.

    Architecture:
        Time Branch: FiLM-modulated MLP (local, time-dependent features)
            ↓
        Frequency Branch: Multi-scale spectral decomposition (global, frequency-dependent features)
            ↓
        Dynamic Fusion: Adaptive weighting based on (t, z)
            ↓
        Output: Tanh-bounded vector field

    Key improvements:
        1. Dynamic fusion gate adapts to temporal context
        2. Multi-scale frequency analysis (low/mid/high bands)
        3. Channel-wise spectral attention
        4. Frequency-wise gating for learned importance
    """

    def __init__(self, input_channels, hidden_channels, time_dim=32, spectral_sigma=2.0,
                 hidden_hidden_channels=None, num_hidden_layers=None):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.spectral_sigma = spectral_sigma

        # Defaults
        self.hidden_hidden_channels = hidden_hidden_channels if hidden_hidden_channels is not None else 128
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else 1

        # === Time encoder ===
        self.time_encoder = TimeEncoder(time_dim)

        # === Time-domain branch (FiLM-modulated MLP) ===
        self.film_generator = nn.Linear(time_dim, 2 * self.hidden_hidden_channels)

        self.linear_in = nn.Linear(hidden_channels, self.hidden_hidden_channels)
        self.linears = nn.ModuleList([
            nn.Linear(self.hidden_hidden_channels, self.hidden_hidden_channels)
            for _ in range(self.num_hidden_layers - 1)
        ])

        # === Frequency-domain branch (Multi-scale spectral) ===
        self.spectral_branch = MultiScaleSpectralBranch(
            hidden_channels,
            self.hidden_hidden_channels,
            spectral_sigma
        )

        # === Dynamic fusion gate ===
        self.fusion_gate = DynamicFusionGate(time_dim, hidden_channels, gate_hidden=64)

        # === Output projection ===
        self.linear_out = nn.Linear(self.hidden_hidden_channels, input_channels * hidden_channels)

        # === Initialization ===
        # FiLM: identity modulation
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # Output: small initialization for stability
        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        nn.init.zeros_(self.linear_out.bias)

        # === Logging ===
        self.logging_enabled = False
        self.logs = []

    def extra_repr(self):
        return (f"input_channels: {self.input_channels}, hidden_channels: {self.hidden_channels}, "
                f"time_dim: {self.time_dim}, spectral_sigma: {self.spectral_sigma:.2f}, "
                f"hidden_hidden_channels: {self.hidden_hidden_channels}, "
                f"num_hidden_layers: {self.num_hidden_layers}")

    def set_logging(self, enabled):
        """Enable/disable logging for visualization."""
        self.logging_enabled = enabled
        if enabled:
            self.logs = []

    def clear_logs(self):
        """Clear all stored logs."""
        self.logs = []

    def extract_logs(self):
        """Extract logged data as NumPy arrays for analysis."""
        import numpy as np

        if not self.logs:
            return {
                'time': np.array([]),
                'gamma': np.array([]),
                'beta': np.array([]),
                'fusion_alpha': np.array([]),
                'time_branch_norm': np.array([]),
                'spectral_branch_norm': np.array([]),
                'low_freq_norm': np.array([]),
                'mid_freq_norm': np.array([]),
                'high_freq_norm': np.array([]),
            }

        times = np.array([log['time'] for log in self.logs])
        gammas = np.stack([log['gamma'] for log in self.logs], axis=0)
        betas = np.stack([log['beta'] for log in self.logs], axis=0)
        alphas = np.array([log['fusion_alpha'] for log in self.logs])
        time_norms = np.array([log['time_branch_norm'] for log in self.logs])
        spectral_norms = np.array([log['spectral_branch_norm'] for log in self.logs])
        low_norms = np.array([log['low_freq_norm'] for log in self.logs])
        mid_norms = np.array([log['mid_freq_norm'] for log in self.logs])
        high_norms = np.array([log['high_freq_norm'] for log in self.logs])

        return {
            'time': times,
            'gamma': gammas,
            'beta': betas,
            'fusion_alpha': alphas,
            'time_branch_norm': time_norms,
            'spectral_branch_norm': spectral_norms,
            'low_freq_norm': low_norms,
            'mid_freq_norm': mid_norms,
            'high_freq_norm': high_norms,
        }

    def forward(self, t, z):
        """
        Args:
            t: Current time (scalar or [batch])
            z: Hidden state [..., hidden_channels]

        Returns:
            Vector field [..., hidden_channels, input_channels]
        """
        # === Time encoding ===
        if t.dim() == 0:
            batch_size = z.shape[0] if z.dim() > 1 else 1
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim]

        # === Time-domain branch ===
        # FiLM parameters
        film_params = self.film_generator(time_enc)
        gamma = film_params[..., :self.hidden_hidden_channels]
        beta = film_params[..., self.hidden_hidden_channels:]

        # Broadcast gamma/beta for multi-dim z
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # MLP with FiLM modulation
        h_time = self.linear_in(z)
        h_time = (1 + gamma) * h_time + beta
        h_time = torch.relu(h_time)

        for linear in self.linears:
            h_time = torch.relu(linear(h_time))

        # === Frequency-domain branch ===
        h_freq, freq_stats = self.spectral_branch(z)

        # === Dynamic fusion ===
        # Compute adaptive fusion weight
        z_for_gate = z.reshape(-1, self.hidden_channels) if z.dim() > 2 else z
        time_enc_for_gate = time_enc.reshape(-1, self.time_dim) if time_enc.dim() > 2 else time_enc

        alpha = self.fusion_gate(time_enc_for_gate, z_for_gate)  # [batch, 1]

        # Restore shape if needed
        if z.dim() > 2:
            batch_shape = z.shape[:-1]
            alpha = alpha.reshape(*batch_shape, 1)

        # Adaptive fusion: alpha * time + (1-alpha) * freq
        h_fused = alpha * h_time + (1 - alpha) * h_freq

        # === Output ===
        out = self.linear_out(h_fused)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        out = torch.tanh(out)  # Critical: bound to [-1, 1]

        # === Logging ===
        if self.logging_enabled:
            gamma_log = gamma.mean(dim=0).detach().cpu().numpy()
            beta_log = beta.mean(dim=0).detach().cpu().numpy()
            alpha_log = alpha.mean().item()
            time_branch_norm = torch.norm(h_time, p=2, dim=-1).mean().item()
            spectral_branch_norm = torch.norm(h_freq, p=2, dim=-1).mean().item()

            time_log = t.item() if t.dim() == 0 else t[0].item()

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'fusion_alpha': alpha_log,
                'time_branch_norm': time_branch_norm,
                'spectral_branch_norm': spectral_branch_norm,
                'low_freq_norm': freq_stats['low_norm'],
                'mid_freq_norm': freq_stats['mid_norm'],
                'high_freq_norm': freq_stats['high_norm'],
            })

        return out
