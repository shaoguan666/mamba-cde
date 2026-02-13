"""
Enhanced Spectral-FiLM Vector Field v3.0

Key changes from v2.0:
1. [FIX] Remove history buffer (incompatible with ODE adaptive step solvers)
2. [FIX] FFT on feature dimension (stateless, clean gradients)
3. [NEW] Time-conditioned spectral weights: freq_weights = sigmoid(MLP(time_enc))
4. [NEW] FiLM modulation on ALL hidden layers
5. [FIX] Softer adaptive band boundaries (scale=2.0 vs 10.0)
6. [KEEP] Dynamic fusion gate
7. [KEEP] Adaptive frequency band boundaries
8. [FIX] Learnable TimeEncoder (trainable frequencies)
"""

import torch
import torch.nn as nn
import math


class TimeEncoder(nn.Module):
    """Learnable continuous time encoding with trainable frequencies.

    Implements: [sin(w_1*t), cos(w_1*t), ..., sin(w_d*t), cos(w_d*t)]
    where w is learnable, initialized with log-linear spacing.
    """
    def __init__(self, time_dim):
        super().__init__()
        assert time_dim % 2 == 0
        self.time_dim = time_dim
        half_dim = time_dim // 2
        init_freq = torch.exp(torch.linspace(0, -math.log(1000), half_dim))
        self.w = nn.Parameter(init_freq)

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            if t.dim() == 2:
                t = t.squeeze(-1)
        angles = t.unsqueeze(-1) * self.w.unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if squeeze:
            encoding = encoding.squeeze(0)
        return encoding


class DynamicFusionGate(nn.Module):
    """Adaptive fusion gate: alpha = sigmoid(MLP(concat(time_enc, z)))

    Dynamically adjusts time vs spectral branch importance
    based on current temporal context and hidden state.
    """
    def __init__(self, time_dim, hidden_channels, gate_hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(time_dim + hidden_channels, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.ReLU(),
            nn.Linear(gate_hidden // 2, 1),
            nn.Sigmoid()
        )
        # Initialize to ~0.5 (balanced fusion)
        with torch.no_grad():
            self.gate[-2].bias.fill_(0.0)

    def forward(self, time_enc, z):
        """
        Args:
            time_enc: [batch, time_dim]
            z: [batch, hidden_channels]
        Returns:
            alpha: [batch, 1]
        """
        if z.dim() > 2:
            z_pooled = z.mean(dim=list(range(1, z.dim() - 1)))
        else:
            z_pooled = z
        gate_input = torch.cat([time_enc, z_pooled], dim=-1)
        return self.gate(gate_input)


class MultiScaleSpectralBranch(nn.Module):
    """Feature-dimension FFT with time-conditioned weights and adaptive bands.

    Key design:
        - FFT on feature dimension (stateless, no history buffer)
        - Frequency weights conditioned on time via MLP
        - Soft adaptive band boundaries (scale=2.0)
        - Per-band projection with learnable importance

    Architecture:
        z -> rfft(dim=-1) -> time_conditioned_filter -> adaptive_bands -> per_band_proj -> fuse
    """
    def __init__(self, hidden_channels, hidden_hidden_channels, time_dim,
                 spectral_sigma=2.0, num_bands=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_bands = num_bands
        self.num_freqs = hidden_channels // 2 + 1

        # === Time-conditioned frequency weights ===
        # MLP: time_enc -> per-frequency importance in [0, 1]
        self.time_to_freq = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, self.num_freqs),
        )
        # Initialize bias to logit of Gaussian low-pass filter
        # so sigmoid(bias) ~= exp(-k^2/(2*sigma^2))
        # Weight init to near-zero for gentle start from low-pass behavior
        with torch.no_grad():
            k = torch.arange(self.num_freqs, dtype=torch.float32)
            lowpass = torch.exp(-k ** 2 / (2 * spectral_sigma ** 2))
            lowpass = lowpass.clamp(min=1e-4, max=1 - 1e-4)
            logit_init = torch.log(lowpass / (1 - lowpass))
            self.time_to_freq[-1].bias.copy_(logit_init)
            nn.init.zeros_(self.time_to_freq[-1].weight)
            nn.init.normal_(self.time_to_freq[0].weight, std=0.01)
            nn.init.zeros_(self.time_to_freq[0].bias)

        # === Adaptive frequency band boundaries ===
        # Two learnable boundaries split [0, num_freqs] into 3 bands
        # sigmoid maps logits to [0,1], then scale to [0, num_freqs]
        # Initialized at ~0.2 and ~0.6 of range
        initial_boundaries = torch.tensor([0.2, 0.6])
        self.band_boundaries_logits = nn.Parameter(initial_boundaries)

        # Per-band importance weights
        self.band_weights = nn.Parameter(torch.ones(num_bands))

        # === Per-band projections ===
        band_dim = hidden_hidden_channels // num_bands
        band_dim_last = hidden_hidden_channels - band_dim * (num_bands - 1)
        self.band_projections = nn.ModuleList([
            nn.Linear(hidden_channels,
                      band_dim if i < num_bands - 1 else band_dim_last)
            for i in range(num_bands)
        ])

        # Final fusion
        self.final_proj = nn.Linear(hidden_hidden_channels, hidden_hidden_channels)

        # Initialization
        for proj in self.band_projections:
            nn.init.normal_(proj.weight, std=0.01)
            nn.init.zeros_(proj.bias)
        nn.init.normal_(self.final_proj.weight, std=0.01)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, z, time_enc):
        """
        Args:
            z: [batch, hidden_channels]
            time_enc: [batch, time_dim]
        Returns:
            [batch, hidden_hidden_channels]
        """
        # FFT on feature dimension (stateless, gradient flows through z)
        z_freq = torch.fft.rfft(z, dim=-1)  # [batch, num_freqs] complex

        # Time-conditioned frequency weights
        freq_weights = torch.sigmoid(self.time_to_freq(time_enc))  # [batch, num_freqs]
        z_freq_filtered = z_freq * freq_weights

        # Adaptive band boundaries
        boundaries_norm = torch.sigmoid(self.band_boundaries_logits)
        boundaries = boundaries_norm * self.num_freqs
        boundaries_sorted, _ = torch.sort(boundaries)
        zeros = torch.zeros(1, device=z.device, dtype=boundaries.dtype)
        end = torch.full((1,), self.num_freqs, device=z.device, dtype=boundaries.dtype)
        boundary_indices = torch.cat([zeros, boundaries_sorted, end])

        freq_idx = torch.arange(self.num_freqs, device=z.device,
                                dtype=boundary_indices.dtype)

        band_features = []
        for i in range(self.num_bands):
            start_val = boundary_indices[i]
            end_val = boundary_indices[i + 1]
            # Soft band mask (scale=2.0 for smooth gradients)
            soft_mask = (torch.sigmoid(2.0 * (freq_idx - start_val))
                         * torch.sigmoid(2.0 * (end_val - freq_idx)))

            band_freq = z_freq_filtered * soft_mask.unsqueeze(0)
            band_spatial = torch.fft.irfft(band_freq, n=self.hidden_channels,
                                           dim=-1)  # [batch, hidden_channels]

            band_feat = self.band_projections[i](band_spatial)
            band_feat = band_feat * self.band_weights[i]
            band_features.append(band_feat)

        multiscale = torch.cat(band_features, dim=-1)
        return self.final_proj(multiscale)


class EnhancedSpectralModulatedVectorField(nn.Module):
    """Enhanced Spectral-FiLM Vector Field v3.0.

    Architecture:
        Time Branch: Multi-layer FiLM-modulated MLP (FiLM on ALL layers)
        Spectral Branch: Feature-dim FFT + Time-conditioned weights + Adaptive bands
        Dynamic Fusion: alpha * time_branch + (1-alpha) * spectral_branch
        Output: tanh-bounded vector field

    Key properties:
        - Fully stateless: no history buffer, compatible with any ODE solver
        - Clean gradient flow: no detach() in spectral path
        - Time-aware spectral filtering: frequency weights depend on t
        - Deep FiLM: time modulation at every hidden layer
    """

    def __init__(self, input_channels, hidden_channels, time_dim=32,
                 spectral_sigma=2.0, hidden_hidden_channels=None,
                 num_hidden_layers=None, num_bands=3, **kwargs):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.spectral_sigma = spectral_sigma
        self.num_bands = num_bands

        self.hidden_hidden_channels = (hidden_hidden_channels
                                       if hidden_hidden_channels is not None
                                       else 128)
        self.num_hidden_layers = (num_hidden_layers
                                  if num_hidden_layers is not None
                                  else 1)

        # === Time encoder (learnable frequencies) ===
        self.time_encoder = TimeEncoder(time_dim)

        # === Time branch: MLP layers ===
        self.linear_in = nn.Linear(hidden_channels, self.hidden_hidden_channels)
        self.linears = nn.ModuleList([
            nn.Linear(self.hidden_hidden_channels, self.hidden_hidden_channels)
            for _ in range(self.num_hidden_layers - 1)
        ])

        # === FiLM generators for ALL hidden layers ===
        self.film_generators = nn.ModuleList([
            nn.Linear(time_dim, 2 * self.hidden_hidden_channels)
            for _ in range(self.num_hidden_layers)
        ])
        for fg in self.film_generators:
            nn.init.zeros_(fg.weight)
            nn.init.zeros_(fg.bias)

        # === Spectral branch (time-conditioned, feature-dim FFT) ===
        self.spectral_branch = MultiScaleSpectralBranch(
            hidden_channels, self.hidden_hidden_channels, time_dim,
            spectral_sigma=spectral_sigma, num_bands=num_bands
        )

        # === Dynamic fusion gate ===
        self.fusion_gate = DynamicFusionGate(
            time_dim, hidden_channels, gate_hidden=64
        )

        # === Output projection ===
        self.linear_out = nn.Linear(
            self.hidden_hidden_channels, input_channels * hidden_channels
        )
        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        nn.init.zeros_(self.linear_out.bias)

        # === Logging ===
        self.logging_enabled = False
        self.logs = []

    def extra_repr(self):
        return ("input_channels: {}, hidden_channels: {}, time_dim: {}, "
                "spectral_sigma: {:.2f}, hidden_hidden_channels: {}, "
                "num_hidden_layers: {}, num_bands: {}").format(
                    self.input_channels, self.hidden_channels, self.time_dim,
                    self.spectral_sigma, self.hidden_hidden_channels,
                    self.num_hidden_layers, self.num_bands)

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
            }

        return {
            'time': np.array([log['time'] for log in self.logs]),
            'gamma': np.stack([log['gamma'] for log in self.logs], axis=0),
            'beta': np.stack([log['beta'] for log in self.logs], axis=0),
            'fusion_alpha': np.array([log['fusion_alpha'] for log in self.logs]),
            'time_branch_norm': np.array([log['time_branch_norm'] for log in self.logs]),
            'spectral_branch_norm': np.array([log['spectral_branch_norm'] for log in self.logs]),
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

        # === Generate FiLM params for all layers ===
        film_params_all = [fg(time_enc) for fg in self.film_generators]
        # Each: [batch, 2 * hidden_hidden_channels]

        # Store first-layer params for logging (before broadcast)
        gamma_0_for_log = film_params_all[0][..., :self.hidden_hidden_channels]
        beta_0_for_log = film_params_all[0][..., self.hidden_hidden_channels:]

        # Broadcast FiLM params for multi-dim z
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            broadcast = []
            for fp in film_params_all:
                for _ in range(extra_dims):
                    fp = fp.unsqueeze(1)
                broadcast.append(fp)
            film_params_all = broadcast

        # === Time branch: FiLM on ALL layers ===
        # Layer 0: linear_in + FiLM_0
        gamma_0 = film_params_all[0][..., :self.hidden_hidden_channels]
        beta_0 = film_params_all[0][..., self.hidden_hidden_channels:]
        h_time = self.linear_in(z)
        h_time = (1 + gamma_0) * h_time + beta_0
        h_time = torch.relu(h_time)

        # Layers 1..N-1: linears[i] + FiLM_{i+1}
        for i, linear in enumerate(self.linears):
            gamma_i = film_params_all[i + 1][..., :self.hidden_hidden_channels]
            beta_i = film_params_all[i + 1][..., self.hidden_hidden_channels:]
            h_time = linear(h_time)
            h_time = (1 + gamma_i) * h_time + beta_i
            h_time = torch.relu(h_time)

        # === Spectral branch (time-conditioned) ===
        # Flatten multi-dim z for spectral processing
        z_flat = z.reshape(-1, self.hidden_channels) if z.dim() > 2 else z
        time_enc_flat = (time_enc.reshape(-1, self.time_dim)
                         if time_enc.dim() > 2 else time_enc)

        h_freq = self.spectral_branch(z_flat, time_enc_flat)

        # Restore shape if needed
        if z.dim() > 2:
            h_freq = h_freq.reshape(*z.shape[:-1],
                                    self.hidden_hidden_channels)

        # === Dynamic fusion ===
        alpha = self.fusion_gate(time_enc_flat, z_flat)  # [batch, 1]

        if z.dim() > 2:
            alpha = alpha.reshape(*z.shape[:-1], 1)

        h_fused = alpha * h_time + (1 - alpha) * h_freq

        # === Output ===
        out = self.linear_out(h_fused)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        out = torch.tanh(out)

        # === Logging ===
        if self.logging_enabled:
            gamma_log = gamma_0_for_log.mean(dim=0).detach().cpu().numpy()
            beta_log = beta_0_for_log.mean(dim=0).detach().cpu().numpy()
            alpha_log = alpha.mean().item()
            time_norm = torch.norm(h_time, p=2, dim=-1).mean().item()
            spectral_norm = torch.norm(h_freq, p=2, dim=-1).mean().item()
            time_log = t.item() if t.dim() == 0 else t[0].item()

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'fusion_alpha': alpha_log,
                'time_branch_norm': time_norm,
                'spectral_branch_norm': spectral_norm,
            })

        return out
