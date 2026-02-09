import torch
import math

from . import metamodel


class TimeEncoder(torch.nn.Module):
    """Learnable continuous time encoding similar to Transformer positional encoding.

    Implements: Phi(t) = [sin(w_1 * t), cos(w_1 * t), ..., sin(w_d * t), cos(w_d * t)]
    where w is a learnable frequency parameter.
    """
    def __init__(self, time_dim):
        """
        Arguments:
            time_dim: Dimension of time encoding output. Must be even (half sin, half cos).
        """
        super(TimeEncoder, self).__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        self.time_dim = time_dim
        half_dim = time_dim // 2

        # Learnable frequency parameters, initialized with log-linear spacing
        # Similar to Transformer positional encoding initialization
        init_freq = torch.exp(torch.linspace(0, -math.log(1000), half_dim))
        self.w = torch.nn.Parameter(init_freq)

    def forward(self, t):
        """
        Arguments:
            t: Time value(s). Can be scalar, [batch], or [batch, 1].

        Returns:
            Time encoding of shape [batch, time_dim] or [time_dim] if scalar input.
        """
        # Handle different input shapes
        if t.dim() == 0:
            # Scalar case: t is 0-dimensional
            t = t.unsqueeze(0)  # [1]
            squeeze_output = True
        else:
            squeeze_output = False
            if t.dim() == 2:
                t = t.squeeze(-1)  # [batch, 1] -> [batch]

        # t: [batch], w: [half_dim]
        # Compute t * w for all combinations: [batch, half_dim]
        angles = t.unsqueeze(-1) * self.w.unsqueeze(0)  # [batch, half_dim]

        # Concatenate sin and cos: [batch, time_dim]
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if squeeze_output:
            encoding = encoding.squeeze(0)

        return encoding


class ModulatedSingleHiddenLayer(torch.nn.Module):
    """Time-modulated vector field using FiLM (Feature-wise Linear Modulation).

    Implements: f(t, z) where time modulates the hidden features via:
        h_raw = Linear(z)
        gamma(t), beta(t) = Linear(Phi(t))
        h_mod = (1 + gamma(t)) * h_raw + beta(t)
        output = Linear(ReLU(h_mod))

    Now supports multi-layer architecture similar to FinalTanh.
    """
    def __init__(self, input_channels, hidden_channels, time_dim=32,
                 hidden_hidden_channels=None, num_hidden_layers=None):
        """
        Arguments:
            input_channels: Number of input channels (from control signal X).
            hidden_channels: Number of hidden channels (state z dimension).
            time_dim: Dimension of time encoding (default 32).
            hidden_hidden_channels: Hidden layer dimension (default 128 if None).
            num_hidden_layers: Number of hidden layers (default 1 if None).
        """
        super(ModulatedSingleHiddenLayer, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim

        # Set defaults if not provided (for backward compatibility)
        self.hidden_hidden_channels = hidden_hidden_channels if hidden_hidden_channels is not None else 128
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else 1

        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # FiLM parameter generator: time_dim -> 2 * hidden_hidden_channels (gamma and beta)
        self.film_generator = torch.nn.Linear(time_dim, 2 * self.hidden_hidden_channels)

        # Multi-layer network architecture (similar to FinalTanh)
        self.linear_in = torch.nn.Linear(hidden_channels, self.hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(self.hidden_hidden_channels, self.hidden_hidden_channels)
            for _ in range(self.num_hidden_layers - 1)
        )
        self.linear_out = torch.nn.Linear(self.hidden_hidden_channels, input_channels * hidden_channels)

        # Initialize FiLM generator to identity modulation at start
        # gamma = 0, beta = 0 means h_mod = (1 + 0) * h_raw + 0 = h_raw
        torch.nn.init.zeros_(self.film_generator.weight)
        torch.nn.init.zeros_(self.film_generator.bias)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, time_dim: {}, hidden_hidden_channels: {}, num_hidden_layers: {}".format(
            self.input_channels, self.hidden_channels, self.time_dim,
            self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t, z):
        """
        Arguments:
            t: Current time (scalar or [batch]).
            z: Hidden state of shape (..., hidden_channels).

        Returns:
            Vector field of shape (..., hidden_channels, input_channels).
        """
        # Get time encoding
        # Handle broadcasting: if t is scalar but z is batched
        if t.dim() == 0:
            batch_size = z.shape[0] if z.dim() > 1 else 1
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim]

        # Generate FiLM parameters
        film_params = self.film_generator(time_enc)  # [batch, 2 * hidden_hidden_channels]
        gamma = film_params[..., :self.hidden_hidden_channels]  # [batch, hidden_hidden_channels]
        beta = film_params[..., self.hidden_hidden_channels:]   # [batch, hidden_hidden_channels]

        # Broadcast gamma and beta to match z's batch dimensions
        # z shape: (..., hidden_channels), we need to align gamma/beta
        if z.dim() > 2:
            # z has extra batch dimensions, need to reshape gamma/beta
            # Assuming first dim of z is batch
            extra_dims = z.dim() - 2  # number of extra dimensions between batch and channels
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # Multi-layer feature extraction with FiLM modulation
        h = self.linear_in(z)  # (..., hidden_hidden_channels)

        # FiLM modulation: (1 + gamma) * h + beta
        h = (1 + gamma) * h + beta
        h = torch.relu(h)

        # Additional hidden layers
        for linear in self.linears:
            h = linear(h)
            h = torch.relu(h)

        # Output layer
        out = self.linear_out(h)

        # Reshape to match expected output: (..., hidden_channels, input_channels)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

        return out


class SpectralModulatedVectorField(torch.nn.Module):
    """Spectral-FiLM hybrid vector field combining time-domain modulation and frequency-domain filtering.

    Architecture:
        - Time-domain branch: FiLM-modulated MLP (handles local, abrupt medical events)
        - Frequency-domain branch: FFT-based global feature mixing (filters noise, captures trends)

    Implements:
        h_time = ReLU((1 + gamma(t)) * Linear(z) + beta(t))  [FiLM branch]
        h_freq = IFFT(W_spectral * FFT(z))                    [Spectral branch]
        h_fused = h_time + alpha * h_freq                     [Fusion]
        output = Linear(h_fused)

    Benefits over pure attention:
        - Complexity: O(D log D) vs O(N^2), where D is feature dimension
        - Stability: Frequency domain is more stable for ODE integration
        - Noise robustness: Low-pass filtering naturally handles sensor noise in medical data
    """

    def __init__(self, input_channels, hidden_channels, time_dim=32, spectral_sigma=2.0,
                 hidden_hidden_channels=None, num_hidden_layers=None):
        """
        Arguments:
            input_channels: Number of input channels (from control signal X).
            hidden_channels: Number of hidden channels (state z dimension).
            time_dim: Dimension of time encoding (default 32).
            spectral_sigma: Controls low-pass filter bandwidth. Smaller = more aggressive filtering.
                           Default 2.0 preserves ~20% of frequencies.
            hidden_hidden_channels: Hidden layer dimension (default 128 if None).
            num_hidden_layers: Number of hidden layers (default 1 if None).
        """
        super(SpectralModulatedVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.spectral_sigma = spectral_sigma

        # Set defaults if not provided (for backward compatibility)
        self.hidden_hidden_channels = hidden_hidden_channels if hidden_hidden_channels is not None else 128
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else 1

        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # FiLM parameter generator: time_dim -> 2 * hidden_hidden_channels (gamma and beta)
        self.film_generator = torch.nn.Linear(time_dim, 2 * self.hidden_hidden_channels)

        # Multi-layer time-domain branch architecture
        self.linear_in = torch.nn.Linear(hidden_channels, self.hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(self.hidden_hidden_channels, self.hidden_hidden_channels)
            for _ in range(self.num_hidden_layers - 1)
        )
        self.linear_out = torch.nn.Linear(self.hidden_hidden_channels, input_channels * hidden_channels)

        # Frequency-domain branch
        # Number of frequency bins for rfft: (hidden_channels // 2 + 1)
        num_freqs = hidden_channels // 2 + 1

        # Spectral weights: real-valued learnable magnitudes (phase fixed at 0)
        # Initialized as low-pass filter (Gaussian in frequency domain)
        # Using real values avoids complex gradient issues in PyTorch
        spectral_magnitudes_init = self._init_lowpass_filter(num_freqs, spectral_sigma)
        self.spectral_magnitudes = torch.nn.Parameter(spectral_magnitudes_init)

        # Frequency-domain projection: hidden_channels -> hidden_hidden_channels
        # This aligns frequency features with time-domain features for fusion
        self.freq_projection = torch.nn.Linear(hidden_channels, self.hidden_hidden_channels)

        # Learnable scale parameter for frequency branch contribution
        # Initialized to small value (0.1) to start with mostly time-domain behavior
        self.freq_scale = torch.nn.Parameter(torch.tensor(0.1))

        # Initialize FiLM generator to identity modulation
        torch.nn.init.zeros_(self.film_generator.weight)
        torch.nn.init.zeros_(self.film_generator.bias)

        # Initialize frequency projection with small weights (gentle start)
        torch.nn.init.normal_(self.freq_projection.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.freq_projection.bias)

    def _init_lowpass_filter(self, num_freqs, sigma):
        """Initialize spectral magnitudes as a low-pass filter (Gaussian decay in frequency).

        Args:
            num_freqs: Number of frequency bins (rfft output size).
            sigma: Standard deviation controlling filter bandwidth.

        Returns:
            Real-valued tensor of shape [num_freqs] representing filter magnitudes.
        """
        # Frequency indices: [0, 1, 2, ..., num_freqs-1]
        k = torch.arange(num_freqs, dtype=torch.float32)

        # Gaussian low-pass filter: exp(-k^2 / (2 * sigma^2))
        # Low frequencies (k ~ 0) have weight ~ 1
        # High frequencies (k large) have weight ~ 0
        magnitudes = torch.exp(-k**2 / (2 * sigma**2))

        return magnitudes

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, time_dim: {}, spectral_sigma: {:.2f}, hidden_hidden_channels: {}, num_hidden_layers: {}".format(
            self.input_channels, self.hidden_channels, self.time_dim, self.spectral_sigma,
            self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t, z):
        """
        Arguments:
            t: Current time (scalar or [batch]).
            z: Hidden state of shape (..., hidden_channels).

        Returns:
            Vector field of shape (..., hidden_channels, input_channels).
        """
        # === NaN check on input ===
        if torch.isnan(z).any():
            raise ValueError(f"NaN detected in input z at time {t}")

        # Check for inf in input
        if torch.isinf(z).any():
            raise ValueError(f"Inf detected in input z at time {t}, z range: [{z.min()}, {z.max()}]")

        # Clamp z to prevent extreme values that could cause NaN when multiplied with weights
        z_safe = torch.clamp(z, -10.0, 10.0)

        # === Time-domain branch (FiLM) ===
        # Get time encoding
        if t.dim() == 0:
            batch_size = z.shape[0] if z.dim() > 1 else 1
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim]

        # Generate FiLM parameters
        film_params = self.film_generator(time_enc)  # [batch, 2 * hidden_hidden_channels]
        gamma = film_params[..., :self.hidden_hidden_channels]  # [batch, hidden_hidden_channels]
        beta = film_params[..., self.hidden_hidden_channels:]   # [batch, hidden_hidden_channels]

        # Clamp FiLM parameters to prevent extreme modulation
        gamma = torch.clamp(gamma, -5.0, 5.0)
        beta = torch.clamp(beta, -10.0, 10.0)

        # Broadcast gamma and beta to match z's batch dimensions
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # Multi-layer feature extraction + FiLM modulation
        h_time = self.linear_in(z_safe)  # (..., hidden_hidden_channels)
        if torch.isnan(h_time).any():
            # Check if weights have NaN
            if torch.isnan(self.linear_in.weight).any() or torch.isnan(self.linear_in.bias).any():
                raise ValueError(f"NaN in linear_in weights/bias at time {t}")
            raise ValueError(f"NaN in h_time after linear_in at time {t}, z_safe range: [{z_safe.min()}, {z_safe.max()}]")

        # FiLM modulation: (1 + gamma) * h + beta
        h_time = (1 + gamma) * h_time + beta
        if torch.isnan(h_time).any():
            raise ValueError(f"NaN in h_time before ReLU at time {t}")

        h_time = torch.relu(h_time)
        if torch.isnan(h_time).any():
            raise ValueError(f"NaN in h_time after ReLU at time {t}")

        # Additional hidden layers
        for linear in self.linears:
            h_time = linear(h_time)
            h_time = torch.relu(h_time)

        # === Frequency-domain branch ===
        # Apply FFT on feature dimension (last dimension of z)
        # Use z_safe for FFT as well
        z_freq = torch.fft.rfft(z_safe, dim=-1)  # (..., num_freqs) complex
        if torch.isnan(z_freq).any():
            raise ValueError(f"NaN in z_freq at time {t}")

        # Apply learnable spectral filter (magnitude-only, phase unchanged)
        # Clamp spectral magnitudes to prevent extreme filtering
        spectral_mags_clamped = torch.clamp(self.spectral_magnitudes, 0.0, 10.0)
        z_freq_filtered = z_freq * spectral_mags_clamped
        if torch.isnan(z_freq_filtered).any():
            raise ValueError(f"NaN in z_freq_filtered at time {t}")

        # Inverse FFT back to spatial domain
        h_freq = torch.fft.irfft(z_freq_filtered, n=self.hidden_channels, dim=-1)  # (..., hidden_channels)
        if torch.isnan(h_freq).any():
            raise ValueError(f"NaN in h_freq at time {t}")

        # Clamp FFT output to prevent extreme values
        h_freq = torch.clamp(h_freq, -100.0, 100.0)

        # Project frequency features to hidden dimension to match h_time's shape
        h_freq_proj = self.freq_projection(h_freq)  # (..., hidden_hidden_channels)
        if torch.isnan(h_freq_proj).any():
            raise ValueError(f"NaN in h_freq_proj at time {t}")

        # === Fusion ===
        # Combine time-domain and frequency-domain features
        # Clamp freq_scale to prevent extreme scaling
        freq_scale_clamped = torch.clamp(self.freq_scale, 0.0, 1.0)
        h_fused = h_time + freq_scale_clamped * h_freq_proj
        if torch.isnan(h_fused).any():
            raise ValueError(f"NaN in h_fused at time {t}")

        # === Output ===
        out = self.linear_out(h_fused)
        if torch.isnan(out).any():
            raise ValueError(f"NaN in out before reshape at time {t}")

        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

        # === NaN check on output ===
        if torch.isnan(out).any():
            raise ValueError(f"NaN detected in output at time {t}")

        return out


class SingleHiddenLayer(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(SingleHiddenLayer, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


class FinalTanh(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class _GRU_ODE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(_GRU_ODE, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.W_r = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_z = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_h = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.U_r = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_z = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_h = torch.nn.Linear(hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, x, h):
        r = self.W_r(x) + self.U_r(h)
        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h)
        z = z.sigmoid()
        g = self.W_h(x) + self.U_h(r * h)
        g = g.tanh()
        return (1 - z) * (g - h)


def GRU_ODE(input_channels, hidden_channels):
    func = _GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
    return metamodel.ContinuousRNNConverter(input_channels=input_channels,
                                            hidden_channels=hidden_channels,
                                            model=func)
