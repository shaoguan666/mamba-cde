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

        # Logging mode for visualization and analysis
        self.logging_enabled = False
        self.logs = []

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

    def set_logging(self, enabled):
        """Enable or disable logging mode for internal dynamics visualization.

        Arguments:
            enabled: Boolean to turn logging on/off.
        """
        self.logging_enabled = enabled
        if enabled:
            self.logs = []  # Clear logs when enabling

    def clear_logs(self):
        """Clear all stored logs."""
        self.logs = []

    def extract_logs(self):
        """Extract logged data as NumPy arrays for analysis.

        Returns:
            Dictionary containing:
                - 'time': Array of time points, shape (num_steps,)
                - 'gamma': Array of FiLM gamma parameters, shape (num_steps, hidden_hidden_channels)
                - 'beta': Array of FiLM beta parameters, shape (num_steps, hidden_hidden_channels)
                - 'spectral_weights': Array of frequency weights, shape (num_steps, num_freqs)
                - 'time_branch_norm': Array of L2 norms for time branch, shape (num_steps,)
                - 'spectral_branch_norm': Array of L2 norms for spectral branch, shape (num_steps,)
                - 'contribution_ratio': Array of spectral/time branch ratio, shape (num_steps,)
        """
        import numpy as np

        if not self.logs:
            return {
                'time': np.array([]),
                'gamma': np.array([]),
                'beta': np.array([]),
                'spectral_weights': np.array([]),
                'time_branch_norm': np.array([]),
                'spectral_branch_norm': np.array([]),
                'contribution_ratio': np.array([])
            }

        # Extract and stack all logged entries
        times = np.array([log['time'] for log in self.logs])
        gammas = np.stack([log['gamma'] for log in self.logs], axis=0)
        betas = np.stack([log['beta'] for log in self.logs], axis=0)
        spectral_weights = np.stack([log['spectral_weights'] for log in self.logs], axis=0)
        time_norms = np.array([log['time_branch_norm'] for log in self.logs])
        spectral_norms = np.array([log['spectral_branch_norm'] for log in self.logs])

        # Compute contribution ratio (spectral / time), avoiding division by zero
        contribution_ratio = np.where(
            time_norms > 1e-8,
            spectral_norms / time_norms,
            0.0
        )

        return {
            'time': times,
            'gamma': gammas,
            'beta': betas,
            'spectral_weights': spectral_weights,
            'time_branch_norm': time_norms,
            'spectral_branch_norm': spectral_norms,
            'contribution_ratio': contribution_ratio
        }

    def forward(self, t, z):
        """
        Arguments:
            t: Current time (scalar or [batch]).
            z: Hidden state of shape (..., hidden_channels).

        Returns:
            Vector field of shape (..., hidden_channels, input_channels).
        """
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

        # Broadcast gamma and beta to match z's batch dimensions
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # Multi-layer feature extraction + FiLM modulation
        h_time = self.linear_in(z)  # (..., hidden_hidden_channels)

        # FiLM modulation: (1 + gamma) * h + beta
        h_time = (1 + gamma) * h_time + beta
        h_time = h_time.relu()

        # Additional hidden layers
        for linear in self.linears:
            h_time = linear(h_time)
            h_time = h_time.relu()

        # === Frequency-domain branch ===
        # Apply FFT on feature dimension (last dimension of z)
        z_freq = torch.fft.rfft(z, dim=-1)  # (..., num_freqs) complex

        # Apply learnable spectral filter (magnitude-only, phase unchanged)
        z_freq_filtered = z_freq * self.spectral_magnitudes
        # Inverse FFT back to spatial domain
        h_freq = torch.fft.irfft(z_freq_filtered, n=self.hidden_channels, dim=-1)  # (..., hidden_channels)

        # Project frequency features to hidden dimension to match h_time's shape
        h_freq_proj = self.freq_projection(h_freq)  # (..., hidden_hidden_channels)

        # === Logging (if enabled) ===
        if self.logging_enabled:
            gamma_log = gamma.mean(dim=0).detach().cpu().numpy()
            beta_log = beta.mean(dim=0).detach().cpu().numpy()
            spectral_weights_log = self.spectral_magnitudes.detach().cpu().numpy()
            time_branch_norm = torch.norm(h_time, p=2, dim=-1).mean().item()
            spectral_branch_norm = torch.norm(h_freq_proj, p=2, dim=-1).mean().item()

            if t.dim() == 0:
                time_log = t.item()
            else:
                time_log = t[0].item()

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'spectral_weights': spectral_weights_log,
                'time_branch_norm': time_branch_norm,
                'spectral_branch_norm': spectral_branch_norm
            })

        # === Fusion ===
        h_fused = h_time + self.freq_scale * h_freq_proj

        # === Output ===
        out = self.linear_out(h_fused)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        # tanh bounds output to [-1, 1], critical for ODE integration stability
        out = out.tanh()

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


# === Mamba-NCDE Integration ===
try:
    from .mamba_vector_field import MambaModulatedVectorField
    MAMBA_AVAILABLE = True
except ImportError:
    MambaModulatedVectorField = None
    MAMBA_AVAILABLE = False


class DeepFiLMVectorField(torch.nn.Module):
    """Deep FiLM-modulated vector field with strong regularization.

    Pure time-branch architecture that removes ineffective global branches (Spectral/Mamba).
    Focus on doing local temporal dynamics extremely well with:
    - Multi-layer FiLM modulation on EVERY layer
    - Per-layer dropout for strong regularization
    - Learnable time encoder
    - Tanh-bounded output for ODE stability

    Design Philosophy:
        "Simple is better than complex. Do one thing well."
        - Sepsis prediction is about local event detection, not global patterns
        - Remove complexity, reduce overfitting, improve generalization

    Key Improvements over Spectral/Mamba:
        1. Fewer parameters (~240K vs 280K/616K) -> less overfitting
        2. Strong regularization (per-layer dropout) -> better generalization
        3. Deeper FiLM (4 layers with modulation on each) -> more expressive
        4. No wasted computation on ineffective branches

    Expected Performance:
        - AUROC: 0.90-0.91 (target: beat baseline 0.8986)
        - Training: More stable, faster convergence
        - Overfitting: Significantly reduced
    """

    def __init__(self, input_channels, hidden_channels, time_dim=32,
                 hidden_hidden_channels=64, num_hidden_layers=4, dropout=0.2):
        """
        Arguments:
            input_channels: Number of input channels (from control signal X).
            hidden_channels: Number of hidden channels (state z dimension).
            time_dim: Dimension of time encoding (default 32).
            hidden_hidden_channels: Hidden layer dimension (default 64).
            num_hidden_layers: Number of hidden layers (default 4).
            dropout: Dropout probability for regularization (default 0.2).
        """
        super(DeepFiLMVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

        # Learnable time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # Multi-layer MLP
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])

        # Per-layer FiLM generators (one for each layer including input)
        self.film_generators = torch.nn.ModuleList([
            torch.nn.Linear(time_dim, 2 * hidden_hidden_channels)
            for _ in range(num_hidden_layers)
        ])

        # Per-layer dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)

        # Output projection
        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                         input_channels * hidden_channels)

        # Initialize weights
        self._init_weights()

        # Logging system for analysis
        self.logging_enabled = False
        self.logs = []

    def _init_weights(self):
        """Initialize weights for stable training start.

        Strategy:
            - FiLM: identity transform (gamma=0, beta=0)
            - Output: small initialization (gain=0.1)
            - This ensures the model starts close to a simple MLP
        """
        # FiLM generators: initialize to identity (no modulation at start)
        for film_gen in self.film_generators:
            torch.nn.init.zeros_(film_gen.weight)
            torch.nn.init.zeros_(film_gen.bias)

        # Output layer: small initialization for stability
        torch.nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self):
        return ("input_channels: {}, hidden_channels: {}, time_dim: {}, "
                "hidden_hidden_channels: {}, num_hidden_layers: {}, dropout: {:.2f}").format(
                    self.input_channels, self.hidden_channels, self.time_dim,
                    self.hidden_hidden_channels, self.num_hidden_layers, self.dropout_p)

    def set_logging(self, enabled):
        """Enable/disable logging for visualization."""
        self.logging_enabled = enabled
        if enabled:
            self.logs = []

    def clear_logs(self):
        """Clear all stored logs."""
        self.logs = []

    def extract_logs(self):
        """Extract logged data as NumPy arrays for analysis.

        Returns:
            Dictionary containing:
                - 'time': time points
                - 'gamma': FiLM gamma parameters (first layer)
                - 'beta': FiLM beta parameters (first layer)
                - 'layer_norms': L2 norms at each layer
        """
        import numpy as np

        if not self.logs:
            return {
                'time': np.array([]),
                'gamma': np.array([]),
                'beta': np.array([]),
                'layer_norms': np.array([])
            }

        return {
            'time': np.array([log['time'] for log in self.logs]),
            'gamma': np.stack([log['gamma'] for log in self.logs], axis=0),
            'beta': np.stack([log['beta'] for log in self.logs], axis=0),
            'layer_norms': np.stack([log['layer_norms'] for log in self.logs], axis=0)
        }

    def forward(self, t, z):
        """
        Arguments:
            t: Current time (scalar or [batch]).
            z: Hidden state of shape (..., hidden_channels).

        Returns:
            Vector field of shape (..., hidden_channels, input_channels).
        """
        # === Time encoding ===
        if t.dim() == 0:
            batch_size = z.shape[0] if z.dim() > 1 else 1
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim]

        # === Generate all FiLM parameters ===
        film_params_all = [fg(time_enc) for fg in self.film_generators]
        # Each: [batch, 2 * hidden_hidden_channels]

        # Store first layer params for logging
        gamma_0 = film_params_all[0][..., :self.hidden_hidden_channels]
        beta_0 = film_params_all[0][..., self.hidden_hidden_channels:]

        # Broadcast FiLM params for multi-dim z
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            film_params_broadcast = []
            for fp in film_params_all:
                for _ in range(extra_dims):
                    fp = fp.unsqueeze(1)
                film_params_broadcast.append(fp)
            film_params_all = film_params_broadcast

        # === Layer 0: linear_in + FiLM + ReLU + Dropout ===
        gamma = film_params_all[0][..., :self.hidden_hidden_channels]
        beta = film_params_all[0][..., self.hidden_hidden_channels:]

        h = self.linear_in(z)
        h = (1 + gamma) * h + beta
        h = torch.relu(h)
        h = self.dropout(h)

        # Track layer norms for logging
        layer_norms = [torch.norm(h, p=2, dim=-1).mean().item()] if self.logging_enabled else []

        # === Layers 1..N-1: linears[i] + FiLM + ReLU + Dropout ===
        for i, linear in enumerate(self.linears):
            gamma_i = film_params_all[i + 1][..., :self.hidden_hidden_channels]
            beta_i = film_params_all[i + 1][..., self.hidden_hidden_channels:]

            h = linear(h)
            h = (1 + gamma_i) * h + beta_i
            h = torch.relu(h)
            h = self.dropout(h)

            if self.logging_enabled:
                layer_norms.append(torch.norm(h, p=2, dim=-1).mean().item())

        # === Output ===
        out = self.linear_out(h)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        out = torch.tanh(out)  # CRITICAL: bounded for ODE stability

        # === Logging ===
        if self.logging_enabled:
            gamma_log = gamma_0.mean(dim=0).detach().cpu().numpy()
            beta_log = beta_0.mean(dim=0).detach().cpu().numpy()
            time_log = t.item() if t.dim() == 0 else t[0].item()

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'layer_norms': layer_norms
            })

        return out


class DeepMLPVectorField(torch.nn.Module):
    """Ablation: Deep MLP without FiLM or Dropout.

    Stateless vector field f(z) - no time dependence.
    Used to isolate whether FiLM time-modulation drives DeepFiLM's improvement.
    """

    def __init__(self, input_channels, hidden_channels,
                 hidden_hidden_channels=64, num_hidden_layers=4):
        super(DeepMLPVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])
        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                          input_channels * hidden_channels)

        torch.nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self):
        return ("input_channels: {}, hidden_channels: {}, "
                "hidden_hidden_channels: {}, num_hidden_layers: {}").format(
                    self.input_channels, self.hidden_channels,
                    self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        h = self.linear_in(z)
        h = torch.relu(h)
        for linear in self.linears:
            h = linear(h)
            h = torch.relu(h)
        out = self.linear_out(h)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return torch.tanh(out)


class DeepMLPDropoutVectorField(torch.nn.Module):
    """Ablation: Deep MLP with Dropout but without FiLM.

    Stateless vector field f(z) - no time dependence.
    Used to isolate regularization contribution from FiLM time-modulation.
    """

    def __init__(self, input_channels, hidden_channels,
                 hidden_hidden_channels=64, num_hidden_layers=4, dropout=0.2):
        super(DeepMLPDropoutVectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])
        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                          input_channels * hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self):
        return ("input_channels: {}, hidden_channels: {}, "
                "hidden_hidden_channels: {}, num_hidden_layers: {}, dropout: {:.2f}").format(
                    self.input_channels, self.hidden_channels,
                    self.hidden_hidden_channels, self.num_hidden_layers, self.dropout_p)

    def forward(self, z):
        h = self.linear_in(z)
        h = torch.relu(h)
        h = self.dropout(h)
        for linear in self.linears:
            h = linear(h)
            h = torch.relu(h)
            h = self.dropout(h)
        out = self.linear_out(h)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return torch.tanh(out)


class LowRankODE_FiLM_VectorField(torch.nn.Module):
    """Low-Rank ODE-Enhanced FiLM Vector Field.

    Combines Mode-Mamba-TS's structured state dynamics with DeepFiLM's proven architecture.

    Architecture:
        1. SSM Structured Branch (Mode-Mamba-TS inspired):
           - S4D diagonal base matrix A_base = -exp(A_log)
           - Time-varying low-rank perturbation: A(t) = A_base + alpha * U(t) @ V(t)^T
           - Selective gating: gate(z) * A(t) @ h

        2. DeepFiLM Branch (proven AUROC 0.903):
           - Multi-layer MLP with per-layer FiLM modulation
           - Per-layer dropout for regularization
           - Tanh-bounded output

        3. Fusion:
           - SSM features injected via residual at Layer 1
           - h_layer1 = (1 + gamma_1) * (h + h_ssm) + beta_1

    Key Design Principles:
        - Stateless: f(t, z) only, compatible with ODE solvers
        - Small perturbation (alpha=0.05): SSM augments, doesn't replace
        - Low-rank (r=4): Parameter-efficient state interaction
        - Selective gating: Input-dependent activation of structured dynamics

    Parameters:
        - SSM components: ~16K additional params
        - DeepFiLM base: ~240K params
        - Total: ~256K params (+6.6% overhead)

    Expected Performance:
        - Target AUROC: 0.905-0.915 (beat DeepFiLM 0.903)
        - Risk: Low (small perturbation, safe fallback to DeepFiLM)
    """

    def __init__(self, input_channels, hidden_channels, time_dim=32,
                 hidden_hidden_channels=49, num_hidden_layers=4,
                 dropout=0.2, r_rank=4):
        """
        Arguments:
            input_channels: Number of input channels (control signal X).
            hidden_channels: Number of hidden channels (state z dimension).
            time_dim: Dimension of time encoding (default 32).
            hidden_hidden_channels: Hidden layer dimension (default 49).
            num_hidden_layers: Number of hidden layers (default 4).
            dropout: Dropout probability (default 0.2).
            r_rank: Rank of low-rank factorization (default 4).
        """
        super(LowRankODE_FiLM_VectorField, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout
        self.r_rank = r_rank

        # === Time Encoder (from DeepFiLM) ===
        self.time_encoder = TimeEncoder(time_dim)

        # === SSM Structured Dynamics (Mode-Mamba-TS inspired) ===
        # S4D diagonal base initialization: A_log = log([1, 2, 3, ..., d])
        A_diag = torch.arange(1, hidden_hidden_channels + 1, dtype=torch.float32)
        self.A_log = torch.nn.Parameter(torch.log(A_diag))
        self.A_log._no_weight_decay = True  # Mark for optimizer

        # Low-rank perturbation scale (small initialization)
        self.alpha = torch.nn.Parameter(torch.tensor(0.05))

        # Time-conditioned low-rank factors: U(t), V(t)
        # u_proj: time_enc [time_dim] -> U [hidden_hidden_channels * r_rank]
        self.u_proj = torch.nn.Linear(time_dim, hidden_hidden_channels * r_rank, bias=False)
        # v_proj: time_enc [time_dim] -> V [r_rank * hidden_hidden_channels]
        self.v_proj = torch.nn.Linear(time_dim, r_rank * hidden_hidden_channels, bias=False)

        # Selective gating (Mamba-inspired): gate = sigmoid(W_g @ z)
        # Projects from z-space to h-space for gating
        self.gate_proj = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        # === DeepFiLM Components (proven architecture) ===
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        # Multi-layer MLP
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])

        # Per-layer FiLM generators
        self.film_generators = torch.nn.ModuleList([
            torch.nn.Linear(time_dim, 2 * hidden_hidden_channels)
            for _ in range(num_hidden_layers)
        ])

        # Per-layer dropout
        self.dropout = torch.nn.Dropout(dropout)

        # Output projection
        self.linear_out = torch.nn.Linear(hidden_hidden_channels,
                                         input_channels * hidden_channels)

        # === Initialization ===
        self._init_weights()

        # === Logging system for visualization ===
        self.logging_enabled = False
        self.logs = []

    def _init_weights(self):
        """Initialize weights for stable training.

        Strategy:
            - SSM: small random init for U, V projections
            - Gate: small init (near 0.5 after sigmoid)
            - FiLM: identity transform (gamma=0, beta=0)
            - alpha: 0.05 (small SSM contribution)
            - Output: small gain (0.1)
        """
        # U, V projections: small random initialization
        torch.nn.init.xavier_uniform_(self.u_proj.weight, gain=0.02)
        torch.nn.init.xavier_uniform_(self.v_proj.weight, gain=0.02)

        # Gate projection: init to give ~0.5 after sigmoid
        torch.nn.init.zeros_(self.gate_proj.weight)
        torch.nn.init.zeros_(self.gate_proj.bias)

        # FiLM generators: identity transform
        for film_gen in self.film_generators:
            torch.nn.init.zeros_(film_gen.weight)
            torch.nn.init.zeros_(film_gen.bias)

        # Output: small initialization
        torch.nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self):
        return (f"input_channels={self.input_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"time_dim={self.time_dim}, "
                f"hidden_hidden_channels={self.hidden_hidden_channels}, "
                f"num_hidden_layers={self.num_hidden_layers}, "
                f"r_rank={self.r_rank}, "
                f"dropout={self.dropout_p:.2f}")

    def set_logging(self, enabled):
        """Enable/disable logging for visualization."""
        self.logging_enabled = enabled
        if enabled:
            self.logs = []

    def clear_logs(self):
        """Clear all stored logs."""
        self.logs = []

    def extract_logs(self):
        """Extract logged data as NumPy arrays for visualization.

        IMPORTANT: This method automatically clears logs after extraction to prevent contamination.
        Time series are sorted by time to handle ODE solver step rejections.

        Returns:
            Dictionary containing:
                - 'time': time points [T]
                - 'gamma': FiLM gamma params [T, D]
                - 'beta': FiLM beta params [T, D]
                - 'alpha': SSM scale factor [T]
                - 'gate_mean': Mean gate activation [T]
                - 'A_base': S4D diagonal values [D]
                - 'ssm_norm': L2 norm of SSM features [T]
                - 'mlp_norm': L2 norm of MLP features [T]
        """
        import numpy as np

        if not self.logs:
            return {
                'time': np.array([]),
                'gamma': np.array([]),
                'beta': np.array([]),
                'alpha': np.array([]),
                'gate_mean': np.array([]),
                'A_base': np.array([]),
                'ssm_norm': np.array([]),
                'mlp_norm': np.array([]),
            }

        # Sort logs by time to handle ODE solver step rejections
        sorted_logs = sorted(self.logs, key=lambda x: x['time'])

        result = {
            'time': np.array([log['time'] for log in sorted_logs]),
            'gamma': np.stack([log['gamma'] for log in sorted_logs], axis=0),
            'beta': np.stack([log['beta'] for log in sorted_logs], axis=0),
            'alpha': np.array([log['alpha'] for log in sorted_logs]),
            'gate_mean': np.array([log['gate_mean'] for log in sorted_logs]),
            'A_base': sorted_logs[0]['A_base'],  # Constant across time
            'ssm_norm': np.array([log['ssm_norm'] for log in sorted_logs]),
            'mlp_norm': np.array([log['mlp_norm'] for log in sorted_logs]),
        }

        # CRITICAL: Clear logs after extraction to prevent contamination
        self.logs = []

        return result

    def forward(self, t, z):
        """
        Arguments:
            t: Current time (scalar or [batch]).
            z: Hidden state of shape [batch, hidden_channels] or [..., hidden_channels].

        Returns:
            Vector field of shape [..., hidden_channels, input_channels].
        """
        batch_shape = z.shape[:-1]
        batch_size = z.shape[0] if z.dim() > 1 else 1

        # === Time Encoding ===
        # Handle scalar time: expand to batch size
        if t.dim() == 0:
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim=32]

        # === Generate FiLM parameters for all layers ===
        film_params_all = [fg(time_enc) for fg in self.film_generators]
        # Each: [batch, 2 * hidden_hidden_channels]

        # Broadcast FiLM params if z has extra batch dimensions
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            film_params_broadcast = []
            for fp in film_params_all:
                for _ in range(extra_dims):
                    fp = fp.unsqueeze(1)
                film_params_broadcast.append(fp)
            film_params_all = film_params_broadcast

        # === Project to Hidden Space ===
        h = self.linear_in(z)  # [..., hidden_hidden_channels=49]

        # === SSM Structured Dynamics ===
        # A_base: S4D diagonal [hidden_hidden_channels]
        A_base = -torch.exp(self.A_log.float())  # Negative for stability

        # Time-varying low-rank factors U(t), V(t)
        U_flat = self.u_proj(time_enc)  # [batch, hidden_hidden_channels * r_rank]
        V_flat = self.v_proj(time_enc)  # [batch, r_rank * hidden_hidden_channels]

        # Reshape to matrix form
        U = U_flat.view(batch_size, self.hidden_hidden_channels, self.r_rank)
        # U: [batch, 49, 4]
        V = V_flat.view(batch_size, self.r_rank, self.hidden_hidden_channels)
        # V: [batch, 4, 49]

        # Compute structured dynamics: h_ssm = A_base * h + alpha * U @ V^T @ h
        # Step 1: A_base * h (element-wise, A_base is diagonal)
        h_diag = A_base * h  # [batch, 49]

        # Step 2: Low-rank correction via U @ (V @ h)
        # V @ h: project h to low-rank space [batch, r_rank]
        Vh = torch.bmm(V, h.unsqueeze(-1)).squeeze(-1)
        # V: [batch, 4, 49] @ h: [batch, 49, 1] = [batch, 4, 1] -> [batch, 4]

        # U @ (V @ h): project back to hidden space [batch, hidden_hidden_channels]
        UVh = torch.bmm(U, Vh.unsqueeze(-1)).squeeze(-1)
        # U: [batch, 49, 4] @ Vh: [batch, 4, 1] = [batch, 49, 1] -> [batch, 49]

        # CRITICAL: Bound low-rank perturbation to [-1, 1] for ODE stability
        # This prevents SSM branch from causing vector field divergence
        UVh = torch.tanh(UVh)

        # Combine: h_ssm = diag + bounded_low_rank
        h_ssm = h_diag + self.alpha * UVh  # [batch, 49]

        # === Selective Gating (Mamba-inspired) ===
        # Gate depends on original state z (input-dependent selectivity)
        gate = torch.sigmoid(self.gate_proj(z))  # [batch, 49]
        h_ssm = gate * h_ssm  # Apply gate

        # === DeepFiLM Layers with SSM Injection ===
        # Layer 0: Inject SSM features via residual
        gamma_0 = film_params_all[0][..., :self.hidden_hidden_channels]
        beta_0 = film_params_all[0][..., self.hidden_hidden_channels:]

        h = (1 + gamma_0) * (h + h_ssm) + beta_0  # SSM injection here!
        h = torch.relu(h)
        h = self.dropout(h)

        # Layers 1..N-1: Standard DeepFiLM
        for i, linear in enumerate(self.linears):
            gamma_i = film_params_all[i + 1][..., :self.hidden_hidden_channels]
            beta_i = film_params_all[i + 1][..., self.hidden_hidden_channels:]

            h = linear(h)
            h = (1 + gamma_i) * h + beta_i
            h = torch.relu(h)
            h = self.dropout(h)

        # === Output ===
        out = self.linear_out(h)  # [batch, hidden_channels * input_channels]
        out = out.view(*batch_shape, self.hidden_channels, self.input_channels)
        out = torch.tanh(out)  # CRITICAL: bounded for ODE stability

        # === Logging (if enabled) ===
        if self.logging_enabled:
            # Extract time value
            time_log = t.item() if t.dim() == 0 else t[0].item()

            # Log FiLM parameters (first layer, batch mean)
            gamma_log = gamma_0.mean(dim=0).detach().cpu().numpy()
            beta_log = beta_0.mean(dim=0).detach().cpu().numpy()

            # Log SSM parameters
            alpha_log = self.alpha.item()
            gate_mean_log = gate.mean().item()
            A_base_log = A_base.detach().cpu().numpy()

            # Log branch norms (before fusion)
            h_mlp_norm = torch.norm(h, p=2, dim=-1).mean().item()
            h_ssm_unfused = h_diag + self.alpha * UVh  # Before gating
            ssm_norm_log = torch.norm(h_ssm_unfused, p=2, dim=-1).mean().item()

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'alpha': alpha_log,
                'gate_mean': gate_mean_log,
                'A_base': A_base_log,
                'ssm_norm': ssm_norm_log,
                'mlp_norm': h_mlp_norm,
            })

        return out
