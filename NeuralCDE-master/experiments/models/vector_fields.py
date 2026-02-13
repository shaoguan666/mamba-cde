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
