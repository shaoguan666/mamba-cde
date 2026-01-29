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
    """
    def __init__(self, input_channels, hidden_channels, time_dim=32):
        """
        Arguments:
            input_channels: Number of input channels (from control signal X).
            hidden_channels: Number of hidden channels (state z dimension).
            time_dim: Dimension of time encoding (default 32).
        """
        super(ModulatedSingleHiddenLayer, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim

        # Hidden layer dimension (hardcoded to 128 for consistency with SingleHiddenLayer)
        self.hidden_hidden = 128

        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # FiLM parameter generator: time_dim -> 2 * 128 (gamma and beta)
        self.film_generator = torch.nn.Linear(time_dim, 2 * self.hidden_hidden)

        # Main network (same structure as SingleHiddenLayer)
        self.linear1 = torch.nn.Linear(hidden_channels, self.hidden_hidden)
        self.linear2 = torch.nn.Linear(self.hidden_hidden, input_channels * hidden_channels)

        # Initialize FiLM generator to identity modulation at start
        # gamma = 0, beta = 0 means h_mod = (1 + 0) * h_raw + 0 = h_raw
        torch.nn.init.zeros_(self.film_generator.weight)
        torch.nn.init.zeros_(self.film_generator.bias)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, time_dim: {}".format(
            self.input_channels, self.hidden_channels, self.time_dim)

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
        film_params = self.film_generator(time_enc)  # [batch, 2 * 128]
        gamma = film_params[..., :self.hidden_hidden]  # [batch, 128]
        beta = film_params[..., self.hidden_hidden:]   # [batch, 128]

        # Broadcast gamma and beta to match z's batch dimensions
        # z shape: (..., hidden_channels), we need to align gamma/beta
        if z.dim() > 2:
            # z has extra batch dimensions, need to reshape gamma/beta
            # Assuming first dim of z is batch
            extra_dims = z.dim() - 2  # number of extra dimensions between batch and channels
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # Feature extraction
        h_raw = self.linear1(z)  # (..., 128)

        # FiLM modulation: (1 + gamma) * h_raw + beta
        h_mod = (1 + gamma) * h_raw + beta

        # Activation and output
        h_mod = torch.relu(h_mod)
        out = self.linear2(h_mod)

        # Reshape to match expected output: (..., hidden_channels, input_channels)
        out = out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

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
