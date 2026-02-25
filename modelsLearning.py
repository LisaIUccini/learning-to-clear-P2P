import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_sz, output_sz,  hidden_sz=20, device=torch.device('cpu')):
        super().__init__()

        self.hidden_sz = hidden_sz

        std = 0.01

        self.hidden1_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.cell1_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.hidden2_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.cell2_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))

        self.recurs = nn.LSTMCell(input_sz, hidden_sz, device=device)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz, device=device)
        self.output = nn.Linear(hidden_sz, output_sz, device=device)

        self.hidden1 = torch.zeros(1, hidden_sz, device=device)
        self.cell1 = torch.zeros(1, hidden_sz, device=device)
        self.hidden2 = torch.zeros(1, hidden_sz, device=device)
        self.cell2 = torch.zeros(1, hidden_sz, device=device)

    def forward(self, inp, hidden1, cell1, hidden2, cell2):

        (hidden1_, cell1_) = self.recurs(inp, (hidden1, cell1))
        (hidden2_, cell2_) = self.recurs2(hidden1_, (hidden2, cell2))

        return self.output(hidden2_), (hidden1_, cell1_), (hidden2_, cell2_)

# ---------------------------------------------------------------------------
# LearnedUpdate:
#
# Computes:
#   v_t = (α₀ + α₁·t + ... + α_q·t^q) · tanh(MLP([x_t, f(x_t), ∇f(x_t)])) · ρ^t,
#
# where:
#   - α are learnable parameters (transformed via softplus to ensure positivity),
#   - The MLP processes the concatenated vector [x_t, f(x_t), ∇f(x_t)],
#   - ρ is a fixed constant (e.g. 0.99) that scales the update over time.
# ---------------------------------------------------------------------------
class LearnedUpdate(nn.Module):
    def __init__(self, d, q, rho):
        """
        d: dimension of parameter vector x.
        q: order of the polynomial (i.e. there are q+1 coefficients). For example, q=2 uses [1, t, t²]
        rho: fixed discount factor (must be less than 1).
        """
        super(LearnedUpdate, self).__init__()
        # The MLP takes a vector of size 2*d + 1 ([x, f(x), grad f(x)]) and outputs a vector of size d.
        # self.mlp = SimpleMLP(input_size=2 * d + 1, hidden_sizes=hidden_sizes, output_size=d)
        
        self.lstm = TwoLayerLSTM(input_sz=d, output_sz=d, hidden_sz=20) 
        
        # Initialize raw alpha parameters; they will be transformed to be positive.
        self.alpha_raw = nn.Parameter(torch.full((q + 1,), -2.0))

        self.q = q
        self.rho = rho  # fixed (non-learnable) discount factor
    
    def forward(self, input, t):
        """
        input: tensor of shape (batch_size, n_agents * d, 1)
        t: current time-step (an integer or float)
        """

        batch_size = input.shape[0]
        results = []

        for i in range(batch_size):
            # # Compute separate alpha values
            # alpha_x = torch.norm(x_batch[i], p=float('inf'))  # Max norm of x_batch[i]
            # alpha_loss = torch.norm(loss_batch[i], p=float('inf'))  # Max norm of loss_batch[i]
            # alpha_grad = torch.norm(grad_batch[i], p=float('inf'))  # Max norm of grad_batch[i]

            # # Prevent division by zero (if any alpha is zero, set it to 1 to avoid NaNs)
            # alpha_x = alpha_x if alpha_x > 0 else torch.tensor(1.0, device=x_batch[i].device)
            # alpha_loss = alpha_loss if alpha_loss > 0 else torch.tensor(1.0, device=loss_batch[i].device)
            # alpha_grad = alpha_grad if alpha_grad > 0 else torch.tensor(1.0, device=grad_batch[i].device)

            # # Scale inputs
            # x_scaled = x_batch[i] / alpha_x
            # loss_scaled = loss_batch[i] / alpha_loss
            # grad_scaled = grad_batch[i] / alpha_grad

            # Stack scaled inputs and pass through MLP
            # input_vec = torch.vstack([x_scaled, loss_scaled, grad_scaled])  # Shape: (2d+1, 1)
            alpha = torch.norm(input[i], p=float('inf'))  # Max norm of input_vec
            alpha = alpha if alpha > 0 else torch.tensor(1.0, device=input[i].device)
            input_vec = input[i] / alpha # Shape: (n_agents*d, 1)

            if t == 0:
                lstm_output, (self.lstm.hidden1, self.lstm.cell1), (self.lstm.hidden2, self.lstm.cell2) = self.lstm.forward(input_vec.T, self.lstm.hidden1_0, self.lstm.cell1_0, self.lstm.hidden2_0, self.lstm.cell2_0)
                lstm_output = torch.tanh(lstm_output).squeeze(0)
            else:
                lstm_output, (self.lstm.hidden1, self.lstm.cell1), (self.lstm.hidden2, self.lstm.cell2) = self.lstm.forward(input_vec.T, self.lstm.hidden1, self.lstm.cell1, self.lstm.hidden2, self.lstm.cell2)
                lstm_output = torch.tanh(lstm_output).squeeze(0)
                    
        
            # Rescale output
            output = lstm_output * alpha

            results.append(output) 

        direction_batch = torch.stack(results, dim=0)  # Shape: (batch_size, 2d+1, d)

        # Compute the polynomial factor
        alpha = F.softplus(self.alpha_raw)
        poly_terms = torch.tensor([t ** j for j in range(self.q + 1)], device=input.device, dtype=input.dtype)
        poly_factor = torch.dot(alpha, poly_terms)

        # Apply update rule
        v_t_batch = poly_factor * direction_batch * (self.rho ** t)  # Shape: (batch_size, 2d+1, d)

        return v_t_batch
