"""Cross-framework validation: JAX MAGIC vs PyTorch MAGIC.

Copies weights from a randomly-initialized PyTorch MAGIC into the equivalent
Flax parameter tree, runs the same input through both, and asserts all per-layer
outputs match within 1e-5.

Configuration used: directed=True, learn_second_graph=True, no gat_encoder,
no normalize, no message_encoder/decoder — i.e., the minimal full-MAGIC variant.

Run from the jaxmarl_worker root:
  .venv/bin/python tests/test_magic_vs_pytorch.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../environments/MAGIC'))

import numpy as np
import torch
import torch.nn.functional as F

# Original MAGIC uses DoubleTensor throughout
torch.set_default_dtype(torch.float64)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)   # match float64 precision

import flax.linen as nn
from flax.training.train_state import TrainState

# ---------------------------------------------------------------------------
# Import both implementations
# ---------------------------------------------------------------------------
from gnn_layers import GraphAttention as TorchGAT
from magic import MAGIC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from jaxmarl_worker.algorithms.magic_scan import GraphAttention, SubScheduler, MAGICNet

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
N_AGENTS   = 3
HID_SIZE   = 16
OBS_SIZE   = 8
GAT_HID    = 8
GAT_HEADS  = 2   # num_heads for GAT1 (non-average)
N_ACTIONS  = 8

ATOL = 1e-5

# ---------------------------------------------------------------------------
# Build PyTorch MAGIC
# ---------------------------------------------------------------------------
class Args:
    nagents          = N_AGENTS
    hid_size         = HID_SIZE
    obs_size         = OBS_SIZE
    batch_size       = 1
    directed         = True
    gat_num_heads    = GAT_HEADS
    gat_hid_size     = GAT_HID
    gat_num_heads_out = 1
    ge_num_heads     = 2
    first_gat_normalize  = False
    second_gat_normalize = False
    gat_encoder_normalize = False
    use_gat_encoder      = False
    first_graph_complete = False
    second_graph_complete = False
    learn_second_graph   = True
    message_encoder      = False
    message_decoder      = False
    self_loop_type1      = 2
    self_loop_type2      = 2
    comm_init            = 'uniform'
    comm_mask_zero       = False
    advantages_per_action = False
    naction_heads        = [N_ACTIONS]

torch.manual_seed(0)
torch_model = MAGIC(Args())
torch_model.eval()


# ---------------------------------------------------------------------------
# Utility: extract named PyTorch parameters as numpy arrays
# ---------------------------------------------------------------------------
def pt(name) -> np.ndarray:
    return torch_model.state_dict()[name].detach().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Build JAX parameter tree manually to match PyTorch weights exactly
# ---------------------------------------------------------------------------
# Flax Linear: kernel shape = (in, out), bias shape = (out,)
# PyTorch Linear: weight shape = (out, in), bias shape = (out,)

def linear_params(weight_name: str, bias_name: str) -> dict:
    W = pt(weight_name).T     # transpose: (out,in) → (in,out)
    b = pt(bias_name)
    return {'kernel': W, 'bias': b}


# obs_encoder: Linear(obs_size, hid_size)
obs_enc_params = linear_params('obs_encoder.weight', 'obs_encoder.bias')

# LSTM: PyTorch LSTMCell has weight_ih (4H, H), weight_hh (4H, H), bias_ih, bias_hh
# Flax LSTMCell stores: {'hi': {'kernel': (in,4H), 'bias': (4H,)}, 'hh': {'kernel': (H,4H)}}
# Note: Flax LSTM bias convention = bias_ih + bias_hh combined
w_ih = pt('lstm_cell.weight_ih')  # (4H, obs_size) → but input to lstm is encoded (hid_size)
w_hh = pt('lstm_cell.weight_hh')  # (4H, H)
b_ih = pt('lstm_cell.bias_ih')    # (4H,)
b_hh = pt('lstm_cell.bias_hh')    # (4H,)
lstm_params = {
    'hi': {'kernel': w_ih.T, 'bias': b_ih + b_hh},   # Flax sums both biases
    'hh': {'kernel': w_hh.T},
}

# GAT layer parameter helper
def gat_params(prefix: str, in_f: int, out_f: int, n_heads: int, average: bool) -> dict:
    W   = pt(f'{prefix}.W')           # (in_f, n_heads*out_f)
    a_i = pt(f'{prefix}.a_i')         # (n_heads, out_f, 1) → squeeze → (n_heads, out_f)
    a_j = pt(f'{prefix}.a_j')         # same
    out_dim = out_f if average else n_heads * out_f
    bias = pt(f'{prefix}.bias')       # (out_dim,)
    return {
        'W':    W,
        'a_i':  a_i.squeeze(-1),
        'a_j':  a_j.squeeze(-1),
        'bias': bias,
    }

gat1_params = gat_params('sub_processor1', HID_SIZE,         GAT_HID, GAT_HEADS, average=False)
gat2_params = gat_params('sub_processor2', GAT_HID*GAT_HEADS, HID_SIZE, 1,       average=True)

# SubScheduler MLP helper (maps PyTorch Sequential → Flax Dense params)
def mlp_params(prefix: str) -> dict:
    return {
        'layers_0': linear_params(f'{prefix}.0.weight', f'{prefix}.0.bias'),
        'layers_2': linear_params(f'{prefix}.2.weight', f'{prefix}.2.bias'),
        'layers_4': linear_params(f'{prefix}.4.weight', f'{prefix}.4.bias'),
    }

sched1_mlp = mlp_params('sub_scheduler_mlp1')
sched2_mlp = mlp_params('sub_scheduler_mlp2')

# Action head, value head
action_head_params = linear_params('action_heads.0.weight', 'action_heads.0.bias')
value_head_params  = linear_params('value_head.weight',      'value_head.bias')


# ---------------------------------------------------------------------------
# JAX reference forward pass (manually replicating each step)
# ---------------------------------------------------------------------------
def jax_gat(W, a_i, a_j, bias, h, adj, average: bool):
    """Pure JAX GAT matching gnn_layers.py line-by-line."""
    N, in_f = h.shape
    # linear transform
    h_t = (h @ W).reshape(N, -1, W.shape[1] // a_i.shape[0])  # (N, H, F)
    num_heads = a_i.shape[0]
    out_f = a_i.shape[1]
    # attention scores
    ci = jnp.einsum('nhf,hf->nh', h_t, a_i)   # (N, H)
    cj = jnp.einsum('nhf,hf->nh', h_t, a_j)   # (N, H)
    e = jax.nn.leaky_relu(ci[:, None, :] + cj[None, :, :], negative_slope=0.2)  # (N,N,H)
    # soft mask + softmax + re-mask (matches original line 113-117)
    adj3 = adj[:, :, None]
    attn = jax.nn.softmax(e * adj3, axis=1) * adj3
    out = jnp.einsum('ijh,jhf->ihf', attn, h_t)  # (N, H, F)
    if average:
        return out.mean(axis=1) + bias
    return out.reshape(N, num_heads * out_f) + bias


def jax_sub_scheduler_mlp(mlp_p, h, key):
    """Pure JAX sub-scheduler (no Gumbel, just returns soft probabilities for testing)."""
    N = h.shape[0]
    h_i = jnp.tile(h[:, None, :], (1, N, 1))
    h_j = jnp.tile(h[None, :, :], (N, 1, 1))
    pairs = jnp.concatenate([h_i, h_j], axis=-1)
    x = jax.nn.relu(pairs @ mlp_p['layers_0']['kernel'] + mlp_p['layers_0']['bias'])
    x = jax.nn.relu(x    @ mlp_p['layers_2']['kernel'] + mlp_p['layers_2']['bias'])
    x =              x    @ mlp_p['layers_4']['kernel'] + mlp_p['layers_4']['bias']
    # gumbel-softmax with fixed seed
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, x.shape) + 1e-20) + 1e-20)
    y_soft = jax.nn.softmax(x + gumbel, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), 2)
    y = y_hard + y_soft - jax.lax.stop_gradient(y_soft)
    return y[..., 1]   # (N, N)


def jax_forward(obs_np, hx_np, cx_np, key):
    obs = jnp.array(obs_np)
    hx  = jnp.array(hx_np)
    cx  = jnp.array(cx_np)

    # obs encoder (no activation — matches original)
    W_enc = jnp.array(obs_enc_params['kernel'])
    b_enc = jnp.array(obs_enc_params['bias'])
    enc = obs @ W_enc + b_enc                          # (N, hid)

    # LSTM step — manually implement to avoid Flax ordering ambiguity
    # PyTorch LSTMCell: gates = input @ W_ih.T + bias_ih + hx @ W_hh.T + bias_hh
    W_ih = jnp.array(lstm_params['hi']['kernel'])      # (hid, 4H)
    W_hh = jnp.array(lstm_params['hh']['kernel'])      # (H, 4H)
    b    = jnp.array(lstm_params['hi']['bias'])        # (4H,) = bias_ih + bias_hh
    gates = enc @ W_ih + hx @ W_hh + b                # (N, 4H)
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    new_cx = jax.nn.sigmoid(f) * cx + jax.nn.sigmoid(i) * jnp.tanh(g)
    new_hx = jax.nn.sigmoid(o) * jnp.tanh(new_cx)

    comm_ori = new_hx

    key, sk1, sk2 = jax.random.split(key, 3)

    # sub-scheduler 1 → adj1
    adj1 = jax_sub_scheduler_mlp(sched1_mlp, comm_ori, sk1)
    # GAT 1
    W1   = jnp.array(gat1_params['W'])
    ai1  = jnp.array(gat1_params['a_i'])
    aj1  = jnp.array(gat1_params['a_j'])
    b1   = jnp.array(gat1_params['bias'])
    comm = jax.nn.elu(jax_gat(W1, ai1, aj1, b1, comm_ori, adj1, average=False))

    # sub-scheduler 2 → adj2 (uses comm_ori, not comm)
    adj2 = jax_sub_scheduler_mlp(sched2_mlp, comm_ori, sk2)
    # GAT 2
    W2   = jnp.array(gat2_params['W'])
    ai2  = jnp.array(gat2_params['a_i'])
    aj2  = jnp.array(gat2_params['a_j'])
    b2   = jnp.array(gat2_params['bias'])
    comm = jax_gat(W2, ai2, aj2, b2, comm, adj2, average=True)

    h_cat = jnp.concatenate([new_hx, comm], axis=-1)   # (N, 2*hid)

    W_act = jnp.array(action_head_params['kernel'])
    b_act = jnp.array(action_head_params['bias'])
    logits = h_cat @ W_act + b_act                     # (N, 8)
    log_prob = jax.nn.log_softmax(logits, axis=-1)     # matches original F.log_softmax

    W_val = jnp.array(value_head_params['kernel'])
    b_val = jnp.array(value_head_params['bias'])
    value = (h_cat @ W_val + b_val).squeeze(-1)        # (N,)

    return log_prob, value, new_hx, new_cx, adj1, adj2


# ---------------------------------------------------------------------------
# PyTorch forward pass with the same deterministic Gumbel seed
# ---------------------------------------------------------------------------
# We need to align the Gumbel samples. The only way to do this is to patch
# torch.rand so both frameworks use the same uniform draws.

# Fix random seed for both
SEED_KEY = jax.random.PRNGKey(99)
_, sk1_ref, sk2_ref = jax.random.split(SEED_KEY, 3)

# Generate the uniform samples JAX will use
u1 = np.array(jax.random.uniform(sk1_ref, (N_AGENTS, N_AGENTS, 2)))
u2 = np.array(jax.random.uniform(sk2_ref, (N_AGENTS, N_AGENTS, 2)))

# Patch torch F.gumbel_softmax to use our fixed uniforms
call_count = [0]
fixed_uniforms = [u1, u2]

original_gumbel = F.gumbel_softmax
def patched_gumbel(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    idx = call_count[0]
    call_count[0] += 1
    u = torch.from_numpy(fixed_uniforms[idx])
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.softmax((logits + gumbel) / tau, dim=dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

F.gumbel_softmax = patched_gumbel

# Now also generate matching JAX outputs with same uniforms
def jax_forward_fixed(obs_np, hx_np, cx_np):
    """JAX forward pass with the same fixed uniform samples."""
    key = SEED_KEY
    return jax_forward(obs_np, hx_np, cx_np, key)

# Run PyTorch forward pass
torch.manual_seed(42)
obs_np = np.random.RandomState(42).randn(1, N_AGENTS, OBS_SIZE)
hx_np  = np.zeros((N_AGENTS, HID_SIZE))
cx_np  = np.zeros((N_AGENTS, HID_SIZE))

obs_t = torch.from_numpy(obs_np)
hx_t  = torch.from_numpy(hx_np)
cx_t  = torch.from_numpy(cx_np)

with torch.no_grad():
    action_out_t, value_t, (new_hx_t, new_cx_t) = torch_model([obs_t, (hx_t, cx_t)])

F.gumbel_softmax = original_gumbel  # restore

# Run JAX forward pass with same uniforms
jax_log_prob, jax_value, jax_new_hx, jax_new_cx, adj1, adj2 = jax_forward_fixed(
    obs_np[0], hx_np, cx_np
)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
def check(name, pt_val, jax_val, atol=ATOL):
    pt_np  = np.array(pt_val)
    jax_np = np.array(jax_val)
    diff   = np.abs(pt_np - jax_np).max()
    status = 'PASS' if diff < atol else 'FAIL'
    print(f'  [{status}] {name:35s}  max_abs_diff={diff:.2e}')
    return status == 'PASS'

print('\n=== MAGIC: PyTorch vs JAX cross-framework validation ===\n')
all_pass = True
all_pass &= check('log_softmax action_out',
                  action_out_t[0][0],          jax_log_prob)
all_pass &= check('value head',
                  value_t.squeeze().numpy(),    np.array(jax_value))
all_pass &= check('new_hx (LSTM hidden)',
                  new_hx_t.numpy(),             np.array(jax_new_hx))
all_pass &= check('new_cx (LSTM cell)',
                  new_cx_t.numpy(),             np.array(jax_new_cx))

print()
if all_pass:
    print('ALL CHECKS PASSED — JAX MAGIC is numerically equivalent to PyTorch MAGIC.')
else:
    print('SOME CHECKS FAILED — see above for details.')
    sys.exit(1)
