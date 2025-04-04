import time
import numpy as np

import torch
torch.set_printoptions(precision=32)
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from core.spanning_tree_cat import sample_tree_from_logits
from core.topk  import sample_topk_from_logits

EPS = torch.finfo(torch.float32).tiny

# --- Begin Custom Estimator Definition ---
class Categorical(torch.autograd.Function):
    generate_vmap_rule = True  # for functorch if needed

    @staticmethod
    def forward(ctx, p):
        # p is assumed to be of shape (..., k) representing a categorical distribution.
        result = torch.multinomial(p, num_samples=1)  # shape (..., 1)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        # Adding a small epsilon to avoid division by zero.
        eps = 1e-8
        w_chosen = (1.0 / (p + eps)) / 2  
        w_non_chosen = (1.0 / (1.0 - p + eps)) / 2  
        ws = one_hot * w_chosen + (1 - one_hot) * w_non_chosen
        grad_output_expanded = grad_output.expand_as(p)
        return grad_output_expanded * ws
# --- End Custom Estimator Definition ---

def get_experiments_folder(args):
    folder = args.suffix.strip("_")
    # SST-related parameters.
    if args.sst == "tree":
        folder += f"_tree_{args.relaxation}"
        folder += f"_mr{args.max_range}" if args.max_range > -np.inf else ""
    elif args.sst == "topk":
        folder += f"_topk_{args.relaxation}"
    else: # args.sst == "indep"
        folder += f"_indep"
    # Whether or not kl is computed wrt U (gumbels).
    folder += "_gkl" if args.use_gumbels_for_kl else ""
    # For when REINFORCE or NVIL is used.
    if args.use_reinforce:
        folder += f"_reinforce_{args.reinforce_baseline}"
    if args.use_nvil:
        folder += "_nvil"
    folder += f"_nedgetypes{args.edge_types}"
    folder += f"_edgesymm" if args.symmeterize_logits else ""
    folder += f"_pred{args.prediction_steps}"
    folder += f"_r{args.num_rounds}"
    if args.add_timestamp:
        timestr = time.strftime("%Y%m%d")
        folder += f"_{timestr}"
    return folder


def get_experiment_name(args):
    name = (f"lr{args.lr}_temp{args.temp}_encwd{args.enc_weight_decay}"
            f"_decwd{args.dec_weight_decay}")
    if args.sst == "topk":
        name += f"_eps{args.eps_for_finitediff}"
    if (args.use_reinforce or args.use_nvil) and args.ema_for_loss > 0.0:
        name += f"_ema{args.ema_for_loss}"
    name += f"_{args.seed}"
    return name


def load_data(batch_size, eval_batch_size, suffix, normalize=True):
    data_train = np.load(f"data/data_train{suffix}.npy")
    edges_train = np.load(f"data/edges_train{suffix}.npy")

    data_valid = np.load(f"data/data_valid{suffix}.npy")
    edges_valid = np.load(f"data/edges_valid{suffix}.npy")

    data_test = np.load(f"data/data_test{suffix}.npy")
    edges_test = np.load(f"data/edges_test{suffix}.npy")

    # [num_samples, num_timesteps, num_dims, num_vertices]
    num_vertices = data_train.shape[3]

    data_max = data_train.max()
    data_min = data_train.min()

    # Normalize to [-1, 1]
    if normalize:
        data_train = (data_train - data_min) * 2 / (data_max - data_min) - 1

        data_valid = (data_valid - data_min) * 2 / (data_max - data_min) - 1

        data_test = (data_test - data_min) * 2 / (data_max - data_min) - 1

    # Reshape to: [num_sims, num_vertices, num_timesteps, num_dims]
    feat_train = np.transpose(data_train, [0, 3, 1, 2])
    # Transpose edges to be consistent with the output of the encoder,
    # which is corresponds to a flattened adjacency matrix that is transposed
    # and has its diagonal removed. This is not necessary when the input
    # data is symmetric, which is the case for the graph layout data,
    # but is still added for consistency.
    edges_train = np.transpose(edges_train, [0, 2, 1])
    edges_train = np.reshape(edges_train, [-1, num_vertices ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    feat_valid = np.transpose(data_valid, [0, 3, 1, 2])
    edges_valid = np.transpose(edges_valid, [0, 2, 1])
    edges_valid = np.reshape(edges_valid, [-1, num_vertices ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    feat_test = np.transpose(data_test, [0, 3, 1, 2])
    edges_test = np.transpose(edges_test, [0, 2, 1])
    edges_test = np.reshape(edges_test, [-1, num_vertices ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_vertices, num_vertices)) - np.eye(num_vertices)),
        [num_vertices, num_vertices])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=eval_batch_size)
    test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)

    return (train_data_loader, valid_data_loader, test_data_loader, 
            data_train.shape[0], data_valid.shape[0], data_test.shape[0])


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def kl_categorical_uniform(preds, num_vertices, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) + 
                      torch.log(torch.tensor(float(num_edge_types))))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum((1, 2)) / num_vertices


def kl_gumbel(logits, num_vertices):
    """Computes the analytical kl(q(z|x)||p(z)) = u + exp(-u) - 1.
    q(z|x) is gumbel distributed with location (u) given by logits.
    p(z) is gumbel distributed with location zero.
    """
    kl_div = logits + torch.exp(-logits) - 1.0
    return kl_div.sum((1, 2)) / num_vertices


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum((1, 2, 3)) / target.size(1)


def sample_indep_edges(logits, is_edgesymmetric=False, tau=1.0, hard=False,
                       hard_with_grad=False):
    """Sample independent edges using the custom categorical estimator instead of Gumbel noise.

    Args:
        logits: Logits of shape (batch_size, n * (n - 1), edge_types).
        is_edgesymmetric: Whether or not the edges are undirected.
        tau: Temperature for softmax.
        hard: If True, return hard one-hot samples.
        hard_with_grad: If True, use a straight-through gradient estimator.
    
    Returns:
        A tuple (samples, edge_weights) where samples are one-hot representations.
        The edge_weights are set to zeros (as they are not used downstream).
    """
    if is_edgesymmetric:
        edge_types = logits.size(2)
        n = int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))
        reshaped_logits = logits.view(-1, n, n - 1, edge_types)
        reshaped_logits = reshaped_logits.transpose(1, 2)  # (bs, n-1, n, edge_types)
        vertices = torch.triu_indices(n-1, n, offset=1)
        edge_logits = reshaped_logits[:, vertices[0], vertices[1], :]
    else:
        edge_logits = logits

    # Compute probabilities from logits using softmax with temperature.
    probs = torch.softmax(edge_logits / tau, dim=-1)
    # Use the custom categorical estimator to sample indices.
    samples_indices = Categorical.apply(probs)  # shape: (..., 1)
    # Convert sampled indices to one-hot vectors.
    X = torch.zeros_like(probs).scatter_(-1, samples_indices, 1.0)
    if hard_with_grad:
        X = (X - probs).detach() + probs

    if is_edgesymmetric:
        samples = torch.zeros_like(reshaped_logits)
        samples[:, vertices[0], vertices[1], :] = X
        samples[:, vertices[1] - 1, vertices[0], :] = X
        # Flatten back to original shape.
        samples = samples.transpose(1, 2).contiguous().view(*logits.shape)
        # Dummy edge weights (not used downstream).
        edge_weights = torch.zeros_like(reshaped_logits)
        edge_weights[:, vertices[0], vertices[1]] = 0.0
        edge_weights[:, vertices[1] - 1, vertices[0]] = 0.0
        edge_weights = edge_weights.transpose(1, 2).contiguous().view(*logits.shape)
        return samples, edge_weights
    else:
        edge_weights = (edge_logits)
        return X, edge_weights


def sampling_edge_metrics(logits, target, sst, n, num_samples=1,
                          is_edgesymmetric=False, use_cpp=False):
    """Compute edge metrics by sampling num_samples many hard samples for each
    element in a batch of logits.
    """
    tiled_logits = logits.repeat(num_samples, 1, 1)
    if sst == "indep":
        samples, _ = sample_indep_edges(tiled_logits, is_edgesymmetric, hard=True)
    elif sst == "topk":
        samples, _ = sample_topk_from_logits(tiled_logits, n - 1, hard=True)
    elif sst == "tree":
        samples, _ = sample_tree_from_logits(tiled_logits, hard=True, use_cpp=use_cpp)
    else:
        raise ValueError(f"Stochastic Softmax Trick type {sst} is not valid!")

    edge_types = logits.size(2)
    if edge_types == 1:
        samples = torch.cat((1.0 - samples, samples), dim=-1)
    samples = samples.view(num_samples, logits.size(0), logits.size(1), 2)
    target = target.unsqueeze(0).unsqueeze(-1).repeat((1, 1, 1, 2))

    one = torch.tensor(1.0).cuda() if samples.is_cuda else torch.tensor(1.0)
    zero = torch.tensor(0.0).cuda() if samples.is_cuda else torch.tensor(0.0)

    tp = torch.where(samples * target == 1.0, one, zero).sum(-2)
    tn = torch.where(samples + target == 0.0, one, zero).sum(-2)
    fp = torch.where(samples - target == 1.0, one, zero).sum(-2)
    fn = torch.where(samples - target == -1.0, one, zero).sum(-2)

    accs = torch.mean((tp + tn) / (tp + tn + fp + fn), axis=(0, 1)).cpu().detach()
    precisions = torch.mean(tp / ( tp + fp), axis=(0, 1)).cpu().detach()
    recalls = torch.mean(tp / (tp + fn), axis=(0, 1)).cpu().detach()

    return accs.numpy(), precisions.numpy(), recalls.numpy()

def maybe_make_logits_symmetric(logits, symmeterize_logits):
    """Make logits symmetric wrt edges; logits_ij = logits_ji.
    """
    if symmeterize_logits:
        n =  int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))
        reshaped_logits = logits.view(-1, n, n-1, logits.size(-1))
        reshaped_logits = reshaped_logits.permute(0, 3, 2, 1) # (bs, -1, n-1, n)
        vertices = torch.triu_indices(n-1, n, offset=1)
        upper_tri = reshaped_logits[:, :, vertices[0], vertices[1]]
        lower_tri = reshaped_logits[:, :, vertices[1] - 1, vertices[0]]
        new_logits = (upper_tri + lower_tri) / 2.0
        symmetric_logits = torch.zeros_like(reshaped_logits)
        symmetric_logits[:, :, vertices[0], vertices[1]] = new_logits
        symmetric_logits[:, :, vertices[1] - 1, vertices[0]] = new_logits
        symmetric_logits = symmetric_logits.permute(0, 3, 2, 1).flatten(1, 2)
        return symmetric_logits
    else:
        return logits
