import time
from itertools import chain, combinations, permutations
import numpy as np

import torch
torch.set_printoptions(precision=32)



from core.kruskals.kruskals import get_tree
from core.kruskals.kruskals import kruskals_pytorch_batched
from core.kruskals.kruskals import kruskals_cpp_pytorch

EPS = torch.finfo(torch.float32).tiny

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
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_edges_from_vertices(vertices, num_vertices):
    idx = 0
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if i in vertices and j in vertices:
                edges.append(idx)
            idx = idx + 1
    return edges


def submatrix_index(n, i):
    bs = i.size(0)
    I = torch.ones((bs, n, n), dtype=bool)
    I[torch.arange(bs), i, :] = False
    I[torch.arange(bs), :, i] = False
    return I


def get_spanning_tree_marginals(logits, n):
    bs = logits.size(0)
    (i, j) = torch.triu_indices(n, n, offset=1).to(torch.device("cpu"))
    c = torch.max(logits, axis=-1, keepdims=True)[0]
    k = torch.argmax(logits, axis=-1).to(torch.device("cpu"))
    removei = i[k]

    weights = torch.exp(logits - c)

    W = torch.zeros(weights.size(0), n, n)
    W = W.cuda() if logits.is_cuda else W
    W[:, i, j] = weights
    W[:, j, i] = weights

    L = torch.diag_embed(W.sum(axis=-1)) - W
    subL = L[submatrix_index(n, removei)].view(bs, n - 1, n - 1)
    logzs = torch.slogdet(subL)[1]
    logzs = torch.sum(logzs + (n - 1) * c.flatten())
    sample = torch.autograd.grad(logzs, logits, create_graph=True)[0]
    return sample


def clip_range(x, max_range=np.inf):
    m = torch.max(x, axis=-1, keepdim=True)[0]
    return torch.max(x, -1.0 * torch.tensor(max_range) * torch.ones_like(x) + m)


def sample_tree_from_logits(logits, tau=1.0, hard=False, hard_with_grad=False,
                            edge_types=1, relaxation="exp_family_entropy",
                            max_range=np.inf, use_cpp=False):
    """
    Samples a spanning tree using the custom categorical estimator.
    
    Args:
        logits: Logits of shape (batch_size, n * (n - 1), 1).
               They represent a flattened, transposed adjacency matrix with the diagonals removed.
        tau: Temperature.
        hard: Whether to sample hard (discrete) trees.
        hard_with_grad: Whether to allow a straight-through gradient.
        edge_types: Number of edge types (1 or 2).
        relaxation: (Not used in this version; kept for API compatibility.)
        max_range: (Not used here.)
        use_cpp: Whether to use the C++ implementation for Kruskal's algorithm.
    
    Returns:
        A tuple (samples, edge_weights) with shapes matching the input logits.
    """
    # Determine number of vertices.
    n = int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))
    # Ensure only one edge type.
    assert logits.size(2) == 1
    # Reshape logits to adjacency-matrix format.
    reshaped_logits = logits.view(-1, n, n - 1)
    reshaped_logits = reshaped_logits.transpose(1, 2)  # (bs, n-1, n)
    
    # Get the upper-triangle indices.
    vertices = torch.triu_indices(n - 1, n, offset=1)
    edge_logits = reshaped_logits[:, vertices[0], vertices[1]]  # (bs, num_edges)

    # --- Replace Gumbel noise with custom categorical estimator ---
    # Compute probabilities from logits using softmax with temperature.
    probs = torch.softmax(edge_logits / tau, dim=-1)
    # Sample indices using the custom estimator.
    sampled_indices = Categorical.apply(probs)  # shape: (bs, num_edges, 1)
    # Convert indices to one-hot representation.
    binary_sample = torch.zeros_like(probs).scatter_(-1, sampled_indices, 1.0)
    # For hard sampling with gradient, use straight-through estimator.
    if hard_with_grad:
        binary_sample = (binary_sample - probs).detach() + probs
    # -------------------------------------------------------------

    # For spanning tree hard sampling, use the binary_sample as edge weights.
    # Expand vertices to match batch size.
    hard = True if hard_with_grad else hard  # enforce hard sampling here.
    tiled_vertices = vertices.transpose(0, 1).repeat((binary_sample.size(0), 1, 1)).float()
    tiled_vertices = tiled_vertices.cuda() if logits.is_cuda else tiled_vertices
    # Concatenate binary_sample (as weight) with edge indices.
    weights_and_edges = torch.cat([binary_sample.unsqueeze(-1), tiled_vertices], axis=-1)
    
    # Use Kruskal's algorithm (either C++ or PyTorch version) to extract a spanning tree.
    if use_cpp:
        samples = kruskals_cpp_pytorch(weights_and_edges.detach().cpu(), n)
        samples = samples.to("cuda") if logits.is_cuda else samples
    else:
        samples = kruskals_pytorch_batched(weights_and_edges, n)
    
    if edge_types == 2:
        null_edges = 1.0 - samples
        samples = torch.stack((null_edges, samples), dim=-1)
    else:
        samples = samples.unsqueeze(-1)
    
    hard_samples = samples
    if hard_with_grad:
        samples = (hard_samples - binary_sample).detach() + binary_sample
    
    # Reshape the samples back to the original flattened format.
    samples = samples.transpose(1, 2).contiguous().view(-1, n * (n - 1), edge_types)
    
    # For edge_weights, reshape binary_sample similarly (here, we simply propagate binary_sample).
    edge_weights_reshaped = torch.zeros_like(reshaped_logits)
    edge_weights_reshaped[:, vertices[0], vertices[1]] = probs
    edge_weights_reshaped[:, vertices[1] - 1, vertices[0]] = probs
    edge_weights = edge_weights_reshaped.transpose(1, 2).contiguous().view(logits.shape)
    
    return samples, edge_weights


def enumerate_spanning_trees(weights_and_edges, n):
    """
    Args:
        weights_and_edges: Shape (n * (n - 2), 3).
        n: Number of vertices.
    """
    probs = {}
    for edgeperm in permutations(weights_and_edges):
        edgeperm = torch.stack(edgeperm)
        tree = get_tree(edgeperm[:, 1:].int(), n)
        weights = edgeperm[:, 0]
        logprob = 0
        for i in range(len(weights)):
            logprob += weights[i] -  torch.logsumexp(weights[i:], dim=0)
        tree_str = "".join([str(x) for x in tree.flatten().int().numpy()])
        if tree_str in probs:
            probs[tree_str] = probs[tree_str] + torch.exp(logprob)
        else:
            probs[tree_str] = torch.exp(logprob)
    return probs


def compute_probs_for_tree(logits, use_gumbels=True):
    if use_gumbels:
        return logits
    n = int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))
    reshaped_logits = logits.view(-1, n, n - 1)
    reshaped_logits = reshaped_logits.transpose(1, 2)  # (bs, n-1, n)
    vertices = torch.triu_indices(n - 1, n, offset=1)
    edge_logits = reshaped_logits[:, vertices[0], vertices[1]]
    probs = []
    for weights in edge_logits:
      weights_and_edges = torch.Tensor(
          [list(e) for e in zip(weights, vertices[0], vertices[1])])
      p_dict = enumerate_spanning_trees(weights_and_edges, n)
      p = torch.tensor(list(p_dict.values()))
      probs.append(p)
    probs = torch.stack(probs)
    return probs


if __name__ == "__main__":
    ##################### Testing compute_probs_for_tree #####################
    bs = 1
    n = 4

    logits = torch.rand((bs, n * (n-1)))
    prob = compute_probs_for_tree(logits, use_gumbels=False)
    np.testing.assert_almost_equal(prob.sum(axis=-1).numpy(), np.ones((bs,)))
