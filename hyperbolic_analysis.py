from sqlite3 import connect, OperationalError
import autograd.numpy as np
import utils
import enums
from scipy.linalg import inv
from scipy.spatial.distance import euclidean
from math import e, inf
from typing import Dict, List
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pymanopt
import spectrograms
import cov_based_analysis


torch.manual_seed(0)


class KNNNode:
    def __init__(
        self,
        val: np.array,
        label: str,
        edges: List["KNNEdge"],
        knn: List["KNNNode"],
    ):
        self.val = val
        self.label = label
        self.edges = edges
        self.knn = knn


class KNNEdge:
    def __init__(
        self,
        points_to: KNNNode,
        weight: float,  # Gaussian kernel weight
    ):
        self.points_to = points_to
        self.weight = weight


class InterpSettings:
    def __init__(
        self,
        lang1: enums.RomanceLanguages,
        lang2: enums.RomanceLanguages,
        speaker1: enums.Speakers,
        speaker2: enums.Speakers,
        digit: int,
    ):
        self.digit = digit
        self.lang1 = lang1
        self.lang2 = lang2
        self.speaker1 = speaker1
        self.speaker2 = speaker2


######################################################################################
######################################################################################
######################################################################################
# NOTE: The below code is pulled from the GitHub repository for Klimovskaia          #
######  et al.'s ``Poincare maps for analyzing complex hierarchies in single-cell    #
######  data'' (2020) with few to no modifications per function. This is the case    #
######  with all the code that follows until otherwise indicated. This code is       #
######  generally for the purpose of learning Poincare embeddings and interplating   #
######  between points in the disk. Here's a link to the license under which their   #
######  code was distributed: https://github.com/facebookresearch/PoincareMaps/blob/ #
######  main/LICENSE. Some changes were made to the code from its original state.    #                                                            
######################################################################################
######################################################################################
######################################################################################

eps = 1e-5
boundary = 1 - eps
spten_t = torch.sparse.FloatTensor


def poincare_grad(p, d_p):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = torch.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1, keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = torch.sum(p.data**2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def euclidean_retraction(p, d_p, lr):
    p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(
        self, params, lr=1e-3, rgrad=poincare_grad, retraction=euclidean_retraction
    ):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]
                d_p = group["rgrad"](p, d_p)
                group["retraction"](p, d_p, lr)

        return loss


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)

        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv


class PoincareOptions:
    def __init__(
        self,
        debugplot=False,
        epochs=500,
        batchsize=-1,
        lr=0.1,
        burnin=500,
        lrm=1.0,
        earlystop=0.0001,
        cuda=0,
    ):
        self.debugplot = debugplot
        self.batchsize = batchsize
        self.epochs = epochs
        self.lr = lr
        self.lrm = lrm
        self.burnin = burnin
        self.debugplot = debugplot


class PoincareEmbedding(nn.Module):
    def __init__(
        self,
        size,
        dim,
        dist=PoincareDistance,
        max_norm=1,
        Qdist="laplace",
        lossfn="klSym",
        gamma=1.0,
        cuda=0,
    ):
        super(PoincareEmbedding, self).__init__()

        self.dim = dim
        self.size = size
        self.lt = nn.Embedding(size, dim, max_norm=max_norm)
        self.lt.weight.data.uniform_(-1e-4, 1e-4)
        self.dist = dist
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == "kl":
            self.lossfn = nn.KLDivLoss()
        elif lossfn == "klSym":
            self.lossfn = klSym
        elif lossfn == "mse":
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()

    def forward(self, inputs):
        embs_all = self.lt.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.lt(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)

        if self.lossfnname == "kl":
            if self.Qdist == "laplace":
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == "gaussian":
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == "student":
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == "klSym":
            if self.Qdist == "laplace":
                return self.sm(-self.gamma * dists)
            elif self.Qdist == "gaussian":
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == "student":
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == "mse":
            return self.sm(-self.gamma * dists)
        else:
            raise NotImplementedError


def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = 1 - sqnormx
    beta = 1 - sqnormv
    z = 1 + 2 * sqdist / (alpha * beta)
    a = (
        ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))
        .unsqueeze(-1)
        .expand_as(x)
    )
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


def train(
    model,
    data,
    optimizer,
    args,
    fout=None,
    labels=None,
    earlystop=0.0,
    color_dict=None,
):
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    pbar = tqdm(range(args.epochs), ncols=80)

    n_iter = 0
    epoch_loss = []
    earlystop_count = 0
    for epoch in pbar:
        grad_norm = []

        # determine learning rate
        lr = args.lr
        if epoch < args.burnin:
            lr = lr * args.lrm

        epoch_error = 0
        for inputs, targets in loader:
            loss = model.lossfn(model(inputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lr=lr)

            epoch_error += loss.item()

            grad_norm.append(model.lt.weight.grad.data.norm().item())

            n_iter += 1

        epoch_error /= len(loader)
        epoch_loss.append(epoch_error)
        pbar.set_description("loss: {:.5f}".format(epoch_error))

        if epoch > 10:
            delta = abs(epoch_loss[epoch] - epoch_loss[epoch - 1])
            if delta < earlystop:
                earlystop_count += 1
            if earlystop_count > 50:
                print(f"\nStopped at epoch {epoch}")
                break

        if args.debugplot:
            if (epoch % args.debugplot) == 0:
                d = model.lt.weight.cpu().detach().numpy()
                # NOTE: below commented out from code's original state
                # titlename = "epoch: {:d}, loss: {:.3e}".format(
                #     epoch, np.mean(epoch_loss)
                # )

                # NOTE: below commented out from code's original state
                # if epoch > 5:
                #     plotPoincareDisc(np.transpose(d), labels, fout, titlename, color_dict=color_dict)
                #     np.savetxt(fout + '.csv', d, delimiter=",")

                ball_norm = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
                if np.max(ball_norm) > 1.001:
                    print("The learning rate is too high.")

                delta = abs(epoch_loss[epoch] - epoch_loss[epoch - 1])

                # NOTE: below commented out from code's original state
                # plot_training(epoch_loss,
                # title_name=f'd={delta:.2e}',
                # file_name=fout+'_loss', d1=4, d2=4)

                # print(f"{epoch}: time={elapsed:.3f}, "
                #       f"loss = {np.mean(epoch_loss):.3e}, "
                #       f"grad_norm = {np.mean(grad_norm):.3e}, "
                #       f"max_norm = {np.max(ball_norm):.4f}, "
                #       f"mean_norm = {np.mean(ball_norm):.4f}")

    delta = abs(epoch_loss[epoch] - epoch_loss[epoch - 1])

    # NOTE: below commented out from code's original state
    # plot_training(epoch_loss, title_name=f'd={delta:.2e}', file_name=fout+'_loss', d1=4, d2=4)

    return model.lt.weight.cpu().detach().numpy(), epoch_error, epoch


def get_geodesic_parameters(u, v):
    nu = u[0] ** 2 + u[1] ** 2
    nv = v[0] ** 2 + v[1] ** 2
    a = (u[1] * nv - v[1] * nu + u[1] - v[1]) / (u[0] * v[1] - u[1] * v[0])
    b = (v[0] * nu - u[0] * nv + v[0] - u[0]) / (u[0] * v[1] - u[1] * v[0])
    return a, b


def poincare_linspace(
    u,
    v,
    n_points=utils.INTERP_NO,
):
    # If u is (0, 0)
    if np.sum(u**2) == 0:
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x
        if v[0] != 0:
            k = v[1] / v[0]
            interpolated[:, 1] = k * interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, v[1], num=n_points)

    # If v is (0, 0)
    elif np.sum(v**2) == 0:
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x
        if u[0] != 0:
            k = u[1] / u[0]
            interpolated[:, 1] = k * interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, u[1], num=n_points)

    else:
        a, b = get_geodesic_parameters(u, v)

        x = np.linspace(u[0], v[0], num=n_points)

        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x

        r = a**2 / 4 + b**2 / 4 - 1
        y_1 = -b / 2 + np.sqrt(r - (x + a / 2) ** 2)
        y_2 = -b / 2 - np.sqrt(r - (x + a / 2) ** 2)

        if max(x**2 + y_1**2) > 1:
            interpolated[:, 1] = y_2
        elif max(x**2 + y_2**2) > 1:
            interpolated[:, 1] = y_1
        elif (np.mean(y_1) <= max(u[1], v[1])) and (np.mean(y_1) >= min(u[1], v[1])):
            interpolated[:, 1] = y_1
        else:
            interpolated[:, 1] = y_2

    return interpolated


######################################################################################
######################################################################################
######################################################################################
# NOTE: End of code which is largely unmodified from its definition in Klimovskaia   #
######  et al.'s GitHub repository. Their repository can be found at                 #
######  https://github.com/facebookresearch/PoincareMaps.                            #
######################################################################################
######################################################################################
######################################################################################


def _get_connected_component_bfs(
    node: "KNNNode",
) -> List[str]:
    """_summary_

    Args:
        node (KNNNode): The kNN graph node at which to begin searching for a connected component
                        in the graph.

    Returns:
        List[str]: A list of graph node labels corresponding to the nodes in the discovered
                   connected component.
    """
    # NOTE: Implementation assumes uniquely-labeled nodes
    frontier = [e.points_to for e in node.edges]
    visited = [node.label]

    while frontier:
        n = frontier.pop()
        n_neighbors = [e.points_to for e in n.edges if e.points_to.label not in visited]
        for neighbor in n_neighbors:
            visited.append(neighbor.label)
            frontier = [neighbor] + frontier

    return visited


def construct_knn_graph(
    digit: int,
    k: int,
    kernel_sigma: float = 100,
    interp_settings: InterpSettings = None,
) -> Dict[str, KNNNode]:
    """Constructs a kNN graph out of our speech data. This process follows that outlined in
    Klimovskaia et al.'s ``Poincare maps for analyzing complex hierarchies in single-cell
    data'' (2020).

    Args:
        digit (int): The digit for which the graph is being constructed.
        k (int): The number of mutual nearest neighbors to represent for each node on the
                 graph.
        kernel_sigma (float, optional): Hyperparameter associated with edge weighting. Defaults
                                        to 100.
        interp_settings (InterpSettings, optional): Information provided if and only if the graph
                                                    seeks to embed covariance-based interpolations.
                                                    Defaults to None.

    Returns:
        Dict[str, KNNNode]: A dictionary with key-value pairs of node labels and KNNNode objects that
                            captures the entire graph.
    """
    # Get all smoothed + time aligned spectrogram arrays for given digit
    arrs = spectrograms.get_digit_spectrogram_arrays(
        db_type=enums._DBType.TIME_ALIGNED_SMOOTHED_SPECTROGRAM.value,
        digit=digit,
    )

    if interp_settings is not None:
        arrs += cov_based_analysis.interspeaker_interp(
            lang1=interp_settings.lang1,
            lang2=interp_settings.lang2,
            speaker1=interp_settings.speaker1,
            speaker2=interp_settings.speaker2,
            digit=digit,
        )

    # NOTE: We're standardizing the spectrograms here
    vecs = [
        (label, (utils._standard_scale_array(arr=arr)).flatten()) for label, arr in arrs
    ]

    # Define KNN graph
    graph = {
        label: KNNNode(val=val, label=label, edges=[], knn=[]) for label, val in vecs
    }

    # Mark keys between which we're interpolating
    if interp_settings:
        key1 = interp_settings.lang1.value + interp_settings.speaker1.value
        key2 = interp_settings.lang2.value + interp_settings.speaker2.value

        # NOTE: Using two loops here so that we're not replacing keys during iteration
        for key in graph:
            if key[:4] == key1:
                new_label = key + "_anchor"
                graph[new_label] = graph.pop(key)
                graph[new_label].label = new_label
                break

        for key in graph:
            if key[:4] == key2:
                new_label = key + "_anchor"
                graph[new_label] = graph.pop(key)
                graph[new_label].label = new_label
                break

    # Get k nearest neighbors for each node
    for node_key in graph:
        graph[node_key].knn = [
            x[1]
            for x in sorted(
                [
                    (
                        euclidean(graph[node_key].val, graph[other_node_key].val),
                        graph[other_node_key],
                    )
                    for other_node_key in graph
                    if graph[node_key] != graph[other_node_key]
                ],
                key=lambda x: x[0],
            )[:k]
        ]

    # Construct bidirectional edges in graph iff two given nodes are within
    # k nearest neighbors of each other
    for node_key_i in graph:
        for node_j in graph[node_key_i].knn:
            # If node i and node j are mutual KNNs and an edge hasn't already
            # been constructed between the two
            if (
                graph[node_key_i] in node_j.knn
                and graph[node_key_i] not in [x.points_to for x in node_j.edges]
                and node_j not in [x.points_to for x in graph[node_key_i].edges]
            ):
                weight = e ** (
                    -(euclidean(graph[node_key_i].val, node_j.val) ** 2)
                    / (2 * kernel_sigma**2)
                )

                i_to_j_edge = KNNEdge(points_to=node_j, weight=weight)
                j_to_i_edge = KNNEdge(points_to=graph[node_key_i], weight=weight)
                graph[node_key_i].edges.append(i_to_j_edge)
                node_j.edges.append(j_to_i_edge)

    conn_comp = _get_connected_component_bfs(list(graph.values())[0])

    # If there are disconnected components
    if len(conn_comp) != len(graph):
        disconn_comps = [conn_comp]
        nodes_left = [x for x in graph if x not in conn_comp]

        while nodes_left:
            conn_comp = _get_connected_component_bfs(graph[nodes_left[0]])
            disconn_comps.append(conn_comp)
            nodes_left = [x for x in nodes_left if x not in conn_comp]

        # For each pair of disconnected components (if any), connect the two components
        # with minimum-size edge
        for i in range(len(disconn_comps)):
            for j in range(i + 1, len(disconn_comps)):
                # Lists of nodes themselves
                disconn_comp_1 = [graph[node_key] for node_key in disconn_comps[i]]
                disconn_comp_2 = [graph[node_key] for node_key in disconn_comps[j]]

                min_pair = ["", ""]
                min_val = inf

                for dc_1 in disconn_comp_1:
                    for dc_2 in disconn_comp_2:
                        euc = euclidean(dc_1.val, dc_2.val)

                        if euc < min_val:
                            min_val = euc
                            min_pair = [dc_1.label, dc_2.label]

                weight = e ** (-(min_val**2) / (2 * kernel_sigma**2))

                k_to_p_edge = KNNEdge(points_to=graph[min_pair[1]], weight=weight)
                p_to_k_edge = KNNEdge(points_to=graph[min_pair[0]], weight=weight)

                graph[min_pair[0]].edges.append(k_to_p_edge)
                graph[min_pair[1]].edges.append(p_to_k_edge)

    return graph


def construct_RFA_matrix(
    graph: KNNNode,
) -> torch.tensor:
    """Definition of the Relative Forest Accessibility (RFA) matrix as outlined in
    Klimovskaia et al.'s ``Poincare maps for analyzing complex hierarchies in single-cell
    data'' (2020).

    Args:
        graph (KNNNode): The kNN graph for which to get the RFA matrix.

    Returns:
        torch.tensor: The RFA matrix.
    """
    graph_len = len(graph)

    # Node-to-index dictionary for nodes in graph
    node_to_index = {list(graph.keys())[i]: i for i in range(graph_len)}

    # Define adjacency matrix A
    A = np.zeros((graph_len, graph_len))

    for node in graph:
        i = node_to_index[node]
        adj_node_indices_and_weights = [
            (node_to_index[edge.points_to.label], edge.weight)
            for edge in graph[node].edges
        ]
        for j, w in adj_node_indices_and_weights:
            A[i, j] += w

    # Define degree matrix D
    D = np.zeros((graph_len, graph_len))

    for i in range(graph_len):
        D[i, i] = A[i].sum()

    # Define Laplacian matrix L
    L = D - A

    # Return RFA
    return torch.tensor(inv(np.identity(graph_len) + L))


# NOTE: Many parts of this code are pulled from Klimovskaia et al.'s repo
def get_embeddings(
    graph: KNNNode,
    gamma=2.0,
    epochs=500,  # NOTE: This was at 300 originally
    debugplot=False,
    batchsize=-1,
    lr=0.1,
    burnin=500,
    lrm=1.0,
    earlystop=0.0001,
    cuda=0,
) -> np.array:
    """Learns Poincare embeddings for the data represented in the kNN graph
    provided. Much of this function's implementation is from Klimovskaia
    et al.'s repository for ``Poincare maps for analyzing complex hierarchies
    in single-cell data'' (2020).

    Args:
        graph (KNNNode): The kNN graph for which the Poincare embeddings are to be
                         learned.

    Returns:
        np.array: The Poincare embeddings.
    """
    # NOTE: Below is Facebook Research implementation
    P = construct_RFA_matrix(graph=graph)
    indices = torch.arange(len(P))

    if batchsize < 0:
        batchsize = min(512, int(len(P) / 10))

    dataset = TensorDataset(indices, P)

    # instantiate our Embedding predictor
    predictor = PoincareEmbedding(
        len(dataset),
        2,
        dist=PoincareDistance,
        max_norm=1,
        Qdist="laplace",
        lossfn="klSym",
        gamma=gamma,
        cuda=cuda,
    )

    optimizer = RiemannianSGD(predictor.parameters(), lr=lr)

    opt = PoincareOptions(
        debugplot=debugplot,
        batchsize=batchsize,
        lr=lr,
        burnin=burnin,
        lrm=lrm,
        earlystop=earlystop,
        cuda=cuda,
        epochs=epochs,
    )

    embeddings, _, _ = train(
        predictor,
        dataset,
        optimizer,
        opt,
    )

    embedding_dict = {
        "FR": [],
        "IT": [],
        "PO": [],
        "SA": [],
        "SI": [],
        "in": [],
        "anchor": [],
    }

    counter = 0
    for key in graph:
        if key[:2] == "FR":
            embedding_dict["FR"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        elif key[:2] == "IT":
            embedding_dict["IT"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        elif key[:2] == "PO":
            embedding_dict["PO"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        elif key[:2] == "SA":
            embedding_dict["SA"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        elif key[:2] == "SI":
            embedding_dict["SI"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        elif key[:2] == "in":
            embedding_dict["in"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    "",
                )
            )

        # Note points between which we're interpolating; note this isn't elif
        if key[-6:] == "anchor":
            embedding_dict["anchor"].append(
                (
                    np.array([embeddings[counter][0], embeddings[counter][1]]),
                    graph[key].val,
                    key[:4],
                )
            )

        counter += 1

    return embedding_dict


def get_poincare_centroid_lang(
    embedding_dict: Dict[str],
    lang: enums.RomanceLanguages,
) -> np.array:
    """Gets the centroids for a given language from a given embedding dictionary.

    Args:
        embedding_dict (Dict[str]): An embedding dictionary as generated by
                                    hyperbolic_analysis.py > get_embeddings.
        lang (enums.RomanceLanguages): The language for which to calculate
                                       centroids.

    Returns:
        np.array: The centroid.
    """
    lang = lang.value

    lang_points = [x[0] for x in embedding_dict[lang]]

    manifold = pymanopt.manifolds.hyperbolic.PoincareBall(
        n=2,
    )

    @pymanopt.function.autograd(manifold)
    def cost(x):
        return sum([utils._poincare_distance_basic(x, pt) ** 2 for pt in lang_points])

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()

    result = optimizer.run(problem)

    return result.point


def get_poincare_centroids(
    embedding_dict: Dict[str],
) -> Dict[str]:
    """Gets the centroids for all languages from a given embedding dictionary.

    Args:
        embedding_dict (Dict[str]): An embedding dictionary as generated by
                                    hyperbolic_analysis.py > get_embeddings.

    Returns:
        Dict[str]: A dictionary of centroids.
    """
    centroid_dict = {
        "FR": None,
        "IT": None,
        "PO": None,
        "SA": None,
        "SI": None,
    }

    FR_centroid_x, FR_centroid_y = get_poincare_centroid_lang(
        embedding_dict, enums.RomanceLanguages.FRENCH
    )
    IT_centroid_x, IT_centroid_y = get_poincare_centroid_lang(
        embedding_dict, enums.RomanceLanguages.ITALIAN
    )
    PO_centroid_x, PO_centroid_y = get_poincare_centroid_lang(
        embedding_dict, enums.RomanceLanguages.PORTUGUESE
    )
    SA_centroid_x, SA_centroid_y = get_poincare_centroid_lang(
        embedding_dict, enums.RomanceLanguages.AMERICAN_SPANISH
    )
    SI_centroid_x, SI_centroid_y = get_poincare_centroid_lang(
        embedding_dict, enums.RomanceLanguages.IBERIAN_SPANISH
    )

    centroid_dict["FR"] = np.array([FR_centroid_x, FR_centroid_y])
    centroid_dict["IT"] = np.array([IT_centroid_x, IT_centroid_y])
    centroid_dict["PO"] = np.array([PO_centroid_x, PO_centroid_y])
    centroid_dict["SA"] = np.array([SA_centroid_x, SA_centroid_y])
    centroid_dict["SI"] = np.array([SI_centroid_x, SI_centroid_y])

    return centroid_dict


def rotate_disk(
    centroids: Dict[str],
    disk: Dict[str],
) -> Dict[str]:
    """Rotates disk such that

    Args:
        centroids (Dict[str]): A dictionary of per-language embeddings in the disk.
        disk (Dict[str]): Another dictionary of per-language embeddings in the disk,
                          this one to be rotated with respect to centroids to minimize
                          the distance between their like points.

    Returns:
        Dict[str]: A dictionary containing the points of the rotated disk.
    """
    obj_r = {
        "FR": euclidean(disk["FR"], np.zeros(2)),
        "IT": euclidean(disk["IT"], np.zeros(2)),
        "PO": euclidean(disk["PO"], np.zeros(2)),
        "SA": euclidean(disk["SA"], np.zeros(2)),
        "SI": euclidean(disk["SI"], np.zeros(2)),
    }

    obj_prior_theta = {
        "FR": utils._recover_theta(disk["FR"]),
        "IT": utils._recover_theta(disk["IT"]),
        "PO": utils._recover_theta(disk["PO"]),
        "SA": utils._recover_theta(disk["SA"]),
        "SI": utils._recover_theta(disk["SI"]),
    }

    manifold = pymanopt.manifolds.sphere.Sphere(2)

    # NOTE: Grabbing the first element of each theta point so we're
    #       effectively optimizing over theta \in [-1, 1]
    @pymanopt.function.autograd(manifold)
    def rotate_cost_1(theta):
        return utils._theta_cost_function(
            theta=theta,
            offset=1 * np.pi / 4,
            centroids=centroids,
            obj_r=obj_r,
            obj_prior_theta=obj_prior_theta,
        )

    @pymanopt.function.autograd(manifold)
    def rotate_cost_2(theta):
        return utils._theta_cost_function(
            theta=theta,
            offset=3 * np.pi / 4,
            centroids=centroids,
            obj_r=obj_r,
            obj_prior_theta=obj_prior_theta,
        )

    @pymanopt.function.autograd(manifold)
    def rotate_cost_3(theta):
        return utils._theta_cost_function(
            theta=theta,
            offset=5 * np.pi / 4,
            centroids=centroids,
            obj_r=obj_r,
            obj_prior_theta=obj_prior_theta,
        )

    @pymanopt.function.autograd(manifold)
    def rotate_cost_4(theta):
        return utils._theta_cost_function(
            theta=theta,
            offset=7 * np.pi / 4,
            centroids=centroids,
            obj_r=obj_r,
            obj_prior_theta=obj_prior_theta,
        )

    theta_dict = {
        1: pymanopt.optimizers.SteepestDescent()
        .run(pymanopt.Problem(manifold, rotate_cost_1))
        .point[0]
        + np.pi / 4,
        2: pymanopt.optimizers.SteepestDescent()
        .run(pymanopt.Problem(manifold, rotate_cost_2))
        .point[0]
        + 3 * np.pi / 4,
        3: pymanopt.optimizers.SteepestDescent()
        .run(pymanopt.Problem(manifold, rotate_cost_3))
        .point[0]
        + 5 * np.pi / 4,
        4: pymanopt.optimizers.SteepestDescent()
        .run(pymanopt.Problem(manifold, rotate_cost_4))
        .point[0]
        + 7 * np.pi / 4,
    }

    min_dist_sum = np.inf
    min_key = -1

    for key in theta_dict:
        dist_sum = sum(
            [
                utils._poincare_distance_basic(
                    centroids[lang],
                    np.array(
                        [
                            obj_r[lang]
                            * np.cos(obj_prior_theta[lang] + theta_dict[key]),
                            obj_r[lang]
                            * np.sin(obj_prior_theta[lang] + theta_dict[key]),
                        ]
                    ),
                )
                ** 2
                for lang in utils.langs
            ]
        )

        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            min_key = key

    min_theta = theta_dict[min_key]

    rotated_obj_disk = {
        lang: np.array(
            [
                obj_r[lang] * np.cos(obj_prior_theta[lang] + min_theta),
                obj_r[lang] * np.sin(obj_prior_theta[lang] + min_theta),
            ]
        )
        for lang in utils.langs
    }

    return rotated_obj_disk


def align_all_digit_disks(
    k: int,
    exclude_digits: List[int] = [],
):
    """Aligns the disks representing the centroids for all language/digit pairs,
    effectively generating a hyperbolic language space.

    Args:
        k (int): The k value used to construct the disks' underlying kNN graphs.
        exclude_digits (List[int], optional): A list of digits to exclude from
                                              alignment. Defaults to [].

    Returns:
        Dict[str]: A dictionary representing the centroids in each language
                   post-alignment.
    """
    digits = [x for x in range(1, 11) if x not in exclude_digits]

    graphs = {digit: construct_knn_graph(digit=digit, k=k) for digit in digits}

    embeddings = {digit: get_embeddings(graph=graphs[digit]) for digit in digits}

    centroids = {digit: get_poincare_centroids(embeddings[digit]) for digit in digits}

    # Currently using average Poincare distance from origin as relevant metric
    cloud_size = {
        digit: np.mean(
            [
                utils._poincare_distance_basic(np.array([0, 0]), centroids[digit][lang])
                for lang in utils.langs
            ]
        )
        for digit in digits
    }

    # Sort cloud size by descending value
    cloud_size = {
        k: v
        for k, v in sorted(cloud_size.items(), key=lambda item: item[1], reverse=True)
    }

    all_points = {lang: [] for lang in utils.langs}

    # Initialize accum_centroids with largest cloud_size digit + take it out of dict
    max_key = max(cloud_size, key=cloud_size.get)
    accum_centroids = centroids[max_key]
    del cloud_size[max_key]

    # Iterate through the digits in descending order of cloud_size
    for dig in cloud_size.keys():
        # Rotate this digit's disk wrt accumulated centroids
        rotated_centroids = rotate_disk(
            centroids=accum_centroids,
            disk=centroids[dig],
        )

        # Add rotated centroids to accum_points
        for lang in utils.langs:
            all_points[lang].append((rotated_centroids[lang], None))

        # Update accum centroids
        accum_centroids = {
            lang: (accum_centroids[lang] + rotated_centroids[lang]) / 2
            for lang in utils.langs
        }

    return all_points
