
"""
Steps:
0. Global weights (at i=0, this is initialized; at i>0, this has been FL trained)
1. Train gradients:
> local updates (train local w on local data)
> global update (train w on root data set)

2. Collect gradients and do FLTrust aggregation
> compute the Trust score: similarity score (per layer), do the ReLU, and then normalize
> Aggregate gradients: grad

3. Update the global weights
> The usual update rule: w = w_prev - lr*grad

"""

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np


def fltrust(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """

    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1

    # use the last gradient (server update) as the trusted source
    baseline = nd.array(param_list[-1]).squeeze() #We can just ge the current global nn parameters.
    # Q: How about the initial params, it should be somewhere ?
    cos_sim = []
    new_param_list = []

    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(
            nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0)  # relu
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9)  # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(
            param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))

    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size

