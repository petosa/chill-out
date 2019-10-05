def get_chain_thaw_policy(n_layers=8):
    # 1) Freeze every layer except the last (softmax) layer and train it.
    # 2) Freeze every layer except the first layer and train it.
    # 3) Freeze every layer except the second etc., until the second last layer.
    # 4) Unfreeze all layers and train entire model.

    layers = n_layers
    policy = [[False]*(layers-1) + [True]]
    policy += [[False] * (i-1) + [True] + [False] * (layers-i) for i in range(1, layers)]
    policy.append([True]*layers)
    return policy