def add_batch_norm_2d(network, weight_map, input, layer_name, EPS):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + EPS)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input = input,
                             mode = trt.ScaleMode.CHANNEL,
                             shift = shift,
                             scale = scale)