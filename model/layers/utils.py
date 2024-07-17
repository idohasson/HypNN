def layer_size(start, stop, n):
    skip = (stop - start) / n
    layer_sizes = [start]
    while len(layer_sizes) <= n:
        layer_sizes.append(layer_sizes[-1] + skip)
    layer_sizes = list(map(round, layer_sizes))
    layer_pairs = zip(layer_sizes[:-1], layer_sizes[1:])
    return list(layer_pairs)
