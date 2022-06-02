import torch.nn as nn

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router] # eg. ['mask', 'task']

    for key in matched_keys:
        val = args[key]
        for depth, ( (f_args, g_args), routes ) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args  # eg. routed_args[depth] = ({'mask': tensor, 'task': 'i2t'}, {})


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route   # eg. router = {'mask':((True, False), (True, False),...), 'task':((True, False), (True, False),...)}
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):  # eg. kwargs = {'mask': tensor, 'task': 'i2t'}
        args = route_args(self.args_route, kwargs, len(self.layers)) # eg. args[depth] = ({'mask': tensor, 'task': 'i2t'}, {})
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x
