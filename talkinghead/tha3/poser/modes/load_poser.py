import torch


def load_poser(model: str, device: torch.device, modelsdir="talkinghead/tha3/models"):
    print("Using the %s model." % model)
    if model == "standard_float":
        from tha3.poser.modes.standard_float import create_poser
        return create_poser(device, modelsdir=modelsdir)
    elif model == "standard_half":
        from tha3.poser.modes.standard_half import create_poser
        return create_poser(device, modelsdir=modelsdir)
    elif model == "separable_float":
        from tha3.poser.modes.separable_float import create_poser
        return create_poser(device, modelsdir=modelsdir)
    elif model == "separable_half":
        from tha3.poser.modes.separable_half import create_poser
        return create_poser(device, modelsdir=modelsdir)
    else:
        raise RuntimeError("Invalid model: '%s'" % model)
