# random transforms that can be used to augment training/test model performance #
import torch
import random

def random_shift(x, shift_range=30):
    # randomly shift an input torch tensor of size [N]. fill in new values with zeros
    # random translation is uniform within +- shift_range

    out = torch.zeros_like(x)

    shift = random.randint(-shift_range, shift_range)

    if shift > 0:
        out[shift:] = x[:-shift]
    elif shift < 0:
        out[:shift] = x[-shift:]
    else:
        return x.clone()

    return out

if __name__ == '__main__':
    x = torch.rand((360))

    random_shift(x, 30)
