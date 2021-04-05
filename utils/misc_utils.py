def combine_first_ax(inp_tensor, keepdim=False):
    inp_shape = inp_tensor.shape
    if keepdim:
        return inp_tensor.view(1, inp_shape[0] * inp_shape[1], *inp_shape[2:])
    return inp_tensor.view(inp_shape[0] * inp_shape[1], *inp_shape[2:])


def uncombine_first_ax(inp_tensor, s0):
    "s0 is the size(0) intended, usually B"
    inp_shape = inp_tensor.shape
    size0 = inp_tensor.size(0)
    assert size0 % s0 == 0
    s1 = size0 // s0
    return inp_tensor.view(s0, s1, *inp_shape[1:])
