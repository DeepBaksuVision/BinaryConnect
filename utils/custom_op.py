import torch
import torch.nn.functional as F

class binary_linear_op(torch.autograd.Function):
    """
    Refer https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, mode="determistic"):

        # Binarization
        if mode == "determistic":
            # Deterministic method
            bin_weight = weight.sign()
            bin_weight[bin_weight == 0] = 1
        elif mode == "stochastic":
            # Stochastic method
            p = torch.sigmoid(weight)
            uniform_matrix = torch.empty(p.shape).uniform_(0, 1)
            bin_weight = (p >= uniform_matrix).type(torch.float32)
            bin_weight[bin_weight == 0] = -1.

        # Save input, binarized weight, bias in context object
        ctx.save_for_backward(input, bin_weight, bias)

        output = input.mm(bin_weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bin_weight, bias = ctx.saved_variables

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(bin_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class binary_conv_op(torch.autograd.Function):
    """
    Refer https://pytorch.org/docs/stable/notes/extending.html

    PyTorch Custom Con has a Bug
    Refer https://github.com/pytorch/pytorch/issues/16012

    Can not use padding & dilation option
    """

    @staticmethod
    def forward(ctx,
                input,
                weight,
                bias=None,
                mode="determistic",
                stride=1,
                padding=0,
                dilation=1,
                groups=1):

        # Binarization
        if mode == "determistic":
            # Deterministic method
            bin_weight = weight.sign()
            bin_weight[bin_weight == 0] = 1
        elif mode == "stochastic":
            # Stochastic method
            p = torch.sigmoid(weight)
            uniform_matrix = torch.empty(p.shape).uniform_(0, 1)
            bin_weight = (p >= uniform_matrix).type(torch.float32)
            bin_weight[bin_weight == 0] = -1.

        # Save input, binarized weight, bias in context object
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.save_for_backward(input, bin_weight, bias)

        output = F.conv2d(input, bin_weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bin_weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape,
                                                    bin_weight,
                                                    grad_output,
                                                    ctx.stride,
                                                    ctx.padding,
                                                    ctx.dilation,
                                                    ctx.groups)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input,
                                                      bin_weight.shape,
                                                      grad_output,
                                                      ctx.stride,
                                                      ctx.padding,
                                                      ctx.dilation,
                                                      ctx.groups)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None, None

        return grad_input, grad_weight, grad_bias


binary_linear = binary_linear_op.apply
binary_conv2d = binary_conv_op.apply


if __name__ == "__main__":
    """
    tmp_weight = torch.randn((3, 3), requires_grad=True)
    tmp_input = torch.randn((3, 3), requires_grad=True)

    output = binary_linear(tmp_input, tmp_weight)
    output.sum().backward()

    print("Tmp Input : {}".format(tmp_input))
    print("Tmp Weight : {}".format(tmp_weight))
    print("Output : {}".format(output))
    print("Grad Func : {}".format(output.grad_fn))
    print("Weights Grad : {}".format(tmp_weight.grad))
    print("Input Grad : {}".format(tmp_input.grad))
    """

    tmp_weight = torch.randn((1, 1, 4, 4), requires_grad=True)
    tmp_input = torch.randn((1, 1, 4, 4), requires_grad=True)

    output = binary_conv2d(tmp_input, tmp_weight, None, "determistic", 1, 0, 1, 1)
    output.sum().backward()
    print("Tmp Input : {}".format(tmp_input))
    print("Tmp Weight : {}".format(tmp_weight))
    print("Output : {}".format(output))
    print("Grad Func : {}".format(output.grad_fn))
    print("Weight Grad : {}".format(tmp_weight.grad))
    print("Input Grad : {}".format(tmp_input.grad))
