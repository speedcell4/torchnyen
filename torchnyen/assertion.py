from itertools import zip_longest
from typing import Sequence

import torch
from torch import Tensor, testing

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def assert_close(actual: Tensor | Sequence[Tensor], expected: Tensor | Sequence[Tensor], **kwargs):
    if torch.is_tensor(actual):
        actual = (actual,)
    if torch.is_tensor(expected):
        expected = (expected,)

    for actual, expected in zip_longest(actual, expected, fillvalue=None):
        testing.assert_close(actual=actual, expected=expected, **kwargs)


def assert_grad_close(actual: Tensor, expected: Tensor, inputs: Sequence[Tensor], **kwargs):
    assert actual.requires_grad
    assert expected.requires_grad

    grad_outputs = torch.randn_like(actual)

    actual_grad = torch.autograd.grad(
        actual, inputs, grad_outputs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    expected_grad = torch.autograd.grad(
        expected, inputs, grad_outputs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    assert_close(actual=actual_grad, expected=expected_grad, **kwargs)
