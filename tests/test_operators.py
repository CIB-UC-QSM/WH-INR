import torch

from wh_inr.operators import build_dipole_kernel, dipole_forward, positive_laplacian


def test_dipole_operator_is_self_adjoint() -> None:
    torch.manual_seed(2)
    shape = (7, 8, 9)
    kernel = build_dipole_kernel(shape, (1.0, 1.2, 0.8), (0.2, 0.1, 1.0))
    left = torch.randn(shape)
    right = torch.randn(shape)
    lhs = torch.sum(dipole_forward(left, kernel) * right)
    rhs = torch.sum(left * dipole_forward(right, kernel))
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)
    assert kernel[0, 0, 0] == 0


def test_positive_laplacian_is_self_adjoint_and_annihilates_constants() -> None:
    torch.manual_seed(3)
    left = torch.randn(6, 7, 8)
    right = torch.randn(6, 7, 8)
    torch.testing.assert_close(
        torch.sum(positive_laplacian(left) * right),
        torch.sum(left * positive_laplacian(right)),
    )
    torch.testing.assert_close(positive_laplacian(torch.ones_like(left)), torch.zeros_like(left))

