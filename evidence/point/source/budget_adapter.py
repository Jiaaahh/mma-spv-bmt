"""Development primitive, not a complete Point-SPV reproduction.

Hard forward budget with sigmoid surrogate gradients; ties use index order.
The surrogate is an adaptation choice, not the derivative of hard ranking.
"""
import torch


def fixed_budget(logits, budget=64):
    if logits.ndim != 2 or logits.shape[1] != 256:
        raise ValueError('Expected [batch, 256] electrode logits')
    if not torch.isfinite(logits).all() or not 0 < budget <= 256:
        raise ValueError('Invalid logits or budget')
    indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :budget]
    hard = torch.zeros_like(logits).scatter_(1, indices, 1)
    soft = logits.sigmoid()
    return hard + (soft - soft.detach())


def render(stimulation, kernels):
    if kernels.shape != (256, 96, 96):
        raise ValueError('Expected original 256 x 96 x 96 kernel bank')
    return torch.einsum('bn,nhw->bhw', stimulation, kernels).unsqueeze(1)


if __name__ == '__main__':
    import json
    from pathlib import Path
    torch.manual_seed(17)
    logits = torch.randn(4, 256, requires_grad=True)
    stimulation = fixed_budget(logits)
    assert torch.equal(stimulation.sum(1), torch.full((4,), 64.))
    assert ((stimulation == 0) | (stimulation == 1)).all()
    tied = fixed_budget(torch.zeros(1, 256))
    assert tied[0, :64].sum() == 64 and tied[0, 64:].sum() == 0
    kernels = torch.rand(256, 96, 96)
    kernels /= kernels.sum((1, 2), keepdim=True)
    image = render(stimulation, kernels)
    assert torch.allclose(image.sum((1, 2, 3)), torch.full((4,), 64.), atol=1e-4)
    image.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0
    rejected = False
    try:
        fixed_budget(torch.full((1, 256), float('nan')))
    except ValueError:
        rejected = True
    assert rejected
    result = dict(status='passed', budget=64, binary=True, tie_order='index ascending',
                  finite_nonzero_gradient=True, nonfinite_rejected=True,
                  scope='Synthetic kernels only; original-renderer parity and training not yet tested')
    Path(__file__).with_name('budget_adapter_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
