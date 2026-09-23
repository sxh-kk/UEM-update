from types import SimpleNamespace

import torch

from egorecover.codec import BodyState
from egorecover.fk import FixedShapeFK


def test_fk_uses_bootstrap_shape_and_predicted_pelvis_only():
    class RecordingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("parents", torch.tensor([-1] + [0] * 21))
            self.register_buffer("left_hand_components", torch.eye(45)[:12])
            self.register_buffer("right_hand_components", torch.eye(45)[:12])
            self.last = None

        def forward(self, *, betas, transl=None, **kwargs):
            self.last = {"betas": betas.clone(), "transl": transl, **kwargs}
            root = betas[:, :3]
            if transl is not None:
                root = root + transl
            return SimpleNamespace(joints=root[:, None].expand(-1, 22, -1))

    smpl = RecordingLayer()
    bootstrap = torch.arange(10, dtype=torch.float32) * 0.1
    fk = FixedShapeFK(smpl, bootstrap)
    physical = torch.eye(4).repeat(2, 22, 1, 1)
    physical[:, 0, :3, 3] = torch.tensor([[3.0, 4.0, 5.0], [8.0, 9.0, 10.0]])
    physical[:, 0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    state = BodyState(physical, torch.eye(4).repeat(2, 1, 1), torch.zeros(2, 36))
    first = fk(state)
    torch.testing.assert_close(first[:, 0], physical[:, 0, :3, 3])
    assert torch.equal(smpl.last["betas"], bootstrap.expand(2, -1))
    state.auxiliary[:, -10:] = 12345  # Predicted per-frame beta never replaces beta_boot.
    assert torch.equal(first, fk(state))
    assert smpl.last["body_pose"].shape == (2, 21, 3, 3)
    torch.testing.assert_close(smpl.last["global_orient"], physical[:, 0, :3, :3])
    torch.testing.assert_close(
        smpl.last["body_pose"], physical[:, 0, :3, :3].transpose(-1, -2)[:, None].expand(-1, 21, -1, -1)
    )
    assert smpl.last["left_hand_pose"].shape == (2, 15, 3, 3)
    fk.train()
    assert not smpl.training
