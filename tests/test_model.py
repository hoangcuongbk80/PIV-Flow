import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_warp_coordinate_direction(self):
        import torch
        from core.geometry import warp
        image = torch.arange(64.).reshape(1, 1, 8, 8)
        flow = torch.zeros(1, 2, 8, 8); flow[:, 0] = 1
        warped, valid = warp(image, flow)
        torch.testing.assert_close(warped[:, :, :, :-1], image[:, :, :, 1:])
        self.assertFalse(valid[:, :, :, -1].any())

    def test_gain_bound_and_degenerate(self):
        import torch
        from core.losses import utility_target
        a, b, gt = [torch.randn(2, 2, 5, 6) for _ in range(3)]
        y, gain, norm, valid, _, _ = utility_target(a, b, gt, torch.ones(2, 1, 5, 6, dtype=torch.bool))
        self.assertTrue((gain.abs() <= norm + 1e-6).all())
        self.assertTrue((y.abs() <= 1).all())
        result = utility_target(a, a, gt, valid)
        self.assertFalse(result[3].any()); self.assertTrue(torch.isfinite(result[0]).all())

    def test_selection_ties_and_zero_step(self):
        import torch
        from core.verifier import select_step
        delta = torch.ones(1, 2, 2, 2)
        q = [torch.zeros(1, 2, 2, 2) for _ in range(2)]
        action, _, feedback = select_step(q, delta)
        self.assertEqual(action.sum().item(), 0)
        q[0][:, 0] = .5; q[1][:, 0] = .25
        action, _, _ = select_step(q, delta)
        self.assertTrue((action == 1).all())
        action, _, feedback = select_step(q, delta * 0, policy="no_zero")
        self.assertEqual(action.sum().item(), 0); self.assertTrue(torch.isfinite(feedback).all())

    def test_reference_origin_and_padding(self):
        import torch
        from core.geometry import feature_reference
        gt = torch.ones(1, 2, 65, 67) * 8
        ref, valid = feature_reference(gt, torch.ones(1, 65, 67, dtype=torch.bool), (9, 9))
        torch.testing.assert_close(ref[0, :, 8, 8], torch.ones(2))
        self.assertTrue(valid.all())

    def test_replay_gradient_separation_and_forward(self):
        import torch
        from core import PIVFlow
        from core.losses import replay_loss, sequence_loss
        model = PIVFlow({"width": 32, "pretrained": False, "iters": 1})
        a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        outputs = [model(a, b, return_aux=True), model(a * .5, b * .7, return_aux=True)]
        self.assertEqual(outputs[0]["final"].shape, (1, 2, 64, 64))
        gt, valid = torch.zeros(1, 2, 64, 64), torch.ones(1, 64, 64, dtype=torch.bool)
        loss = replay_loss(model, outputs, gt, valid)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.verifier.parameters()))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("verifier.")))
        model.zero_grad(set_to_none=True)
        # A fresh graph is used for a separate flow-gradient check.
        output = model(a, b)
        sequence_loss(output, gt, valid).backward()
        self.assertTrue(any(p.grad is not None for p in model.flow_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.verifier.parameters()))

    def test_ordered_quantiles(self):
        import torch
        from core import PIVFlow
        model = PIVFlow({"width": 32, "pretrained": False, "iters": 1})
        output = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), return_aux=True)
        trace = output["traces"][0]
        q = model.verifier(output["evidence"], trace["anchor"], trace["full"])
        self.assertTrue((q[:, 0] <= q[:, 1]).all())
        self.assertTrue((q.abs() <= 1).all())

    def test_rgb_and_phase_transitions(self):
        import torch
        from core import PIVFlow
        model = PIVFlow({"width": 32, "iters": 1, "use_verifier": False, "policy": "full"})
        model.set_phase("verifier")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        model.set_phase("joint")
        result = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        self.assertTrue(torch.isfinite(result["final"]).all())
