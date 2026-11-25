"""Check rare-class emphasis and equivalence to full-batch weighted training."""
import unittest

import torch
from torch.nn import functional as F

from train import get_class_weights, classification_loss_sum, normalize_gradients


class ClassWeightTest(unittest.TestCase):
    def test_balancing_equalizes_total_class_contribution(self):
        counts = torch.tensor([2, 4, 10, 20, 4])
        labels = torch.repeat_interleave(torch.arange(5), counts)
        weights = get_class_weights(labels.numpy())
        torch.testing.assert_close(weights * counts, torch.full((5,), 8.0))
        self.assertGreater(weights[0], weights[3])
        self.assertIsNone(get_class_weights(labels, "none"))

    def test_missing_class_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "all five classes"):
            get_class_weights([0, 1, 2, 3])

    def test_rare_example_has_larger_gradient(self):
        weights = get_class_weights([0] + [1] * 2 + [2] * 3 + [3] * 10 + [4] * 4)
        logits = torch.zeros(2, 5, requires_grad=True)
        loss, denominator = classification_loss_sum(logits, torch.tensor([0, 3]), weights)
        (loss / denominator).backward()
        self.assertAlmostEqual(float(logits.grad[0].norm() / logits.grad[1].norm()), 10.0, places=5)

    def test_accumulation_matches_full_batch_including_partial_window(self):
        torch.manual_seed(7)
        inputs = torch.randn(7, 3)
        labels = torch.tensor([3, 3, 0, 1, 4, 2, 3])
        balanced = get_class_weights(labels.numpy())
        for weights in (balanced, None):
            # Include a short last optimizer window, plus microbatch size 1 where
            # normalizing each weighted loss separately would cancel the weights.
            for window, sizes in ((slice(0, 5), [1, 3, 1]), (slice(5, 7), [1, 1])):
                with self.subTest(weighted=weights is not None, sizes=sizes):
                    full = torch.nn.Linear(3, 5)
                    accumulated = torch.nn.Linear(3, 5)
                    accumulated.load_state_dict(full.state_dict())
                    x, y = inputs[window], labels[window]
                    F.cross_entropy(full(x), y, weight=weights).backward()
                    offset, denominator = 0, 0
                    for size in sizes:
                        loss, weight_sum = classification_loss_sum(accumulated(x[offset:offset + size]), y[offset:offset + size], weights)
                        loss.backward()
                        denominator += weight_sum
                        offset += size
                    normalize_gradients(accumulated.parameters(), denominator)
                    for actual, expected in zip(accumulated.parameters(), full.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad)


if __name__ == "__main__":
    unittest.main()
