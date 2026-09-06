"""Deep Ensemble implementation for aggregating M independent segmentation models."""

from typing import List, Dict, Union
import torch
import torch.nn as nn
from src.models.unet import PneumothoraxUNet


class DeepEnsemble(nn.Module):
    """Container for M independently trained neural networks for epistemic uncertainty quantification."""

    def __init__(self, models: List[PneumothoraxUNet]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    @classmethod
    def load_from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        device: Union[str, torch.device] = "cpu",
    ) -> "DeepEnsemble":
        """Instantiate an ensemble by loading weights from M checkpoint files.

        Args:
            checkpoint_paths: List of file paths to saved .pth checkpoints.
            device: Target device.

        Returns:
            DeepEnsemble instance.
        """
        models = []
        for path in checkpoint_paths:
            model = PneumothoraxUNet(dropout_rate=0.0, pretrained=False)
            state_dict = torch.load(path, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
        return cls(models)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference across all M ensemble members and decompose predictive uncertainty.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Dict containing:
                - 'mean': Ensemble mean probability map in [0, 1].
                - 'variance': Ensemble variance across members (epistemic).
                - 'total_entropy': Total predictive entropy H(mean).
                - 'expected_entropy': Expected entropy across members (aleatoric proxy).
                - 'mutual_information': Epistemic uncertainty = total_entropy - expected_entropy.
                - 'member_probs': Stacked tensor of individual predictions (M, B, 1, H, W).
        """
        member_probs = []
        member_entropies = []
        eps = 1e-7

        for model in self.models:
            model.eval()
            logits = model(x)
            prob = torch.sigmoid(logits)
            member_probs.append(prob)

            p_clamped = torch.clamp(prob, eps, 1.0 - eps)
            ent = -p_clamped * torch.log2(p_clamped) - (1.0 - p_clamped) * torch.log2(
                1.0 - p_clamped
            )
            member_entropies.append(ent)

        # Stack across ensemble dimension: shape (M, B, 1, H, W)
        stacked_probs = torch.stack(member_probs, dim=0)
        stacked_ents = torch.stack(member_entropies, dim=0)

        # 1. Ensemble Mean Prediction
        mean_prob = torch.mean(stacked_probs, dim=0)

        # 2. Ensemble Predictive Variance
        variance = torch.var(stacked_probs, dim=0, unbiased=True)

        # 3. Total Predictive Entropy: H(mean)
        mean_clamped = torch.clamp(mean_prob, eps, 1.0 - eps)
        total_entropy = -mean_clamped * torch.log2(mean_clamped) - (
            1.0 - mean_clamped
        ) * torch.log2(1.0 - mean_clamped)

        # 4. Expected Entropy (Aleatoric): E[H(p_m)]
        expected_entropy = torch.mean(stacked_ents, dim=0)

        # 5. Epistemic Uncertainty (Mutual Information): I(Y; theta | x) = H(mean) - E[H(p_m)]
        # Clamped to zero to prevent numerical precision underflow
        mutual_info = torch.clamp(total_entropy - expected_entropy, min=0.0)

        return {
            "mean": mean_prob,
            "variance": variance,
            "total_entropy": total_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "member_probs": stacked_probs,
        }
