"""ResNet34 U-Net architecture supporting deterministic inference and MC Dropout."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Standard U-Net decoder block with optional Spatial Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_dropout = dropout_rate > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # Handle slight padding discrepancy
            diff_y = skip.size()[2] - x.size()[2]
            diff_x = skip.size()[3] - x.size()[3]
            if diff_y != 0 or diff_x != 0:
                x = F.pad(
                    x,
                    [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_dropout:
            x = self.dropout(x)
        return x


class PneumothoraxUNet(nn.Module):
    """ResNet34 U-Net with configurable Spatial Dropout for MC Dropout and Deep Ensembles."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Try loading smp or fallback to torchvision
        try:
            import segmentation_models_pytorch as smp

            weights = "imagenet" if pretrained else None
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes,
                decoder_channels=(256, 128, 64, 32, 16),
            )
            self.is_smp = True
            if dropout_rate > 0.0:
                # Add SpatialDropout2d to decoder blocks
                for idx in range(len(self.model.decoder.blocks)):
                    self.model.decoder.blocks[idx].add_module(
                        "spatial_dropout", nn.Dropout2d(p=dropout_rate)
                    )
        except ImportError:
            try:
                import torchvision.models as models

                self.is_smp = False
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                resnet = models.resnet34(weights=weights)

                self.init_conv = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                )
                self.maxpool = resnet.maxpool
                self.layer1 = resnet.layer1  # 64 channels
                self.layer2 = resnet.layer2  # 128 channels
                self.layer3 = resnet.layer3  # 256 channels
                self.layer4 = resnet.layer4  # 512 channels

                self.bottleneck_dropout = (
                    nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
                )

                # Decoder blocks
                self.dec4 = DecoderBlock(512, 256, 256, dropout_rate=dropout_rate)
                self.dec3 = DecoderBlock(256, 128, 128, dropout_rate=dropout_rate)
                self.dec2 = DecoderBlock(128, 64, 64, dropout_rate=dropout_rate)
                self.dec1 = DecoderBlock(64, 64, 32, dropout_rate=dropout_rate)
                self.dec0 = DecoderBlock(32, 0, 16, dropout_rate=0.0)

                self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
            except ImportError:
                # Standalone pure PyTorch fallback for CPU testing without torchvision
                self.is_smp = False
                self.init_conv = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
                self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
                self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())

                self.bottleneck_dropout = (
                    nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
                )
                self.dec4 = DecoderBlock(512, 256, 256, dropout_rate=dropout_rate)
                self.dec3 = DecoderBlock(256, 128, 128, dropout_rate=dropout_rate)
                self.dec2 = DecoderBlock(128, 64, 64, dropout_rate=dropout_rate)
                self.dec1 = DecoderBlock(64, 64, 32, dropout_rate=dropout_rate)
                self.dec0 = DecoderBlock(32, 0, 16, dropout_rate=0.0)
                self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning pre-sigmoid logits."""
        if self.is_smp:
            return self.model(x)

        x0 = self.init_conv(x)  # shape: (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)  # shape: (B, 64, H/4, W/4)

        x1 = self.layer1(x_pool)  # shape: (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # shape: (B, 128, H/8, W/8)
        x3 = self.layer3(x2)  # shape: (B, 256, H/16, W/16)
        x4 = self.layer4(x3)  # shape: (B, 512, H/32, W/32)

        x4 = self.bottleneck_dropout(x4)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1, None)

        return self.final_conv(d0)

    @torch.no_grad()
    def predict_deterministic(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a single deterministic evaluation pass.

        Returns:
            Dict containing:
                - 'prob': Sigmoid probability map in [0, 1].
                - 'entropy': Shannon predictive entropy map in [0, 1].
        """
        self.eval()
        logits = self.forward(x)
        prob = torch.sigmoid(logits)

        # Shannon entropy: -p log2(p) - (1-p) log2(1-p)
        eps = 1e-7
        p_clamped = torch.clamp(prob, eps, 1.0 - eps)
        entropy = -p_clamped * torch.log2(p_clamped) - (1.0 - p_clamped) * torch.log2(
            1.0 - p_clamped
        )

        return {"prob": prob, "entropy": entropy}

    @torch.no_grad()
    def predict_mc_dropout(
        self,
        x: torch.Tensor,
        num_samples: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """Perform Monte Carlo Dropout inference by sampling T stochastic passes.

        Keeps dropout active during evaluation.

        Args:
            x: Input tensor of shape (B, C, H, W).
            num_samples: Number of stochastic passes (default: 20).

        Returns:
            Dict containing:
                - 'mean': Mean predictive probability map across T passes.
                - 'variance': Predictive variance map across T passes (epistemic).
                - 'entropy': Total predictive entropy of the mean probability.
                - 'samples': Full tensor of sampled probabilities (T, B, 1, H, W).
        """
        self.train()  # Activates dropout layers

        samples = []
        for _ in range(num_samples):
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            samples.append(prob)

        # Stack samples: shape (T, B, 1, H, W)
        stacked = torch.stack(samples, dim=0)

        # Predictive mean and variance
        mean_prob = torch.mean(stacked, dim=0)
        variance = torch.var(stacked, dim=0, unbiased=True)

        # Total predictive entropy of the mean probability
        eps = 1e-7
        p_clamped = torch.clamp(mean_prob, eps, 1.0 - eps)
        entropy = -p_clamped * torch.log2(p_clamped) - (1.0 - p_clamped) * torch.log2(
            1.0 - p_clamped
        )

        return {
            "mean": mean_prob,
            "variance": variance,
            "entropy": entropy,
            "samples": stacked,
        }
