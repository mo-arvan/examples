import torch.nn


class GaussianNoise(torch.nn.Module):
    """
    Multiplicative Gaussian Noise applied to the input
    """

    def __init__(self, std, *args, **kwargs):
        super(GaussianNoise, self).__init__(*args, **kwargs)
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        if not self.training:
            return x
        noise = torch.normal(mean=1., std=self.std, size=x.size(), device=x.device)

        return x * noise
