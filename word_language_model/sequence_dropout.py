import torch.nn


class SequenceDropout(torch.nn.Dropout):
    """[summary]
    
    Args:
        torch ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    def __init__(self, *args, **kwargs):
        super(SequenceDropout, self).__init__(*args, **kwargs)
        self.mask: torch.Tensor = torch.empty(1)
        self.new_mask = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        if not self.training:
            return x
        if self.new_mask:
            self.mask = torch.nn.functional.dropout(torch.ones_like(x), p=self.p)
            self.new_mask = False

        return x * self.mask

    def generate_new_mask(self):
        self.new_mask = True
