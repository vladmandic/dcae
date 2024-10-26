import torch
from torchmetrics.image.fid import FrechetInceptionDistance

class FID():
    def __init__(self, feature: int = 2048):
        self.feature = feature
        self.fid = FrechetInceptionDistance(feature)

    def __call__(self, real, fake) -> torch.Tensor:
        if real.shape != fake.shape:
            raise ValueError('real and fake must have the same shape')
        real = torch.from_numpy(real).permute(2, 0, 1).unsqueeze(0)
        fake = torch.from_numpy(fake).permute(2, 0, 1).unsqueeze(0)
        real = torch.cat((real, real), 0)
        fake = torch.cat((fake, fake), 0)
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        score = self.fid.compute()
        return score.item()
