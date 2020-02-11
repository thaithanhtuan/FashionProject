class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self,x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) + torch.mul(y1 - y2, y1 - y2))
        return dt

class GicLoss(nn.Module):
    def __init__(self, batch_size, imageheight, imagewidth):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.dtx = torch.zeros([batch_size, imageheight, imagewidth])
        self.dty = torch.zeros([batch_size, imageheight, imagewidth])


    def buildLoss(self,  Gx, Gy):
        Lgic = 0
        for n in range(Gx.shape[0]):
            for y in range(0, Gx.shape[1] - 2):
                for x in range(0, Gx.shape[2] - 2):
                    self.dtx[n, y, x] = self.dT(Gx[n, y, x], Gy[n, y, x], Gx[n, y, x + 1], Gy[n, y, x + 1])
                    self.dty[n, y, x] = self.dT(Gx[n, y, x], Gy[n, y, x], Gx[n, y + 1, x], Gy[n, y + 1, x])

        for n in range(Gx.shape[0]):
            for y in range(1, Gx.shape[1] - 2):
                for x in range(1, Gx.shape[2] - 2):
                    Lgic = Lgic + torch.abs(self.dtx[n, y - 1, x - 1] - self.dtx[n, y - 1, x]) + torch.abs(
                        self.dty[n, y - 1, x - 1] - self.dty[n, y, x - 1])
        return Lgic/(Gx.shape[0]*Gx.shape[1]*Gx.shape[2])

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]

        return self.buildLoss(Gx=Gx, Gy=Gy)
