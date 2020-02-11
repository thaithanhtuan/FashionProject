class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self,x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) + torch.mul(y1 - y2, y1 - y2))
        return dt

class GicLoss(nn.Module):
    def __init__(self,opt):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.opt = opt



    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gxup = Gx[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gxdown = Gx[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gxleft = Gx[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gxright = Gx[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        Gycenter = Gy[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gyup = Gy[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gydown = Gy[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gyleft = Gy[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gyright = Gy[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        dtleft = self.dT(Gxleft, Gyleft, Gxcenter, Gycenter)
        dtright = self.dT(Gxright, Gyright, Gxcenter, Gycenter)
        dtup = self.dT(Gxup, Gyup, Gxcenter, Gycenter)
        dtdown = self.dT(Gxdown, Gydown, Gxcenter, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))

