import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def DT(x1,y1,x2,y2):
    dt = torch.sqrt(torch.mul(x1- x2, x1- x2) + torch.mul(y1-y2,y1-y2))
    return dt

def main():
    fine_width = 4
    fine_height = 5
    grid = [[[[1,2],[3,4],[5,6],[7,8]],[[9,10],[11,12],[13,14],[15,16]],[[17,18],[19,20],[21,22],[23,24]],[[25,26],[27,28],[29,30],[31,32]],[[33,34],[35,36],[37,38],[39,40]]]]
    grid = np.asarray(grid)
    gridtensor = torch.FloatTensor(grid)
    Gx = gridtensor[:, :, :, 0]
    Gy = gridtensor[:, :, :, 1]

    Gxcenter = Gx[:, 1:fine_height-1, 1:fine_width-1 ]
    Gxup = Gx[:, 0:fine_height - 2, 1:fine_width - 1]
    Gxdown = Gx[:, 2:fine_height , 1:fine_width -1]
    Gxleft = Gx[:, 1:fine_height -1 , 0:fine_width - 2]
    Gxright = Gx[:, 1:fine_height -1, 2:fine_width ]
    print(Gxcenter.detach().numpy())
    """
    [[[11. 13.]
    [19. 21.]
    [27. 29.]]]
    """
    print(Gxup.detach().numpy())
    """
    [[[ 3.  5.]
    [11. 13.]
    [19. 21.]]]
    """
    print(Gxdown.detach().numpy())
    """
    [[[19. 21.]
    [27. 29.]
    [35. 37.]]]
    """
    print(Gxleft.detach().numpy())
    """
    [[[ 9. 11.]
    [17. 19.]
    [25. 27.]]]
    """
    print(Gxright.detach().numpy())
    """
    [[[13. 15.]
    [21. 23.]
    [29. 31.]]]
    """
    print(Gx.detach().numpy())
    print(Gy.detach().numpy())
    """
    [[[ 1.  3.  5.  7.]
    [ 9. 11. 13. 15.]
    [17. 19. 21. 23.]
    [25. 27. 29. 31.]
    [33. 35. 37. 39.]]]
    
    [[[ 2.  4.  6.  8.]
    [10. 12. 14. 16.]
    [18. 20. 22. 24.]
    [26. 28. 30. 32.]
    [34. 36. 38. 40.]]]
    """
    Gycenter = Gy[:, 1:fine_height - 1, 1:fine_width - 1]
    Gyup = Gy[:, 0:fine_height - 2, 1:fine_width - 1]
    Gydown = Gy[:, 2:fine_height, 1:fine_width - 1]
    Gyleft = Gy[:, 1:fine_height - 1, 0:fine_width - 2]
    Gyright = Gy[:, 1:fine_height - 1, 2:fine_width]


    dtleft = DT(Gxleft,Gyleft,Gxcenter,Gycenter)
    dtright = DT(Gxright,Gyright,Gxcenter,Gycenter)
    dtup = DT(Gxup,Gyup,Gxcenter,Gycenter)
    dtdown = DT(Gxdown,Gydown,Gxcenter,Gycenter)
    print("distance:")
    print(dtleft.detach().numpy())
    """
    [[[2.828427 2.828427]
    [2.828427 2.828427]
    [2.828427 2.828427]]]
    """
    print(dtright.detach().numpy())
    """
    [[[2.828427 2.828427]
    [2.828427 2.828427]
    [2.828427 2.828427]]]
    """
    print(dtup.detach().numpy())
    """
    [[[11.313708 11.313708]
    [11.313708 11.313708]
    [11.313708 11.313708]]]
    """
    print(dtdown.detach().numpy())
    """
    [[[11.313708 11.313708]
    [11.313708 11.313708]
    [11.313708 11.313708]]]
    """
    loss = torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))

    print("total loss:", loss.item())


if __name__ == "__main__":
    main()
