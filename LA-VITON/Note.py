 starttime = time.time()
        Gx = grid[:, :, :, 0]

        Gy = grid[:, :, :, 1]

        dtx = np.zeros([Gx.shape[0], Gx.shape[1], Gx.shape[2]])
        dty = np.zeros([Gx.shape[0], Gx.shape[1], Gx.shape[2]])
        Gx.cpu()
        Gy.cpu()
        Lgic = 0
        print("start")
        for n in range(Gx.shape[0]):
            for y in range(0, Gx.shape[1] - 2):
                for x in range(0, Gx.shape[2] - 2):
                    dtx[n, y, x] = DT(Gx[n, y, x].item(), Gy[n, y, x].item(), Gx[n, y, x + 1].item(), Gy[n, y, x + 1].item())
                    dty[n, y, x] = DT(Gx[n, y, x].item(), Gy[n, y, x].item(), Gx[n, y + 1, x].item(), Gy[n, y + 1, x].item())

        for n in range(Gx.shape[0]):
            for y in range(1, Gx.shape[1] - 2):
                for x in range(1, Gx.shape[2] - 2):
                    Lgic = Lgic + np.abs(dtx[n, y - 1, x - 1] - dtx[n, y - 1, x]) + np.abs(
                        dty[n, y - 1, x - 1] - dty[n, y, x - 1])

        #endtime = time.time()
        #print("time:", endtime - starttime)
        Lgic = Lgic / (Gx.shape[0] * Gx.shape[1] * Gx.shape[2])
        Lwarp = criterionL1(warped_cloth, im_c)
        loss = Lwarp + Lgic
        #Lgic = gicloss(grid)
        endtime = time.time()
        print("CalLgic:", endtime - starttime)

        #Lwarp = criterionL1(warped_cloth, im_c)
        #loss = Lwarp + Lgic


        #loss = Lwarp + Lgic
        #Lwarp.cuda()
        #loss.cuda()
        optimizer.zero_grad()
        print("begin backward: ", time.time() - endtime)
        loss.backward()
        print("finish backward: ", time.time() - endtime)
        optimizer.step()
