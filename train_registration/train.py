import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from registration_segmentation.model import VxmDense
from train_registration.utils import TrainSetLoader
from sklearn.metrics import *
from visualization.view_2D import *
from tqdm import tqdm
from registration_2025.losses import *

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.9
                    , help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    cuda = opt.cuda
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = VxmDense(inshape=(512, 512))
    # model.load_state_dict(torch.load("/home/chuy/Projections/registration_2/checkpoint/Corona.pth"))
    model = model.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        data_set = TrainSetLoader(dataset_dir="/data/X-ray/rescaled_data/Corona/train")
        data_loader = DataLoader(dataset=data_set, num_workers=opt.threads,
                                 batch_size=opt.batchSize, shuffle=True)
        trainor(data_loader, optimizer, model, epoch)

        scheduler.step()


def trainor(data_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0

    L_NCC = NCC()
    # L_MSE = nn.MSELoss()
    L_regu = Grad(penalty='l2')

    for iteration, data in tqdm(enumerate(data_loader)):

        # print(data.shape)
        moving = data[:, 0].unsqueeze(1)
        fixed = data[:, 1].unsqueeze(1)
        registered_img, pos_flow = model(moving, fixed)

        # raw_cpu = moving.cpu().detach().numpy()[0, 0]
        # fixed_cpu = fixed.cpu().detach().numpy()[0, 0]
        # re_cpu = registered_img.cpu().detach().numpy()[0, 0]
        # flow_cpu = pos_flow.cpu().detach().numpy()[0]
        #
        # plot_parallel(
        #     a=raw_cpu,
        #     b=fixed_cpu,
        #     c=re_cpu,
        # )

        # plot_parallel(
        #     a=flow_cpu[0],
        #     b=flow_cpu[1],
        # )

        # loss_image = L_MSE(registered_img, fixed)
        loss_image = L_NCC(registered_img, fixed)
        loss_flow = L_regu(pos_flow)

        loss = loss_image * 10 + loss_flow

        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 100 == 0:
            print("===> Epoch[{}]: average_loss: {:.5f}".format
                  (epoch, loss_epoch / (iteration % 100 + 1)))
            loss_epoch = 0

            save_checkpoint(model, "./checkpoint")
            print("model has benn saved")


def save_checkpoint(model, path):
    model_out_path = os.path.join(path, f"Xray.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('forkserver', force=True)
    # torch.multiprocessing.set_start_method("spawn", force=True)
    from torch.multiprocessing import Pool, Process, set_start_method

    # torch.multiprocessing.set_start_method("spawn", force=True)

    train()
