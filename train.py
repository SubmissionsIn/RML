import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
import torch
from network import Network
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import argparse
import random
from loss import Loss
from dataloader import load_data
from torch.utils.data import DataLoader
from metric import valid
import time
from TSNE import TSNE_PLOT as ttsne

# RML
# RML_LCE

Datasets = [
    'DHA',
    'BDGP',
    'Prokaryotic',
    'Cora',
    'YoutubeVideo',
    'WebKB',
    'VOC',
    'NGs',
    'Cifar100',
]

Dataname = 'DHA'
MODE = 'RML'
tsne = False  # True / False
T = 5

if MODE == 'RML':
    train_rate = 1.0
    noise_label_rate = 0
if MODE == 'RML_LCE':
    train_rate = 0.7
    noise_label_rate = 0.5

multi_blocks = 1
multi_heads = 1

if noise_label_rate < 0.3:
    Lambda = 1
else:
    Lambda = 1000

# print(Lambda)

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)   # 256
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)  # 0.0003
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--con_iterations", default=50)
parser.add_argument("--tune_iterations", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--contrastive_feature_dim", default=256)
parser.add_argument('--mode', type=str, default=MODE)
parser.add_argument('--miss_rate', type=str, default=0.25)
parser.add_argument('--noise_rate', type=str, default=0.25)
parser.add_argument('--Gaussian_noise', type=str, default=0.4)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "MNIST-USPS":
    args.miss_rate = 0.75        # 0.75
    args.noise_rate = 0.75       # 0.75
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 5000
    seed = 1
if args.dataset == "BDGP":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 1000   # 1000
    seed = 0
if args.dataset == "LandUse_21":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 1000
    seed = 3
if args.dataset == "Scene_15":
    args.miss_rate = 0.75        # 0.25
    args.noise_rate = 0.75       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 20000
    seed = 3
if args.dataset == "NUS_WIDE":
    args.miss_rate = 0.75        # 0.25
    args.noise_rate = 0.75       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 100000
    seed = 3
if args.dataset == "MSCOCO":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 5000
    seed = 4
if args.dataset == "MIRFLICKR25K":
    args.miss_rate = 0.75        # 0.25
    args.noise_rate = 0.75       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 5000
    seed = 4
if args.dataset == "IAPR":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 50000
    seed = 4
if args.dataset == "Cora":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500    # 500
    seed = 9  # 9
if args.dataset == "Caltech101_20":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500
    seed = 3
if args.dataset == "Caltech":
    args.miss_rate = 0.75        # 0.25
    args.noise_rate = 0.75       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 1000   # 1000
    seed = 3
if args.dataset == "CCV":
    args.miss_rate = 0.75         # 0.75
    args.noise_rate = 0.75        # 0.75
    args.Gaussian_noise = 0.4     # 0.4
    args.con_iterations = 20000  # 10000
    seed = 1
if args.dataset == "Fashion":
    args.miss_rate = 0.75        # 0.25
    args.noise_rate = 0.75       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 10000         # 20000
    seed = 1
if args.dataset == "DHA":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 800   # 1000
    seed = 4  # 4
if args.dataset == "WebKB":
    args.miss_rate = 0.75        # 0.75
    args.noise_rate = 0.75       # 0.75
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 400    # 400
    seed = 1  # 1
if args.dataset == "NGs":
    args.miss_rate = 0.50        # 0.25
    args.noise_rate = 0.50       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 400    # 400
    seed = 1                     # 1
if args.dataset == "VOC":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500   # 500
    seed = 1                     # 1
if args.dataset == "Fc_COIL_20":
    args.miss_rate = 0.25        # 0.25 0.75
    args.noise_rate = 0.25       # 0.25 0.75
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 2000
    seed = 1
if args.dataset == "RGB-D":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500
    seed = 4
if args.dataset == "YoutubeVideo":
    args.miss_rate = 0.25         # 0.25
    args.noise_rate = 0.25        # 0.25
    args.Gaussian_noise = 0.4     # 0.4
    args.con_iterations = 200000  # 200000
    seed = 0
if args.dataset == "Prokaryotic":
    args.miss_rate = 0.50        # 0.50
    args.noise_rate = 0.50       # 0.50
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 1000    # 500
    seed = 9  # 9
if args.dataset == "Synthetic3d":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500
    seed = 4
if args.dataset == "Cifar100":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 10000
    seed = 4
if args.dataset == "pascal07_six_view":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500
    seed = 4
if args.dataset == "mirflickr_six_view":
    args.miss_rate = 0.25        # 0.25
    args.noise_rate = 0.25       # 0.25
    args.Gaussian_noise = 0.4    # 0.4
    args.con_iterations = 500
    seed = 4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mask(rows, cols, p):
    tensor = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        if i < int(rows * p):
            while True:
                row = np.random.randint(0, 2, size=cols)
                if np.count_nonzero(row) < cols and np.count_nonzero(row) > 0:
                    tensor[i, :] = row
                    break
        else:
            tensor[i, :] = 1
    np.random.shuffle(tensor)
    tensor = torch.tensor(tensor)
    return tensor


def add_noise(matrix, std, p):
    rows, cols = matrix.shape
    noisy_matrix = matrix.clone()
    for i in range(rows):
        if random.random() < p:
            noise = torch.randn(cols, device=device) * std
            noisy_matrix[i] += noise
    return noisy_matrix


def RML(iteration, model, mode, miss_rate, noise_rate, Gaussian_noise, data_loader):
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
            y = y.to(device)
        break
    masked_xs = []
    noised_xs = []
    num_rows = xs[0].shape[0]

    mask_tensor = mask(num_rows, view, miss_rate).to(device)
    for v in range(view):
        masked_x = mask_tensor[:, v].unsqueeze(1) * xs[v]
        masked_xs.append(masked_x)

    for v in range(view):
        noised_x = add_noise(xs[v], Gaussian_noise, noise_rate)
        noised_xs.append(noised_x)

    xs_all = torch.cat(xs, dim=1)
    # mask_all = torch.cat(masked_xs, dim=1)
    # noise_all = torch.cat(noised_xs, dim=1)
    optimizer.zero_grad()

    _, xs_z, q, scores, hs = model(xs)
    _, mask_z, mask_q, _, _ = model(masked_xs)
    _, noise_z, noise_q, _, _ = model(noised_xs)

    if mode == 'RML' or mode == 'RML_LCE':
        loss_con_1 = criterion.forward_feature(noise_z, mask_z)
        loss_con_2 = criterion.forward_feature(mask_z, noise_z)
        loss_con = loss_con_1 + loss_con_2
        loss = loss_con
    if mode == 'RML_LCE':
        crossentropyloss = nn.CrossEntropyLoss()
        loss_ce_x = crossentropyloss(q, y.long())
        loss_ce_mask = crossentropyloss(mask_q, y.long())
        loss_ce_noise = crossentropyloss(noise_q, y.long())
        loss_y = loss_ce_x + loss_ce_mask + loss_ce_noise

        # loss = loss_ce_x
        # loss = loss_y
        # loss = loss_ce_x + Lambda * loss_con
        loss = loss_y + Lambda * loss_con

    loss.backward()
    optimizer.step()
    print('\r', 'Iteration {}'.format(iteration), 'Loss:{:.6f}'.format(loss), end='')


metric1 = []
metric2 = []
metric3 = []
metric4 = []

for i in range(T):
    print("\n")
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    # setup_seed(seed+i)
    dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='train', seed=i, mode=MODE, noise_rate=noise_label_rate)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    epochs = int(args.con_iterations / (data_size / args.batch_size))
    print("Totally needed epochs: " + str(epochs))
    if epochs < 5:
        print("Training epochs are too less, and more training iterations are needed.")
        exit(0)
    model = Network(class_num, args.feature_dim, args.contrastive_feature_dim, device, dims, view, multi_blocks=multi_blocks, multi_heads=multi_heads)
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
    mode = args.mode
    miss_rate = args.miss_rate
    noise_rate = args.noise_rate
    Gaussian_noise = args.Gaussian_noise

    time0 = time.time()
    iteration = 1
    while iteration <= args.con_iterations:
        RML(iteration, model, mode, miss_rate, noise_rate, Gaussian_noise, data_loader)
        iteration += 1

    if mode == 'RML':
        dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='train', seed=i, noise_rate=0)
        m1, m2, m3, m4 = valid(model, device, dataset, view, data_size, class_num, eval_z=True)
        # acc, nmi, ari, pur
    if mode == 'RML_LCE':
        dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='test', seed=i, noise_rate=noise_label_rate)
        m1, m2, m3, m4 = valid(model, device, dataset, view, data_size, class_num, eval_q=True)
        # accuracy, precision, f1_score, recall
    metric1.append(m1)
    metric2.append(m2)
    metric3.append(m3)
    metric4.append(m4)

print('%.3f'% np.mean(metric1), '± %.3f'% np.std(metric1), metric1)
print('%.3f'% np.mean(metric2), '± %.3f'% np.std(metric2), metric2)
print('%.3f'% np.mean(metric3), '± %.3f'% np.std(metric3), metric3)
print('%.3f'% np.mean(metric4), '± %.3f'% np.std(metric4), metric4)

if tsne == True:
    mask_x = []
    noise_x = []
    model.eval()
    ALL_loader = DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    for step, (xs, ys, _) in enumerate(ALL_loader):
        ys = ys.numpy()
        for v in range(view):
            xs[v] = xs[v].to(device)
        num_rows = xs[0].shape[0]
        miss = mask(num_rows, view, miss_rate).to(device)
        for v in range(view):
            miss = miss[:, v].unsqueeze(1)*xs[v]
            mask_x.append(miss)
        for v in range(view):
            noisedx = add_noise(xs[v], Gaussian_noise, noise_rate)
            noise_x.append(noisedx)

        with torch.no_grad():
            xr, _, z, _, _, _ = model.forward(xs)
            xmr, _, zm, _, _, _ = model.forward(mask_x)
            xnr, _, zn, _, _, _ = model.forward(noise_x)
            z = z.cpu().detach().numpy()
            zm = zm.cpu().detach().numpy()
            zn = zn.cpu().detach().numpy()
        ttsne(z, ys, "z")
        # ttsne(zm, ys, "zm")
        # ttsne(zn, ys, "zn")
