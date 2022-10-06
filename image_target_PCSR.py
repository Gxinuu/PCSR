import argparse
import os, sys
import os.path as osp

import faiss
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import KnowledgeDistillationLoss
from numpy import linalg as LA
from torch.nn import functional as F


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):

    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):

    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load_original(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_class[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_class[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["source_tr"] = ImageList(txt_src, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(txt_src, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs * 3, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output_t = outputs.float().cpu()
                all_label_t = labels.float()
                start_test = False
            else:
                all_output_t = torch.cat((all_output_t, outputs.float().cpu()), 0)
                all_label_t = torch.cat((all_label_t, labels.float()), 0)

    # all_output_t = nn.Softmax(dim=1)(all_output_t)
    _, predict_t = torch.max(all_output_t, 1)
    accuracy_t = torch.sum(torch.squeeze(predict_t).float() == all_label_t).item() / float(all_label_t.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output_t)).cpu().data.item()
    if flag:
        matrix = confusion_matrix(all_label_t, predict_t)
        matrix = matrix[np.unique(all_label_t).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy_t * 100

def train_target(args):
    dset_loaders = data_load(args)

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()


    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]  # lr_decay1 = 0.1
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]  # lr_decay1 = 1.0
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])

    for epoch_idx in range(args.max_epoch):
        iter_idx = epoch_idx * len(dset_loaders["target"])
        with torch.no_grad():
            netF.eval()
            netB.eval()

            glob_multi_feat_cent, all_psd_label, dd = init_multi_cent_psd_label(dset_loaders['test'], netF,
                                                                                 netB, netC, args)
            netF.train()
            netB.train()

        iter_num = 0
        while iter_num < len(dset_loaders["target"]):
            try:
                inputs_target, _, tar_idx = iter_target.next()
            except:
                iter_target = iter(dset_loaders["target"])
                inputs_target, _, tar_idx = iter_target.next()

            if inputs_target.size(0) == 1:
                continue

            iter_idx += 1
            iter_num += 1

            inputs_target = inputs_target.cuda()
            tar_idx = tar_idx.cuda()

            features_test = netB(netF(inputs_target))
            outputs_test = netC(features_test)
            pred = all_psd_label[tar_idx]

            if args.cls_par > 0:

                pred_soft = dd[tar_idx]
                classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
                classifier_loss *= args.cls_par

                if args.kd:
                    kd_loss = KnowledgeDistillationLoss()(outputs_test, pred_soft)
                    classifier_loss += kd_loss

                if epoch_idx < 1 and args.dset == "VISDA-C":
                    classifier_loss *= 0
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
                classifier_loss += im_loss

            lr_scheduler(optimizer, iter_num=iter_idx, max_iter=max_iter, power=1.5)
            optimizer.zero_grad()
            classifier_loss.backward()

            if args.mix > 0:
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs_target.size()[0]).cuda()
                mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
                mixed_output = (lam * softmax_out + (1 - lam) * softmax_out[index, :]).detach()

                model = nn.Sequential(netF, netB, netC).cuda()

                update_batch_stats(model, False)
                outputs_target_m = model(mixed_input)
                update_batch_stats(model, True)
                outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
                classifier_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
                classifier_loss.backward()

            optimizer.step()

        with torch.no_grad():
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list= cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_idx, max_iter,
                                                                        acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                print(acc_s_te)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_idx, max_iter,
                                                                       acc_s_te)
            netF.train()
            netB.train()

        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def init_multi_cent_psd_label(loader, netF, netB, netC, args):
    emd_feat_stack = []
    cls_out_stack = []
    gt_label_stack = []

    idx = 0
    while idx < len(loader):
        try:
            data_test, data_label, data_idx = iter_target.next()
        except:
            iter_target = iter(loader)
            data_test, data_label, data_idx = iter_target.next()

        if data_test.size(0) == 1:
            continue

        data_test = data_test.cuda()
        data_label = data_label.cuda()
        embed_feat = netB(netF(data_test))
        cls_out = netC(embed_feat)

        emd_feat_stack.append(embed_feat)
        cls_out_stack.append(cls_out)
        gt_label_stack.append(data_label)
        idx += 1

    all_gt_label = torch.cat(gt_label_stack, dim=0)
    all_emd_feat = torch.cat(emd_feat_stack, dim=0)
    all_emd_feat = all_emd_feat / torch.norm(all_emd_feat, p=2, dim=1, keepdim=True)
    topk_num = max(all_emd_feat.shape[0] // (args.class_num * args.topk_seg), 1)  # Mk topk_seg为r 3

    all_cls_out = torch.cat(cls_out_stack, dim=0)
    all_output = nn.Softmax(dim=1)(all_cls_out)
    _, all_psd_label = torch.max(all_output, dim=1)  # all_psd_label
    #_, all_psd_label = torch.max(all_cls_out, dim=1)  # all_psd_label
    acc = torch.sum(all_gt_label == all_psd_label).true_divide(len(all_gt_label))
    acc_list = [acc]

    multi_cent_num = args.multi_cent_num
    feat_multi_cent = torch.zeros((args.class_num, multi_cent_num, args.embed_feat_dim)).cuda()
    faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, multi_cent_num, niter=100, verbose=False,
                                min_points_per_centroid=1)

    iter_nums = 2
    for iter_idx in range(iter_nums):
        for cls_idx in range(args.class_num):
            if iter_idx == 0:
                feat_sample_idx = torch.topk(all_output[:, cls_idx], topk_num)[1]
            else:
                feat_sample_idx = torch.topk(feat_dist[:, cls_idx], topk_num)[1]

            feat_cls_sample = all_emd_feat[feat_sample_idx, :].cpu().numpy()
            # print(feat_cls_sample.shape)
            faiss_kmeans.train(feat_cls_sample)
            feat_multi_cent[cls_idx, :] = torch.from_numpy(faiss_kmeans.centroids).cuda()

        feat_dist = torch.einsum("cmk, nk -> ncm", feat_multi_cent, all_emd_feat)
        feat_dist, _ = torch.max(feat_dist, dim=2)
        feat_dist = torch.softmax(feat_dist, dim=1)

        _, all_psd_label = torch.max(feat_dist, dim=1)

        acc = torch.sum(all_psd_label == all_gt_label).true_divide(len(all_gt_label))
        acc_list.append(acc)

    log = "acc:" + " --> ".join("{:.4f}".format(acc) for acc in acc_list)
    args.out_file.write(log + "\n")
    args.out_file.flush()
    print(log)

    if args.dset == 'VISDA-C':
        psd_confu_mat = confusion_matrix(all_gt_label.cpu(), all_psd_label.cpu())
        psd_acc_list = psd_confu_mat.diagonal() / psd_confu_mat.sum(axis=1) * 100
        psd_acc = psd_acc_list.mean()
        psd_acc_str = "{:.2f}       ".format(psd_acc) + " ".join(["{:.2f}".format(i) for i in psd_acc_list])
        print(psd_acc_str)

    return feat_multi_cent, all_psd_label, feat_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    # 新增TransDA
    parser.add_argument('--kd', type=bool, default=True)
    parser.add_argument('--se', type=bool, default=True)
    parser.add_argument('--nl', type=bool, default=True)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--mix', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument("--multi_cent_num", default=3, type=int)
    parser.add_argument("--topk_seg", default=3, type=int)
    parser.add_argument("--embed_feat_dim", type=int, default=256)

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = './data/'
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)

        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        train_target(args)
        args.out_file.close()
