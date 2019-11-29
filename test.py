import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(preds, targets, k):
    print(targets)
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    print(pred[0][0].size())
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


total_iterations = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(net, loader, optimizer, tracker, criterion, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    # log_softmax = nn.LogSoftmax(dim = 1).cuda()
    for v, q, a, idx, q_len, a_len in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        # v = Variable(v.cuda(async=True), **var_params)
        v = v.to(device)
        # q = Variable(q.cuda(async=True), **var_params)
        q = q.to(device)
        # a = Variable(a.cuda(async=True), **var_params)
        a = a.to(device)
        # q_len = Variable(q_len.cuda(async=True), **var_params)
        q_len = q_len.to(device)

        out = net(v, q, q_len, a, a_len)
        out = out.to(device)
        # print(out.shape)
        # nll = -log_softmax(out)
        # loss = (nll * a / 10).sum(dim=1).mean()
        # acc = utils.batch_accuracy(out.data, a.data).cpu()


        output_dim = out.shape[-1]

        output = out[1:].view(-1, output_dim)
        trg = a[1:].view(-1)
        # print(a.shape)
        acc = accuracy(out, a, 1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        # acc_tracker.append(acc)
        # print(acc)
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs
        # return answ, idxs


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    # train_loader = data.get_loader(config.train_path, train=True)
    test_loader = data.get_loader(config.test_path, test=True)

    net = nn.DataParallel(model.Net(test_loader.dataset.num_tokens[0],test_loader.dataset.num_tokens[1]).to(device))    
    # net = model.Net(train_loader.dataset.num_tokens[0],train_loader.dataset.num_tokens[1]).to(device)  
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    # for k,v in vars(config).items():
    #     print(k)
    # sdfsd
    #BRING THIS BACK
    # criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    criterion = nn.CrossEntropyLoss(ignore_index = test_loader.dataset.answer_to_index['<pad>'])
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not k.startswith('os') and not k.startswith('expanduser') and not k.startswith('platform')}


    for i in range(config.epochs):
        _ = run(net, test_loader, optimizer, tracker, criterion, train=False, prefix='train', epoch=i)
        # r = run(net, val_loader, optimizer, tracker, criterion, train=False, prefix='val', epoch=i)
        # print(train_loader.dataset.token_to_index)
        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.module.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'cap_vocab': test_loader.dataset.token_to_index,
            'hash_vocab': test_loader.dataset.answer_to_index
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
