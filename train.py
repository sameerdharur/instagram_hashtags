import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import config
import data
import model
import nltk
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

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ') 
                 if ('<eos>' not in x and '<sos>' not in x and '<unk>' not in x and '<pad>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ') 
                  if ('<eos>' not in x and '<sos>' not in x and '<unk>' not in x and '<pad>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    if len(captions.shape) == 3:
        captions = captions.squeeze(dim = 2)
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t].item()]
            # print(word)
            if word != '<pad>':
                words.append(word)
            if word == '<eos>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


total_iterations = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(net, loader, optimizer, tracker, criterion, lr_scheduler, cap_vcb, hash_vcb, train=False, prefix='', epoch=0):
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
        blues = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    blue_tracker = tracker.track('{}_blue'.format(prefix), tracker_class(**tracker_params))

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
        _, predictions = out.topk(1,2,True,True)
        inv_hash_dict = {v: k for k, v in hash_vcb.items()}
        inv_cap_dict = {v: k for k, v in cap_vcb.items()}
        decoded_predictions = decode_captions(predictions, inv_hash_dict)
        decoded_hashtags = decode_captions(a, inv_hash_dict)
        bleu_arr = []
        for i in range(len(decoded_hashtags)):
            bleu_arr.append(BLEU_score(decoded_hashtags[i], decoded_predictions[i]))
        avg_blue = float(sum(bleu_arr))/len(bleu_arr)
        # print(a.shape)
        # acc = accuracy(out, a, 1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)

        if train:
            global total_iterations
            # update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            blues.append(avg_blue)
            # accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        blue_tracker.append(avg_blue)
        # acc_tracker.append(acc)
        # print(acc)
        # for a in acc:
        #     acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), blue = fmt(blue_tracker.mean.value))

    # if not train:
    #     answ = list(torch.cat(answ, dim=0))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    #     # accs = list(torch.cat(accs, dim=0))
    #     # blues = list(torch.cat(blues, dim=0))
        
    #     idxs = list(torch.cat(idxs, dim=0))
    #     # return answ, accs, idxs
    #     return answ, idxs,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              blues


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(config.train_path, train=True)
    val_loader = data.get_loader(config.val_path, val=True)

    # net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens[0],train_loader.dataset.num_tokens[1]).to(device))    
    net = model.Net(train_loader.dataset.num_tokens[0],train_loader.dataset.num_tokens[1]).to(device)  
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * len(train_loader), 0.0001)

    tracker = utils.Tracker()
    # for k,v in vars(config).items():
    #     print(k)
    # sdfsd
    #BRING THIS BACK
    # criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    criterion = nn.CrossEntropyLoss(ignore_index = train_loader.dataset.answer_to_index['<pad>'])
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not k.startswith('os') and not k.startswith('expanduser') and not k.startswith('platform')}

    cap_vcb = train_loader.dataset.token_to_index
    hash_vcb = train_loader.dataset.answer_to_index
    for i in range(config.epochs):
        # _ = run(net, train_loader, optimizer, tracker, criterion, lr_scheduler, cap_vcb, hash_vcb, train=True, prefix='train', epoch=i)
        # r = run(net, val_loader, optimizer, tracker, criterion, lr_scheduler, cap_vcb, hash_vcb, train=False, prefix='val', epoch=i)
        run(net, train_loader, optimizer, tracker, criterion, lr_scheduler, cap_vcb, hash_vcb, train=True, prefix='train', epoch=i)
        run(net, val_loader, optimizer, tracker, criterion, lr_scheduler, cap_vcb, hash_vcb, train=False, prefix='val', epoch=i)
        # print(train_loader.dataset.token_to_index)
        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            # 'eval': {
            #     'answers': r[0],
            #     # 'accuracies': r[1],
            #     'blues': r[2],
            #     'idx': r[1],
            # },
            'cap_vocab': train_loader.dataset.token_to_index,
            'hash_vocab': train_loader.dataset.answer_to_index
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
