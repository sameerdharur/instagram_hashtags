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
import argparse
import utils
from torchvision.utils import save_image
from collections import defaultdict
import nltk


ap = argparse.ArgumentParser()
ap.add_argument("--model_path" , required= True, type = str)
args = ap.parse_args()

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 2, True, True)
    # print(pred[0][0].size())
    # targets = targets.permute(1,0)
    pred = pred.squeeze()
    pred = pred.permute(1,0)
    # print(pred.shape)
    # print(targets.shape)
    correct = 0
    total = 0
    for i in range(targets.shape[0]):
        for j in range(1,targets.shape[1]):
            if pred[i,j] == targets[i,j]:
                correct += 1
            total += 1
            if targets[i,j] == 2:
                break
    # correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    # correct_total = correct.view(-1).float().sum()
    # return correct_total.item() * (100.0 / batch_size)
    return float(correct) / total

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
    # print(captions)
    # print(captions.shape)
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t].item()]
            # print(word)
            if word != '<pad>' and word!= '<sos>':
                words.append(word)
            if word == '<eos>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


total_iterations = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(net, loader, tracker, criterion, cap_vcb, hash_vcb, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {}
        answ = []
        idxs = []
        accs = []
        blues = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    blue_tracker = tracker.track('{}_blue'.format(prefix), tracker_class(**tracker_params))

    # log_softmax = nn.LogSoftmax(dim = 1).cuda()
    ans_dict = defaultdict(str)
    cap_dict = defaultdict(str)
    pred_dict = defaultdict(str)
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
        out = net(v, q, q_len, a, a_len, teacher_forcing_ratio = 0)
        out = out.to(device)
        # a = a.permute(1,0)
        # print(out.shape)
        # nll = -log_softmax(out)
        # loss = (nll * a / 10).sum(dim=1).mean()
        # acc = utils.batch_accuracy(out.data, a.data).cpu()


        output_dim = out.shape[-1]
        # print(a[1:])
        output = out[1:].view(-1, output_dim)
        # trg = a[1:].view(-1)
        # print(a[:,1:])
        trg = a[:,1:].reshape(-1)
        # print(a[:,1:].reshape(-1))
        # kjsajkhsdf   
        # print(a.shape)
        # print(out.shape)
        # print(a.shape)
        acc = accuracy(out, a, 1)
        _, predictions = out.topk(1,2,True,True)
        predictions = predictions.squeeze()
        predictions = predictions.permute(1,0)
        # print(predictions.shape)
        # print(a.shape)
        # ksljdf
        inv_hash_dict = {v: k for k, v in hash_vcb.items()}
        inv_cap_dict = {v: k for k, v in cap_vcb.items()}
        decoded_predictions = decode_captions(predictions, inv_hash_dict)
        decoded_hashtags = decode_captions(a, inv_hash_dict)
        decoded_captions = decode_captions(q, inv_cap_dict)
        # print(decoded_captions)
        # ksldhf
        # print(idx)
        # print(decoded_hashtags[0])
        for id in idx: 
            ans_dict[id] = decoded_hashtags[id%len(decoded_hashtags)]
        for id in idx: 
            pred_dict[id] = decoded_predictions[id%len(decoded_predictions)]
        for id in idx: 
            cap_dict[id] = decoded_captions[id%len(decoded_captions)]
        
        for i,id in enumerate(idx):
            # print(v[0].shape)
            save_image(v[i], 'output/images/img_' + str(id.item()) + '.png')
        # ksldjfsjf
        bleu_arr = []
        for i in range(len(decoded_hashtags)):
            bleu_arr.append(BLEU_score(decoded_hashtags[i], decoded_predictions[i]))
        avg_blue = float(sum(bleu_arr))/len(bleu_arr)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        # loss = criterion(output, trg)
        # print(loss)

        if train:
            # global total_iterations
            # update_learning_rate(optimizer, total_iterations)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc)
            blues.append(avg_blue)
            idxs.append(idx.view(-1).clone())

        # loss_tracker.append(loss.item())
        blue_tracker.append(avg_blue)
        acc_tracker.append(acc)
        # print(acc)
        # for a in acc:
        #     acc_tracker.append(a.item())
        # print(loss_tracker.mean.value)
        fmt = '{:.4f}'.format
        tq.set_postfix(blue = fmt(blue_tracker.mean.value), acc = fmt(acc_tracker.mean.value))

    if not train:
        # answ = list(torch.cat(answ, dim=0))
        # accs = list(torch.cat(accs, dim=0))
        # idxs = list(torch.cat(idxs, dim=0))
        return ans_dict, cap_dict, pred_dict
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
    cap_vcb = test_loader.dataset.token_to_index
    hash_vcb = test_loader.dataset.answer_to_index
    inv_hash_dict = {v: k for k, v in hash_vcb.items()}
    inv_cap_dict = {v: k for k, v in cap_vcb.items()}

    net = model.Net(test_loader.dataset.num_tokens[0],test_loader.dataset.num_tokens[1], [], []).to(device)    
    # net = model.Net(train_loader.dataset.num_tokens[0],train_loader.dataset.num_tokens[1]).to(device)  
    # optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    # print(torch.load('logs/' + args.model_path)['weights'])
    net.load_state_dict(torch.load('logs/' + args.model_path)['weights'])

    tracker = utils.Tracker()
    # for k,v in vars(config).items():
    #     print(k)
    # sdfsd
    #BRING THIS BACK
    # criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    criterion = nn.CrossEntropyLoss(ignore_index = test_loader.dataset.answer_to_index['<pad>'])
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not k.startswith('os') and not k.startswith('expanduser') and not k.startswith('platform')}


    r = run(net, test_loader, tracker, criterion, cap_vcb, hash_vcb, train=False, prefix='test', epoch=0)
    with open('output/hashtags.csv', 'w') as f:
        for key in r[0].keys():
            f.write("%s,%s\n"%(key,r[0][key]))
    with open('output/captions.csv', 'w') as f:
        for key in r[1].keys():
            f.write("%s,%s\n"%(key,r[1][key]))
    with open('output/predictions.csv', 'w') as f:
        for key in r[2].keys():
            f.write("%s,%s\n"%(key,r[2][key]))
    # r = run(net, val_loader, optimizer, tracker, criterion, train=False, prefix='val', epoch=i)
    # print(train_loader.dataset.token_to_index)
    # results = {
    #     'name': name,
    #     'tracker': tracker.to_dict(),
    #     'config': config_as_dict,
    #     'weights': net.module.state_dict(),
    #     # 'eval': {
    #     #     'answers': r[0],
    #     #     'accuracies': r[1],
    #     #     'idx': r[2],
    #     # },
    #     'cap_vocab': test_loader.dataset.token_to_index,
    #     'hash_vocab': test_loader.dataset.answer_to_index
    # }


if __name__ == '__main__':
    main()
