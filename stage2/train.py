from time import time
from model import SLFIR_Model
import time
import os
import torch
import numpy as np
import argparse
from dataset import *
from torch.utils.tensorboard import SummaryWriter

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ChairV2', help='ChairV2 / ShoeV2')
parser.add_argument('--root_dir', type=str, default="/home/ubuntu/workplace/benke-2020/chair_shoe/")
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--print_freq_iter', type=int, default=20)
parser.add_argument('--nThreads', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoches', type=int, default=300)
parser.add_argument('--feature_num', type=int, default=64)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--stage2_net', type=str, default='LSTM', help='LSTM / MLP')
hp = parser.parse_args()
hp.device = torch.device('cuda:' + str(hp.gpu_id) if torch.cuda.is_available() else 'cpu')

if hp.dataset_name == 'ShoeV2':
    hp.condition_num = 15
elif hp.dataset_name == 'ChairV2':
    hp.condition_num = 19

hp.backbone_model_dir = '../' +'stage1/models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(0) + '_backbone_best.pth'
hp.attn_model_dir = '../' + 'stage1/models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(0) + '_attn_best.pth'
hp.linear_model_dir = '../' + 'stage1/models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(0) + '_linear_best.pth'

print(hp)

tb_logdir = r"./run/"
slfir_model = SLFIR_Model(hp)
dataloader_sketch_train, dataloader_sketch_test = get_dataloader(hp)

def main_train():
    meanMB_buffer = 0
    real_p = [0, 0, 0, 0, 0, 0]
    loss_buffer = []
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    Top1_Song = [0]
    Top5_Song = [0]
    Top10_Song = [0]
    meanMB_Song = []
    meanMA_Song = []
    meanWMB_Song = []
    meanWMA_Song = []
    step_stddev = 0
    for epoch in range(hp.epoches):
        for i, sanpled_batch in enumerate(dataloader_sketch_train):
            start_time = time.time()
            loss_triplet = slfir_model.train_model(sanpled_batch)
            loss_buffer.append(loss_triplet)

            step_stddev += 1
            tb_writer.add_scalar('total loss', loss_triplet, step_stddev)
            print('epoch: {}, iter: {}, loss: {}, time cost{}'.format(epoch, step_stddev, loss_triplet, time.time()-start_time))

            if epoch >= 5 and step_stddev % hp.print_freq_iter==0:

                with torch.no_grad():
                    start_time = time.time()
                    top1, top5, top10, meanMB, meanMA, meanWMB, meanWMA = slfir_model.evaluate_NN(dataloader_sketch_test)
                    slfir_model.train()
                    print('Epoch: {}, Iteration: {}:'.format(epoch, step_stddev))
                    print("TEST A@1: {}".format(top1))
                    print("TEST A@5: {}".format(top5))
                    print("TEST A@10: {}".format(top10))
                    print("TEST M@B: {}".format(meanMB))
                    print("TEST M@A: {}".format(meanMA))
                    print("TEST W@MB: {}".format(meanWMB))
                    print("TEST W@MA: {}".format(meanWMA))
                    print("TEST Time: {}".format(time.time()-start_time))
                    Top1_Song.append(top1)
                    Top5_Song.append(top5)
                    Top10_Song.append(top10)
                    meanMB_Song.append(meanMB)
                    meanMA_Song.append(meanMA)
                    meanWMB_Song.append(meanWMB)
                    meanWMA_Song.append(meanWMA)
                    tb_writer.add_scalar('TEST A@1', top1, step_stddev)
                    tb_writer.add_scalar('TEST A@5', top5, step_stddev)
                    tb_writer.add_scalar('TEST A@10', top10, step_stddev)
                    tb_writer.add_scalar('TEST M@B', meanMB, step_stddev)
                    tb_writer.add_scalar('TEST M@A', meanMA, step_stddev)
                    tb_writer.add_scalar('TEST W@MB', meanWMB, step_stddev)
                    tb_writer.add_scalar('TEST W@MA', meanWMA, step_stddev)

                if meanMB > meanMB_buffer:
                   
                    torch.save(slfir_model.stage2_network.state_dict(), './models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(hp.condition_num) + '_' + str(hp.stage2_net) + '.pth')
                    
                    meanMB_buffer = meanMB

                    real_p = [top1, top5, top10, meanMA, meanWMB, meanWMA]
                    print('Model Updated')
                print('REAL performance: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {}, WMB: {}, WMA: {}'.format(real_p[0], real_p[1],
                                                                                                                real_p[2],
                                                                                                                meanMB_buffer,
                                                                                                                real_p[3],
                                                                                                                real_p[4],
                                                                                                                real_p[5]))

    print("TOP1_MAX: {}".format(max(Top1_Song)))
    print("TOP5_MAX: {}".format(max(Top5_Song)))
    print("TOP10_MAX: {}".format(max(Top10_Song)))
    print("meaIOU_MAX: {}".format(max((meanMB_Song))))
    print("meaMA_MAX: {}".format(max((meanMA_Song))))
    print("meanWMB_MAX: {}".format(max(meanWMB_Song)))
    print("meanWMA_MAX: {}".format(max(meanWMA_Song)))
    print(Top1_Song)
    print(Top5_Song)
    print(Top10_Song)
    print(meanMB_Song)
    print(meanMA_Song)
    print(meanWMB_Song)
    print(meanWMA_Song)

if __name__ == "__main__":
    main_train()




