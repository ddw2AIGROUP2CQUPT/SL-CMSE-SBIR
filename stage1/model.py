import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Linear
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F

class SLFIR_Model(nn.Module):
    def __init__(self, hp):
        super(SLFIR_Model, self).__init__()

        self.backbone_network = InceptionV3_Network()
        self.backbone_train_params = self.backbone_network.parameters()

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.attn_network = Attention()
        self.attn_network.apply(init_weights)
        self.attn_train_params = self.attn_network.parameters()

        self.linear_network = Linear(hp.feature_num)
        self.linear_network.apply(init_weights)
        self.linear_train_params = self.linear_network.parameters()

        self.optimizer = optim.Adam([
            {'params': filter(lambda param: param.requires_grad, self.backbone_train_params), 'lr': hp.backbone_lr},
            {'params': self.attn_train_params, 'lr': hp.lr},
            {'params': self.linear_train_params, 'lr': hp.lr}])
        # 训练的模型
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.hp = hp
    
    def train_model(self, batch):
        self.train()
        positive_feature = self.linear_network(self.attn_network(
            self.backbone_network(batch['positive_img'].to(self.hp.device))))
        negative_feature = self.linear_network(self.attn_network(
            self.backbone_network(batch['negative_img'].to(self.hp.device))))
        sample_feature = self.linear_network(self.attn_network(
            self.backbone_network(batch['sketch_img'].to(self.hp.device))))

        loss = self.loss(sample_feature, positive_feature, negative_feature)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_NN(self, dataloader):
        self.eval()

        self.Sketch_Array_Test = []
        self.Image_Array_Test = []
        self.Sketch_Path = []
        self.Image_Path = []
        for idx, batch in enumerate(dataloader):
            sketch_feature = self.attn_network(
                self.backbone_network(batch['sketch_img'].to(self.hp.device)))
            positive_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['positive_img'].to(self.hp.device))))
            self.Sketch_Array_Test.append(sketch_feature)
            self.Sketch_Path.append(batch['sketch_path'])
 
            for i_num, positive_path in enumerate(batch['positive_path']):
                if positive_path not in self.Image_Path:
                    self.Image_Path.append(batch['positive_path'][i_num])
                    self.Image_Array_Test.append(positive_feature[i_num])

        self.Sketch_Array_Test = torch.stack(self.Sketch_Array_Test)
        self.Image_Array_Test = torch.stack(self.Image_Array_Test)
        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1,num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        factor = np.exp(1 - exps) / np.e
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            mean_rank_ourB = []
            mean_rank_ourA = []
            
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = self.linear_network(sanpled_batch[i_sketch].unsqueeze(0).to(self.hp.device))

                s_path =self.Sketch_Path[i_batch]
                s_path=''.join(s_path)
                positive_path = '_'.join(s_path.split('/')[-1].split('_')[:-1])
                position_query = self.Image_Path.index(positive_path)

                target_distance = F.pairwise_distance(F.normalize(sketch_feature.to(self.hp.device)), self.Image_Array_Test[position_query].unsqueeze(0).to(self.hp.device))
                distance = F.pairwise_distance(F.normalize(sketch_feature.to(self.hp.device)), self.Image_Array_Test.to(self.hp.device))

                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)

                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])

            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]

        meanIOU = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanIOU, meanMA, meanOurB, meanOurA
