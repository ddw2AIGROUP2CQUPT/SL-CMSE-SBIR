import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Block_lstm, Linear, Linear_s2
from torch import optim
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SLFIR_Model(nn.Module):
    def __init__(self, opt):
        super(SLFIR_Model, self).__init__()

        self.backbone_network = InceptionV3_Network()
        self.backbone_network.load_state_dict(torch.load(opt.backbone_model_dir, map_location=opt.device))
        self.backbone_network.to(opt.device)
        self.backbone_network.fixed_param()
        self.backbone_network.eval()

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.attn_network = Attention()
        self.attn_network.load_state_dict(torch.load(opt.attn_model_dir, map_location=opt.device))
        self.attn_network.to(opt.device)
        self.attn_network.fixed_param()
        self.attn_network.eval()

        self.linear_network = Linear(opt.feature_num)
        self.linear_network.load_state_dict(torch.load(opt.linear_model_dir, map_location=opt.device))
        self.linear_network.to(opt.device)
        self.linear_network.fixed_param()
        self.linear_network.eval()

        if opt.stage2_net == 'LSTM':
            self.stage2_network = Block_lstm(opt)
        elif opt.stage2_net == 'MLP':
            self.stage2_network = Linear_s2(opt.feature_num, opt.condition_num)
        else:
            print("不支持的stage2 network")
            exit(0)
        self.stage2_network.apply(init_weights)
        self.stage2_network.train()
        self.stage2_network.to(opt.device)
        self.stage2_net_train_params = self.stage2_network.parameters()

        self.optimizer = optim.Adam([
            {'params': self.stage2_net_train_params, 'lr': opt.lr}])

        self.loss = nn.TripletMarginLoss(margin=0.3, p=2)
        self.opt = opt

    def train_model(self, batch):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.linear_network.eval()
        self.stage2_network.train()
        loss = 0

        for idx in range(len(batch['sketch_seq'])):
            positive_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['positive_img'][idx].unsqueeze(0).to(self.opt.device))))
            negative_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['negative_img'][idx].unsqueeze(0).to(self.opt.device))))
            sketch_seq_feature = self.stage2_network(self.cat_feature(self.attn_network(
                self.backbone_network(batch['sketch_seq'][idx].to(self.opt.device))),
                batch['condition'][idx].repeat(batch['sketch_seq'][idx].shape[0], 1).to(self.opt.device)))

            positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
            loss += self.loss(sketch_seq_feature, positive_feature, negative_feature)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_NN(self, dataloader):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.stage2_network.eval()

        self.Sketch_Array_Test = []
        self.Image_Array_Test = []
        self.Sketch_Path = []
        self.Image_Path = []

        for idx, batch in enumerate(dataloader):
            positive_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['positive_img'].to(self.opt.device))))

            sketch_feature = self.cat_feature(self.attn_network(
                self.backbone_network(batch['sketch_seq'].squeeze(0).to(self.opt.device))),
                batch['condition'].repeat(batch['sketch_seq'].squeeze(0).shape[0], 1).to(self.opt.device))

            self.Sketch_Array_Test.append(sketch_feature)
            self.Sketch_Path.append(batch['sketch_seq_paths'])

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

        exps = np.linspace(1, num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        factor = np.exp(1 - exps) / np.e
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
     
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            mean_rank_ourB = []
            mean_rank_ourA = []

            for i_sketch in range(sanpled_batch.shape[0]):

                sketch_feature = self.stage2_network(sanpled_batch[:i_sketch+1].to(self.opt.device))

                s_path =self.Sketch_Path[i_batch]
                s_path=''.join(s_path)
                positive_name = '_'.join(s_path.split('/')[-1].split('_')[:-1])
                position_query = self.Image_Path.index(positive_name)

                target_distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(self.opt.device)), self.Image_Array_Test[position_query].unsqueeze(0).to(self.opt.device))
                distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(self.opt.device)), self.Image_Array_Test.to(self.opt.device))
               
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)

                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item()*factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])

            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanMB, meanMA, meanOurB, meanOurA

    def cat_feature(self, feature_attn, extend_feature):
        return torch.cat([feature_attn, extend_feature], dim=1)