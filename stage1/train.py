import torch
import time
from model import SLFIR_Model
from dataset import get_dataloader
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLFIR Model')
    parser.add_argument('--dataset_name', type=str, default='ChairV2', help='ChairV2 / ShoeV2')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/workplace/benke-2020/chair_shoe/")
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--backbone_lr', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--feature_num', type=int, default=64)
    hp = parser.parse_args()
    if hp.dataset_name == 'ShoeV2':
        hp.batchsize = 64
        hp.eval_freq_iter = 50
    elif hp.dataset_name == 'ChairV2':
        hp.batchsize = 32
        hp.eval_freq_iter = 20


    hp.device = torch.device("cuda:" + str(hp.gpu_id) if torch.cuda.is_available() else "cpu")
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = SLFIR_Model(hp)
    model.to(hp.device)
    step_count, top1, top5, top10, top50, top100 = -1, 0, 0, 0, 0, 0
    mean_IOU_buffer = 0
    real_p = [0, 0, 0, 0, 0, 0]

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)
            if step_count % hp.print_freq_iter == 0:
                print(
                    'Epoch: {}, Iteration: {}, Loss: {:.8f}, Top1_Accuracy: {:.5f}, Top5_Accuracy; {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format(
                        i_epoch, step_count, loss, top1, top5, top10, time.time() - start))

            if i_epoch >= 0 and step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    start_time = time.time()
                    top1, top5, top10, mean_IOU, mean_MA, mean_OurB, mean_OurA = model.evaluate_NN(dataloader_Test)
                    model.train()
                    print('Epoch: {}, Iteration: {}:'.format(i_epoch, step_count))
                    print("TEST A@1: {}".format(top1))
                    print("TEST A@5: {}".format(top5))
                    print("TEST A@10: {}".format(top10))
                    print("TEST M@B: {}".format(mean_IOU))
                    print("TEST M@A: {}".format(mean_MA))
                    print("TEST OurB: {}".format(mean_OurB))
                    print("TEST OurA: {}".format(mean_OurA))
                    print("TEST Time: {}".format(time.time() - start_time))
                if mean_IOU > mean_IOU_buffer:
                    torch.save(model.backbone_network.state_dict(),
                               './models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(
                                   0) + '_backbone_best.pth')
                    torch.save(model.attn_network.state_dict(),
                               './models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(
                                   0) + '_attn_best.pth')
                    torch.save(model.linear_network.state_dict(),
                               './models/' + hp.dataset_name + '_feature' + str(hp.feature_num) + '_condition' + str(
                                   0) + '_linear_best.pth')
                
                    mean_IOU_buffer = mean_IOU

                    real_p = [top1, top5, top10, mean_MA, mean_OurB, mean_OurA]

                    print('Model Updated')

