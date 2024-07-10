import os
from scipy.stats import pearsonr
from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from configs.configs import args
from dataset_rise3k_45 import fetch_dataset
from model_ker_wheat import build_model
from losses import build_loss
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
from utils import *
from tensorboardX import SummaryWriter

def train_one_epoch(i_epoch, loader, model, criterion, optimizer, args):
    print(f"start epoch [{i_epoch}/{args.epoch}]")

    model.train()
    loss = 0
    for step, data in enumerate(loader):
        x = data["input"] 
        label = data["label"]

        if args.cuda:
            x = x.cuda()
            label = label.cuda()

        output = model(x)
        loss = criterion(output.squeeze(-1), label)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if step % args.log_iter == 0:
            print(f"epoch [{i_epoch}/{args.epoch}] step [{step}/{len(loader)}] loss:{loss}")
    return loss


@torch.no_grad()
def test_one_epoch(i_epoch, loader, model, criterion, args):
    print(f"test epoch [{i_epoch}/{args.epoch}]")

    correct_results = {}
    total_results = {}
    total_loss = 0.0

    model.eval()
    for step, data in enumerate(loader):
        x = data["input"] 
        label = data["label"]

        if args.cuda:
            x = x.cuda()
            label = label.cuda()
        
        with torch.no_grad():
            output = model(x)
            loss = criterion(output.squeeze(-1), label)
            total_loss += loss.detach().cpu().item()

            predicted_labels = torch.argmax(output, dim=1)
            correct_predictions = (predicted_labels == label).float() 
            for class_label in torch.unique(label):
                class_mask = (label == class_label).float()
                class_correct = (correct_predictions * class_mask).sum().item()
                class_total = class_mask.sum().item()
                
                if class_label.item() not in correct_results:
                    correct_results[class_label.item()] = 0 
                if class_label.item() not in total_results:
                    total_results[class_label.item()] = 0 

                correct_results[class_label.item()] += class_correct
                total_results[class_label.item()] += class_total

    total_loss /= len(loader)

    total_acc = sum(correct_results.values()) / sum(total_results.values())
    # for key in total_results.keys():
    #     print("class {} accuracy: {:.2f}".format(key, correct_results[key] / total_results[key]))
    #     logger.info("class {} accuracy: {:.2f}".format(key, correct_results[key] / total_results[key]))        

    print("total accuracy: {:.2f}".format(total_acc))
    metric_value = total_acc

    print(f"test epoch[{i_epoch}/{args.epoch}] test loss:{total_loss}")

    return {"loss": total_loss, "accuracy": metric_value}



def normal_train_test(train_loader, test_loader, model, optimizer, criterion, args):

    # 获取当前日期和时间
    current_date = datetime.now()

    # 提取年、月、日
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day
    epoch = args.epoch
    best_metric = -1.0
    name = 'logs/'+str(current_year)+str(current_month)+str(current_day)+'_'+args.dataset_task_name+'_'+str(args.which_k)+'_'+str(args.epoch)+'_'+str(next(model.parameters()).device)
    writer = SummaryWriter(name)
    # optimizer = optim.AdamW([
    #     {"params": model.parameters(), "lr":lr, "weight_decay":weight_decay}
    # ])

    
    for i in tqdm(range(1, epoch+1)):
        train_loss = train_one_epoch(i, train_loader, model, criterion, optimizer, args)
        
        test_result = test_one_epoch(i, test_loader, model, criterion, args)
        writer.add_scalar('Train/loss', train_loss, i)
        writer.add_scalar('Test/loss', test_result["loss"], i)
        writer.add_scalar('Test/accuracy',test_result["accuracy"], i)
        if best_metric < test_result["accuracy"]:
            best_metric = test_result["accuracy"]
            name_pth = './checkpoints/checkpoint_'+str(current_year)+str(current_month)+str(current_day)+'_'+args.dataset_task_name+'_'+str(args.which_k)+'_'+str(args.epoch)+'_'+str(next(model.parameters()).device)+'.pth'
            torch.save(model.state_dict(), name_pth)

    print(f"finish train | best metric is {best_metric}")
    writer.close()



def main():
    init_seed(args.seed)

    # prepare data
    batch_size = args.batch_size
    num_workers = args.num_workers
    torch.distributed.init_process_group(backend="nccl")
    train_dataset, test_dataset = fetch_dataset(args)
    print(len(train_dataset))
    print(len(test_dataset))
    print('dataset finished')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
            num_workers=num_workers, drop_last=False, sampler=DistributedSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    print('loader finished')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # build model & loss
    model = build_model(args)
    model.to(device)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr":args.lr, "weight_decay":args.weight_decay}
    ])
    if args.cuda:
        model = model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level = 'O1')
    model = DistributedDataParallel(model)
    criterion = build_loss(args)

    normal_train_test(train_loader, test_loader, model, optimizer, criterion, args)


if __name__ == "__main__":
    main()