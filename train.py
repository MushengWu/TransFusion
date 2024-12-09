import os
import time
import numpy as np
from tqdm import trange
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter

from utils import AverageMeter


def save_checkpoint_loss(model,
                         epoch,
                         args,
                         filename='best_loss_model.pth',
                         best_loss=0.,
                         optimizer=None,
                         scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_loss": best_loss,
        "state_dict": state_dict
    }

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    filename = os.path.join(args.log_dir, filename)
    torch.save(save_dict, filename)
    print('Saving the best loss checkpoint in: ', filename)


def save_checkpoint_acc(model,
                        epoch,
                        args,
                        filename='best_acc_model.pt',
                        best_acc=0.,
                        optimizer=None,
                        scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict
    }

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    filename = os.path.join(args.log_dir, filename)
    torch.save(save_dict, filename)
    print('Saving the best acc checkpoint in: ', filename)


def save_checkpoint_period(model,
                           epoch,
                           args,
                           acc,
                           optimizer=None,
                           scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_acc": acc,
        "state_dict": state_dict
    }

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    filename = os.path.join(args.log_dir, 'checkpoint_latest.pt')
    torch.save(save_dict, filename)
    print('Saving checkpoint every period ({}/{}) in: '.format(epoch, args.epochs), filename)


def print_training_logs(epoch,
                        train_loss, train_acc,
                        val_loss, val_acc):
    logs_table = PrettyTable()
    logs_table.title = f'epochs: {epoch}  Metrics: Dice, Hausdorff Distance(95%)'
    logs_table.field_names = ['', 'Loss', 'Accuracy_1', 'Accuracy_2']
    logs_table.add_row(['Train', f'{train_loss}', f'{train_acc}', 'None'])
    logs_table.add_row(['Val', f'{val_loss}', f'{val_acc[0]}', f'{val_acc[1]}'])
    print(logs_table)


def Train_epoch(args, model, device, loader, loss_func, optimizer, scaler):
    model.train()

    train_loss = AverageMeter()

    for idx, data in enumerate(loader):
        if isinstance(data, list):
            image = data[0]["image"].to(device)
            mask = data[0]["mask"].to(device)
        else:
            image = data["image"].to(device)
            mask = data["mask"].to(device)

        for param in model.parameters():
            param.grad = None

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(image)
            loss = loss_func(output, mask)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

        train_loss.update(loss.item(), n=args.batch_size)
    return train_loss.avg


def Val_epoch(args, model, device, loader, loss_func, acc_func, model_inferer):
    model.eval()

    val_loss = AverageMeter()
    val_acc_1 = AverageMeter()
    val_acc_2 = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(loader):
            if isinstance(data, list):
                image = data[0]["image"].to(device)
                mask = data[0]["mask"].to(device)
            else:
                image = data["image"].to(device)
                mask = data["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                if model_inferer is not None:
                    output = model_inferer(image)
                else:
                    output = model(image)

            loss = loss_func(output, mask)

            output = torch.sigmoid(output)
            output = (output >= 0.5) * 1.0
            acc_1 = acc_func[0](output, mask, Binary=False, Sigmoid=False)
            acc_2 = acc_func[1](output, mask)

            val_loss.update(loss.item(), n=1)
            val_acc_1.update(acc_1.item(), n=1)
            val_acc_2.update(acc_2.item(), n=1)
    return val_loss.avg, [val_acc_1.avg, val_acc_2.avg]


def Train(args,
          model,
          device,
          train_loader,
          val_loader,
          loss_func,
          acc_func,
          optimizer,
          scheduler=None,
          saving_mode='acc',
          model_inferer=None,
          start_epoch=0):
    time_stamp = time.strftime('%Y-%m-%d %H：%M：%S', time.localtime())
    print("Start time: ", time_stamp)

    if args.log_dir_name is not None:
        log_dir_name = args.log_dir_name
        log_dir = os.path.join(args.log_dir, log_dir_name, time_stamp)
    else:
        log_dir = os.path.join(args.log_dir, time_stamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    writer = SummaryWriter(logdir=log_dir)
    print("Writing Tensorboard logs to ", str(Path(log_dir).absolute()))

    min_loss, max_acc = (np.inf, -np.inf)

    scaler = None
    if args.amp:
        scaler = GradScaler()

    for epoch in trange(start_epoch, args.epochs):
        train_loss = Train_epoch(args,
                                 model,
                                 device,
                                 train_loader,
                                 loss_func=loss_func,
                                 optimizer=optimizer,
                                 scaler=scaler)

        val_loss, val_acc = Val_epoch(args,
                                      model,
                                      device,
                                      val_loader,
                                      loss_func=loss_func,
                                      acc_func=acc_func,
                                      model_inferer=model_inferer)

        writer.add_scalar("train_loss", train_loss, epoch)

        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc_1", val_acc[0], epoch)
        writer.add_scalar("val_acc_2", val_acc[1], epoch)

        assert saving_mode == 'loss' or saving_mode == 'acc'
        if saving_mode == "loss":
            if val_loss < min_loss:
                min_loss = val_loss
                save_checkpoint_loss(model,
                                     epoch,
                                     args,
                                     best_loss=min_loss,
                                     optimizer=optimizer,
                                     scheduler=optimizer)
        else:
            if val_acc[0] > max_acc:
                max_acc = val_acc[0]
                save_checkpoint_acc(model,
                                    epoch,
                                    args,
                                    best_acc=max_acc,
                                    optimizer=optimizer,
                                    scheduler=optimizer)

        if (epoch + 1) % args.save_period == 0:
            save_checkpoint_period(model,
                                   epoch,
                                   args,
                                   acc=val_acc,
                                   optimizer=optimizer,
                                   scheduler=optimizer)

        print_training_logs(epoch,
                            train_loss=train_loss, train_acc=None,
                            val_loss=val_loss,
                            val_acc=val_acc)

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', max_acc)

    return max_acc
