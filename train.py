import os
import torch
import models
import argparse
import utils as U
import numpy as np
import pandas as pd
import torch.nn.functional as F

from apex import amp
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from torchvision import transforms

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser(description='Train a model with the parameters specified.')

    parser.add_argument('-s', '-seed', type=int, default=0, help='Specify manual seed')
    parser.add_argument('-m', '-model', choices=models.get_available_models(),
                        type=str, default='effnet-b5', help='Specify the type of model to train')
    parser.add_argument('-v', '-variant', choices=[0, 1, 2], type=int, default=0,
                        help='Especificar variante de entrenamiento 0, 1 or 2')
    parser.add_argument('-epochs', type=int, default=30, help='Total number of epochs')
    parser.add_argument('-epoch_size', type=int, default=2500, help='Size (in num of batches) of each epoch')
    parser.add_argument('-batch_size', type=int, default=42, help='Batch size')
    parser.add_argument('--local_rank', type=int, required=True, default=0)
    parser.add_argument('--ngpu', type=int, default=1, required=True, help='Number of GPUs')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for the DataLoader')
    parser.add_argument('--optim_step', type=int, default=1, help='Specify the number of iterations before optimizer updates')
    parser.add_argument('--distributed', type=bool, default=True, help='Specify whether to use Distributed Model or not')
    parser.add_argument('--amp', type=bool, default=False, help='Specify whether to use Nvidia Automatic Mixed Precision or not')
    parser.add_argument('--data_path', type=str, default='../data', help='Specify path to data.')

    args = parser.parse_args()

    # Load the arguments
    seed = args.s
    model_name = args.m
    variant = args.v
    epochs = args.epochs
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    n_workers = args.n_workers
    optim_step = args.optim_step
    distributed = args.distributed
    amp_ = args.amp
    data_path = args.data_path

    path_to_save = f'models/{model_name}_{seed}_v{variant}.pth'

    assert distributed or not amp_, "Mixed precision only allowed in distributed training mode."

    best_loss = 1000 # Sets a high initial best loss so the first model derived is saved

    if distributed:

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set seed manually for reproducibility
    torch.manual_seed(seed)

    # Load the configuration
    model, resize, criterion, optimizer, scheduler, normalization, desc = models.load_config(model_name, variant, epochs, epoch_size)
    model = model.cuda()

    if distributed:
        model = convert_syncbn_model(model)
    if amp_:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')

    if distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Generate tensorboard metrics report.
    writer = SummaryWriter(f'runs/{model_name}_v{variant}_{seed}', flush_secs=15)

    if args.local_rank == 0:
        writer.add_text('Description', str(desc), global_step=0)

    # Initialize datasets
    dataset = DeepFakeClassifierDataset(crops_dir='crops', data_path=f'{data_path}', hardcore=True, normalize = normalization, folds_csv=f'{data_path}/folds.csv', fold=3, transforms=U.create_train_transforms(resize))

    val_dataset = DeepFakeClassifierDataset(crops_dir='crops',  data_path=f'{data_path}', mode='val', normalize = normalization, folds_csv=f'{data_path}/folds.csv', fold=3, reduce_val=False, transforms=U.create_val_transforms(resize))
    val_dataset.reset(1, seed)

    for epoch in range(epochs):

        model.train()

        dataset.reset(epoch, seed)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            sampler.set_epoch(epoch)
        else:
            sampler = None

        train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, args.local_rank, n_workers, optim_step, amp_)


        if args.local_rank == 0:
            model.eval()
            loss = validate(model, val_dataset, epoch, criterion, min(2*batch_size, 64), writer, args.local_rank)

            if loss < best_loss:
                to_save = { 'epoch': epoch,
                            'variant': variant,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'best_loss': loss,}

                torch.save(to_save, path_to_save)


def train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, local_rank, n_workers, optim_step, amp_):

    losses = U.AverageMeter()
    f_losses = U.AverageMeter()
    r_losses = U.AverageMeter()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=sampler is None, sampler=sampler, pin_memory=False, drop_last=True)

    optimizer.zero_grad()

    for idx, (data) in enumerate(tqdm(loader, desc=f'Epoch {epoch}: lr: {scheduler["scheduler"].get_lr()}', total=epoch_size)):

        inputs, labels = data['image'].cuda(), data['labels'].float().cuda()

        valid_idx = data["valid"].float().cuda() > 0

        outputs = model(inputs)

        outputs = outputs[valid_idx]
        labels = labels[valid_idx]

        if labels.size(0) < 1:
            continue

        loss = criterion(outputs, labels)

        losses.update(loss.item(), labels.size(0))

        if amp_:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)

        if idx%optim_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        fake_idx = labels > .5
        real_idx = labels < .5

        fake_loss = F.binary_cross_entropy_with_logits(outputs[fake_idx], labels[fake_idx]).item() if fake_idx.size(0) > 0 else 0.
        real_loss = F.binary_cross_entropy_with_logits(outputs[real_idx], labels[real_idx]).item() if real_idx.size(0) > 0 else 0.

        f_losses.update(fake_loss, fake_idx.size(0) if fake_idx.size(0) > 0 else 1)
        r_losses.update(real_loss, real_idx.size(0) if real_idx.size(0) > 0 else 1)

        if scheduler['mode'] == 'iteration':
            scheduler['scheduler'].step()

        if idx > epoch_size - 1:
            break

    if scheduler['mode'] == 'epoch':
        scheduler['scheduler'].step()

    if local_rank == 0:
        writer.add_scalar('Train/Total Loss', losses.avg, global_step=epoch)
        writer.add_scalar('Train/Fake Loss', f_losses.avg, global_step=epoch)
        writer.add_scalar('Train/Real Loss', r_losses.avg, global_step=epoch)


def validate(model, dataset, epoch, criterion, batch_size, writer, local_rank):

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle= None, pin_memory=True)

    losses = U.AverageMeter()
    f_losses = U.AverageMeter()
    r_losses = U.AverageMeter()

    with torch.no_grad():

        for data in tqdm(loader):

            # Get the inputs and labels
            inputs, labels, img_name, valid, rotations = [*data.values()]
            inputs, labels = inputs.cuda(), labels.float().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Save statistics
            losses.update(loss.item(), outputs.size(0))

            fake_idx = labels > .5
            real_idx = labels < .5

            fake_loss = F.binary_cross_entropy_with_logits(outputs[fake_idx], labels[fake_idx]).item() if fake_idx.size(0) > 0 else 0.
            real_loss = F.binary_cross_entropy_with_logits(outputs[real_idx], labels[real_idx]).item() if real_idx.size(0) > 0 else 0.

            f_losses.update(fake_loss, fake_idx.size(0) if fake_idx.size(0) > 0 else 1)
            r_losses.update(real_loss, real_idx.size(0) if real_idx.size(0) > 0 else 1)

        if local_rank ==0:

            writer.add_scalar('Validation/Total Loss', losses.avg, global_step = epoch)
            writer.add_scalar('Validation/Fake Loss - val', f_losses.avg, global_step=epoch)
            writer.add_scalar('Validation/Real Loss - val', r_losses.avg, global_step=epoch)

        return losses.avg

if __name__ == '__main__':

    main()