import os
import torch
import models
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from apex import amp
from tqdm import tqdm
from utils import DataAugmentationTransforms as DAT, AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from torchvision import transforms
from data_loader import get_loader, read_dataset, CompositeDataset

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
    writer = SummaryWriter(f'runs/{model_name}-tum_DA_v{variant}_{seed}', flush_secs=15)

    if args.local_rank == 0:
        writer.add_text('Description', str(desc), global_step=0)

    # Initialize datasets


    train, _, _ = read_training_dataset(data_path, DAT.create_train_transforms(resize), normalization=normalization)
    _, val, _ = read_training_dataset(data_path, DAT.create_val_transforms(resize), normalization=normalization)
    del _

    for epoch in range(epochs):

        model.train()

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(train)
            sampler.set_epoch(epoch)
        else:
            sampler = None

        train_iteration(model, train, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, args.local_rank, n_workers, optim_step, amp_)

        if args.local_rank == 0:
            model.eval()

            loss = validate(model, val, epoch, criterion, min(2*batch_size, 64), writer, args.local_rank)

            if loss < best_loss:
                to_save = { 'epoch': epoch,
                            'variant': variant,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'best_loss': loss,}

                torch.save(to_save, path_to_save)


def train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, local_rank, n_workers, optim_step, amp_):

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=sampler is None, sampler=sampler, pin_memory=False, drop_last=True)

    optimizer.zero_grad()

    for idx, (video_ids, frame_ids, images, targets) in enumerate(tqdm(loader, desc=f'Epoch {epoch}: lr: {scheduler["scheduler"].get_lr()}', total=epoch_size)):

        inputs, labels = images.cuda(), targets.float().cuda()

        #valid_idx = data["valid"].float().cuda() > 0

        outputs = model(inputs)

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

        fake_loss = F.binary_cross_entropy_with_logits(outputs[fake_idx], labels[fake_idx]).item() if fake_idx.any() else 0.
        real_loss = F.binary_cross_entropy_with_logits(outputs[real_idx], labels[real_idx]).item() if real_idx.any() else 0.

        f_losses.update(fake_loss, fake_idx.size(0) if fake_idx.any() > 0 else 0)
        r_losses.update(real_loss, real_idx.size(0) if real_idx.any() > 0 else 0)

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

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    with torch.no_grad():

        for idx, data in enumerate(tqdm(loader, desc=f'Validation epoch {epoch} - local_rank={local_rank}')):

            # Get the inputs and labels
            video_ids, frame_ids, images, targets = data
            inputs, labels = images.cuda(), targets.float().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Save statistics
            losses.update(loss.item(), outputs.size(0))

            fake_idx = labels > .5
            real_idx = labels < .5

            fake_loss = F.binary_cross_entropy_with_logits(outputs[fake_idx], labels[fake_idx]).item() if fake_idx.any() else 0.
            real_loss = F.binary_cross_entropy_with_logits(outputs[real_idx], labels[real_idx]).item() if real_idx.any() else 0.

            f_losses.update(fake_loss, fake_idx.size(0) if fake_idx.any() > 0 else 0)
            r_losses.update(real_loss, real_idx.size(0) if real_idx.any() > 0 else 0)

        if local_rank ==0:
            writer.add_scalar('Validation/Total Loss', losses.avg, global_step = epoch)
            writer.add_scalar('Validation/Fake Loss - val', f_losses.avg, global_step=epoch)
            writer.add_scalar('Validation/Real Loss - val', r_losses.avg, global_step=epoch)

        return losses.avg


def read_training_dataset(data_dir, transform, normalization, max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits'):

    datasets = read_dataset(data_dir, normalization = normalization, transform=transform, max_images_per_video=max_images_per_video, max_videos=max_videos,
                            window_size=window_size, splits_path=splits_path)

    # only neural textures and original
    datasets = {
        k: v for k, v in datasets.items()
        if 'original' in k or 'neural' in k
    }
    print('Using training data: ')
    print('\n'.join(sorted(datasets.keys())))

    trains, vals, tests = [], [], []
    for data_dir_name, dataset in datasets.items():
        train, val, test = dataset
        # repeat original data multiple times to balance out training data
        compression = data_dir_name.split('_')[-1]
        num_tampered_with_same_compression = len({x for x in datasets.keys() if compression in x}) - 1
        count = 1 if 'original' not in data_dir_name else num_tampered_with_same_compression
        for _ in range(count):
            trains.append(train)
        vals.append(val)
        tests.append(test)
    return CompositeDataset(*trains), CompositeDataset(*vals), CompositeDataset(*tests)


if __name__ == '__main__':

    main()