import os
import torch
import models
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from apex import amp
from tqdm import tqdm
from utils import DataAugmentationTransforms as DAT, AverageMeter, read_training_dataset, create_train_transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from torchvision import transforms
from sklearn.metrics import accuracy_score

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser(description='Train a model with the parameters specified.')

    parser.add_argument('--seed', type=int, default=101, help='Specify manual seed')
    parser.add_argument('--model', choices=models.get_available_models(),
                        type=str, default='efficientnet-b3', help='Specify the type of model to train')
    parser.add_argument('--v', '-variant', choices=[0, 1, 2], type=int, default=0,
                        help='Specify training variant')
    parser.add_argument('--epochs', type=int, default=30, help='Total number of epochs')
    parser.add_argument('--epoch_size', type=int, default=300, help='Size (in num of batches) of each epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--local_rank', type=int, required=True, default=0)
    parser.add_argument('--ngpu', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for the DataLoader')
    parser.add_argument('--optim_step', type=int, default=1,
                        help='Specify the number of iterations before optimizer updates')
    parser.add_argument('--distributed', action='store_true',
                        help='Specify whether to use Distributed Model or not')
    parser.add_argument('--amp', action='store_true',
                        help='Specify whether to use Nvidia Automatic Mixed Precision or not')
    parser.add_argument('--data_path', type=str, default='../datasets/mtcnn', help='Specify path to data.')

    parser.add_argument('--data_augment', choices=['no_da', 'simple_da', 'occlusions_da', 'cutout_da'],
                        type=str, default='cutout_da', help='Specify training variant')

    parser.add_argument('--da_dataset', choices=['faceforensics', 'dfdc', 'other'],
                        type=str, default='faceforensics', help='Specify type of base data augmentation transformations.')

    args = parser.parse_args()

    # Load the arguments
    seed = args.seed
    model_name = args.model
    variant = args.v
    epochs = args.epochs
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    n_workers = args.n_workers
    optim_step = args.optim_step
    distributed = args.distributed
    amp_ = args.amp
    data_path = args.data_path
    data_augment = args.data_augment
    da_dataset = args.da_dataset

    # Path to save
    execution_id = f'{model_name}-tum_{data_augment}_{da_dataset}_v{variant}_{seed}'
    path_to_save = f'models/{execution_id}.pth'

    assert distributed or not amp_, "Mixed precision only allowed in distributed training mode."

    best_loss = 1000 # Sets a high initial best loss so the first model derived is saved

    if distributed:

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set seed manually for reproducibility
    torch.manual_seed(seed)

    # Load the configuration
    model, resize, criterion, optimizer, scheduler, normalization, desc = models.load_config(model_name, variant,
                                                                                             epochs, epoch_size)
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
    writer = SummaryWriter(f'runs/{execution_id}', flush_secs=15)

    if args.local_rank == 0:
        writer.add_text('Description', str(desc), global_step=0)

    train_transform = create_train_transforms(resize, option=data_augment, dataset=da_dataset,
                                              shift_limit=.01, scale_limit=.05, rotate_limit=5,)
    # Train split containing both raw, c23 and c40
    train, _, _ = read_training_dataset(data_path, train_transform,
                                        max_videos = 1000, max_images_per_video=10, normalization=normalization)

    # For validation, different levels of compression go separately.
    raw_val = c23_val = c40_val = None

    if args.local_rank == 0:
        # validation set specific for raw
        raw_filter = lambda x: ('original' in x or 'neural' in x) and 'raw' in x
        _, raw_val, _ = read_training_dataset(data_path, DAT.create_val_transforms(resize), normalization=normalization,
                                              filter=raw_filter)
        del _
    
        # validation set specific for c23
        c23_filter = lambda x: ('original' in x or 'neural' in x) and 'c23' in x
        _, c23_val, _ = read_training_dataset(data_path, DAT.create_val_transforms(resize), normalization=normalization,
                                              filter=c23_filter)
        del _

        # validation set specific for c40
        c40_filter = lambda x: ('original' in x or 'neural' in x) and 'c40' in x
        _, c40_val, _ = read_training_dataset(data_path, DAT.create_val_transforms(resize), normalization=normalization,
                                              filter=c40_filter)
        del _

    val_datasets = {'c40' : c40_val, 'c23' : c23_val, 'raw': raw_val}

    for epoch in range(epochs):

        model.train()

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(train)
            sampler.set_epoch(epoch)
        else:
            sampler = None

        train_iteration(model, train, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer,
                        args.local_rank, n_workers, optim_step, amp_)


        if args.local_rank == 0:

            model.eval()

            c23_loss = validate(model, val_datasets, 'c23', epoch, batch_size, writer, args.local_rank, execution_id)
            c40_loss = validate(model, val_datasets, 'c40', epoch, batch_size, writer, args.local_rank, execution_id)
            raw_loss = validate(model, val_datasets, 'raw', epoch, batch_size, writer, args.local_rank, execution_id)

            if np.mean((c23_loss, c40_loss, raw_loss)) < best_loss:
                print('Saving checkpoint at epoch', epoch , '!!')
                to_save = { 'epoch': epoch,
                            'variant': variant,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'best_loss': np.mean((c23_loss, c40_loss, raw_loss)),}

                torch.save(to_save, path_to_save)


def train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer,
                    local_rank, n_workers, optim_step, amp_):

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=sampler is None, sampler=sampler,
                        pin_memory=False, drop_last=True)

    optimizer.zero_grad()

    pbar = tqdm(loader, total=epoch_size)

    for idx, (video_ids, frame_ids, images, targets) in enumerate(pbar):

        inputs, labels = images.cuda(), targets.float().cuda()

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
            print(epoch, idx, 'optimizing')
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

        pbar.set_description(f'Epoch {epoch} - lr: {scheduler["scheduler"].get_lr()[-1]:.5f} loss:{losses.avg:.4f}')

        if idx > epoch_size - 1:
            break



    if scheduler['mode'] == 'epoch':
        scheduler['scheduler'].step()

    if local_rank == 0:
        writer.add_scalar('Train/Total Loss', losses.avg, global_step=epoch)
        writer.add_scalar('Train/BCE Fake Loss', f_losses.avg, global_step=epoch)
        writer.add_scalar('Train/BCE Real Loss', r_losses.avg, global_step=epoch)


def validate(model, datasets, key, epoch, batch_size, writer, local_rank, execution_id):

    loader = DataLoader(datasets[key], batch_size=batch_size, num_workers=8, shuffle=None, pin_memory=True)

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    pairs_prob_and_target = []

    with torch.no_grad():

        pbar = tqdm(loader)

        for idx, data in enumerate(pbar):

            # Get the inputs and labels
            video_ids, frame_ids, images, targets = data
            inputs, labels = images.cuda(), targets.float().cuda()
            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            probs = torch.sigmoid(outputs) # Output to probability.

            # Measure accuracy based on thresholds

            probs_and_targets = torch.stack((probs.cpu(),targets)).squeeze(2)
            pairs_prob_and_target.append(probs_and_targets)

            # Save statistics
            losses.update(loss.item(), outputs.size(0))

            fake_idx = labels > .5
            real_idx = labels < .5

            fake_loss = F.binary_cross_entropy_with_logits(outputs[fake_idx], labels[fake_idx]).item() if fake_idx.any() else 0.
            real_loss = F.binary_cross_entropy_with_logits(outputs[real_idx], labels[real_idx]).item() if real_idx.any() else 0.

            f_losses.update(fake_loss, fake_idx.size(0) if fake_idx.any() > 0 else 0)
            r_losses.update(real_loss, real_idx.size(0) if real_idx.any() > 0 else 0)

            pbar.set_description(f'Epoch {epoch} val. {key} Loss:{losses.avg:.4f}-Fake.Loss:{f_losses.avg:.4f}-Real.Loss:{r_losses.avg:.4f}')


        if local_rank ==0:
            writer.add_scalar(f'Validation-{key}/Loss/BCE Total Loss', losses.avg, global_step = epoch)
            writer.add_scalar(f'Validation-{key}/Loss/BCE Fake Loss', f_losses.avg, global_step=epoch)
            writer.add_scalar(f'Validation-{key}/Loss/BCE Real Loss', r_losses.avg, global_step=epoch)

            ppt = torch.cat(pairs_prob_and_target, dim=1)

            accuracy_02 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.2).float().numpy())
            accuracy_05 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.5).float().numpy())
            accuracy_07 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.7).float().numpy())

            os.makedirs(f'outputs/{execution_id}/{key}', exist_ok=True)

            pd.DataFrame(ppt.transpose(1,0).numpy(), columns=['prob', 'target']).to_csv(f'outputs/{execution_id}/{key}/val_vector_epoch{epoch:02}.csv', index=False)

            writer.add_scalar(f'Validation-{key}/Accuracy/Threshold 0.2', accuracy_02, global_step=epoch)
            writer.add_scalar(f'Validation-{key}/Accuracy/Threshold 0.5', accuracy_05, global_step=epoch)
            writer.add_scalar(f'Validation-{key}/Accuracy/Threshold 0.7', accuracy_07, global_step=epoch)



    return losses.avg



if __name__ == '__main__':

    main()