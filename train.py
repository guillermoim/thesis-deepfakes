import os
import torch
import models
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from apex import amp
from tqdm import tqdm
from utils import DataAugmentationTransforms as DAT, AverageMeter, create_train_transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dfdc_dataset import DeepFakeTrainingDataset
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
    parser.add_argument('--epoch_size', type=int, default=1200, help='Size (in num of batches) of each epoch')
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

    parser.add_argument('--da_dataset', choices=['faceforensics', 'dfdc', 'other', 'other_dfdc'],
                        type=str, default='dfdc',
                        help='Specify type of base data augmentation transformations.')

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


    execution_id = f'{model_name}-dfdc_{data_augment}_{da_dataset}_v{variant}_{seed}'
    path_to_save = f'models_dfdc/{execution_id}.pth'

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
    writer = SummaryWriter(f'runs_dfdc/{execution_id}', flush_secs=15)

    if args.local_rank == 0:
        writer.add_text('Description', str(desc), global_step=0)


    train_transforms = create_train_transforms(resize, option=data_augment, dataset=da_dataset)
    # Initialize datasets
    dataset = DeepFakeTrainingDataset(crops_dir='crops', data_path=f'{data_path}', hardcore=True, normalize = normalization, folds_csv=f'{data_path}/folds.csv', fold=8, transforms=train_transforms)

    val_dataset = DeepFakeTrainingDataset(crops_dir='crops',  data_path=f'{data_path}', mode='val', normalize = normalization, folds_csv=f'{data_path}/folds.csv', fold=8, reduce_val=True, transforms=DAT.create_val_transforms(resize))
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

            loss = validate(model, val_dataset, epoch, criterion, 2*batch_size, writer, execution_id, args.local_rank)

            if loss < best_loss:
                best_loss = loss

                to_save = { 'epoch': epoch,
                            'variant': variant,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'best_loss': best_loss,}

                torch.save(to_save, path_to_save)


def train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, local_rank, n_workers, optim_step, amp_):

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=sampler is None, sampler=sampler, pin_memory=False, drop_last=True)

    optimizer.zero_grad()

    for idx, (data) in enumerate(tqdm(loader, desc=f'Epoch {epoch}: lr: {scheduler["scheduler"].get_last_lr()}', total=epoch_size)):

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


def validate(model, dataset, epoch, criterion, batch_size, writer, execution_id, local_rank):

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle= None, pin_memory=True)

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    pairs_prob_and_target = []

    with torch.no_grad():

        for data in tqdm(loader):

            # Get the inputs and labels
            inputs, labels, img_name, valid = [*data.values()]
            inputs, labels = inputs.cuda(), labels.float().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)  # Output to probability.

            # Measure accuracy based on thresholds

            probs_and_targets = torch.stack((probs.cpu(), labels.cpu())).squeeze(2)
            pairs_prob_and_target.append(probs_and_targets)

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

            ppt = torch.cat(pairs_prob_and_target, dim=1)

            accuracy_02 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.2).float().numpy())
            accuracy_05 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.5).float().numpy())
            accuracy_07 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.7).float().numpy())

            os.makedirs(f'outputs_dfdc/{execution_id}', exist_ok=True)

            pd.DataFrame(ppt.transpose(1, 0).numpy(), columns=['prob', 'target']).to_csv(f'outputs_dfdc/{execution_id}/val_vector_epoch{epoch:02}.csv', index=False)

            writer.add_scalar(f'Validation/Accuracy/Threshold 0.2', accuracy_02, global_step=epoch)
            writer.add_scalar(f'Validation/Accuracy/Threshold 0.5', accuracy_05, global_step=epoch)
            writer.add_scalar(f'Validation/Accuracy/Threshold 0.7', accuracy_07, global_step=epoch)

        return losses.avg

if __name__ == '__main__':

    main()