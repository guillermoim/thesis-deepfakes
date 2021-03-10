import os
import torch
from models import models
import argparse
import numpy as np
from apex import amp
from datasets.builder import create_train_dataset, create_val_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from functions.pipeline import train_iteration, validate

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser(description='Train a model with the parameters specified.')

    parser.add_argument('--seed', type=int, default=101, help='Specify manual seed')
    parser.add_argument('--model', choices=models.get_available_models(),
                        type=str, required=True, help='Specify the type of model to train')
    parser.add_argument('--v', choices=[0, 1, 2], type=int, default=0, help='Specify training variant')
    parser.add_argument('--epochs', type=int, default=30, help='Total number of epochs')
    parser.add_argument('--epoch_size', type=int, default=300, help='Size (in num of batches) of each epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for the DataLoader')
    parser.add_argument('--distributed', action='store_true', help='Specify whether to use Distributed Model or not')
    parser.add_argument('--amp', action='store_true', help='Specify whether to use Nvidia Automatic Mixed Precision or not')
    parser.add_argument('--data_path', type=str, required=True, help='Specify path to data.')
    parser.add_argument('--policy', choices=[f'policy_{i}'for i in range(1, 6)], type=str, default='policy_1',
                        help='Specify training variant')
    parser.add_argument('--dataset', choices=['faceforensics', 'dfdc', 'other'],
                        type=str, default='faceforensics', help='Specify type of base data augmentation transformations.')

    parser.add_argument('--save_path', type=str, required=True, help='Specify location to save model.')

    args = parser.parse_args()

    # Load the arguments
    seed = args.seed
    model_name = args.model
    variant = args.v
    epochs = args.epochs
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    n_workers = args.n_workers
    distributed = args.distributed
    amp_ = args.amp
    data_path = args.data_path
    data_augment = args.policy
    da_dataset = args.dataset
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)

    assert distributed or not amp_, "Mixed precision only allowed in distributed training mode."

    # Create execution_id (which is the name under which info related to a certain execution is stored) and
    # path_to_save
    execution_id = f'{model_name}-{da_dataset}-{data_augment}_v{variant}_s{seed}'
    path_to_save = os.path.join(save_path, f'{execution_id}.pth')
    # Set seed manually for reproducibility
    torch.manual_seed(seed)
    # We will save the model with best validation loss, thus we initiate the best validation loss with very high value
    best_val_loss = 1000

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

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

    # Generate tensorboard writer with metrics report. Only add information in local_rank == 0 in distributed setting.
    writer = SummaryWriter(f'runs/{execution_id}', flush_secs=15)
    if args.local_rank == 0:
        writer.add_text('Description', str(desc), global_step=0)

    # Train split containing both raw, c23 and c40
    train = create_train_dataset(da_dataset, data_path, normalization, resize, data_augment)

    # For validation, different levels of compression go separately.
    val_datasets = create_val_dataset(da_dataset, data_path, normalization, 220)

    for epoch in range(epochs):

        model.train()

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(train)
            sampler.set_epoch(epoch)
        else:
            sampler = None

        train_iteration(model, train, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer,
                        args.local_rank, n_workers, amp_)

        if args.local_rank == 0:

            model.eval()

            losses = []

            for key in val_datasets:
                val_dataset = val_datasets[key]
                loss = validate(model, key, val_dataset, epoch, criterion, batch_size, writer, execution_id)
                losses.append(loss)

            if np.mean(losses) < best_val_loss:
                print('Saving checkpoint at epoch', epoch , '!!')
                to_save = { 'epoch': epoch,
                            'variant': variant,
                            'model_name': model_name,
                            'model_state_dict': model.state_dict(),
                            'best_loss': np.mean(losses),}

                torch.save(to_save, path_to_save)


if __name__ == '__main__':
    main()