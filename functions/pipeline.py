import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from functions.utils import AverageMeter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from apex import amp


def train_iteration(model, dataset, criterion, optimizer, scheduler, epoch, epoch_size, batch_size, sampler, writer, local_rank, n_workers, amp_):

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


def validate(model, val_ds_name, dataset, epoch, criterion, batch_size, writer, execution_id):

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=None, pin_memory=True)

    losses = AverageMeter()
    f_losses = AverageMeter()
    r_losses = AverageMeter()

    pairs_prob_and_target = []

    pbar = tqdm(loader)

    for idx, data in enumerate(pbar):

        # Get the inputs and labels
        inputs, labels = data["image"], data["labels"]
        inputs, labels = inputs.cuda(non_blocking=True), labels.float().cuda(non_blocking=True)

        with torch.no_grad():
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

        pbar.set_description(f'Epoch {epoch} val. {val_ds_name} Loss:{losses.avg:.4f}-Fake.Loss:{f_losses.avg:.4f}-Real.Loss:{r_losses.avg:.4f}')

    writer.add_scalar(f'Validation-{val_ds_name}/Total Loss', losses.avg, global_step=epoch)
    writer.add_scalar(f'Validation-{val_ds_name}/Fake Loss - val', f_losses.avg, global_step=epoch)
    writer.add_scalar(f'Validation-{val_ds_name}/Real Loss - val', r_losses.avg, global_step=epoch)

    ppt = torch.cat(pairs_prob_and_target, dim=1)

    accuracy_02 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.2).float().numpy())
    accuracy_05 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.5).float().numpy())
    accuracy_07 = accuracy_score(ppt[1, :].numpy(), (ppt[0, :] > 0.7).float().numpy())

    os.makedirs(f'outputs/{execution_id}/{val_ds_name}', exist_ok=True)

    pd.DataFrame(ppt.transpose(1, 0).numpy(), columns=['prob', 'target']).to_csv(f'outputs/{execution_id}/{val_ds_name}/val_vector_epoch{epoch:02}.csv', index=False)

    writer.add_scalar(f'Validation-{val_ds_name}/Accuracy/Threshold 0.2', accuracy_02, global_step=epoch)
    writer.add_scalar(f'Validation-{val_ds_name}/Accuracy/Threshold 0.5', accuracy_05, global_step=epoch)
    writer.add_scalar(f'Validation-{val_ds_name}/Accuracy/Threshold 0.7', accuracy_07, global_step=epoch)

    return losses.avg