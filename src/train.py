import argparse
import datetime
import json
import os
import os.path as osp
import random
import shutil
import warnings
from collections import defaultdict
from functools import partial
from pprint import pprint
from typing import Optional

import numpy as np
import termcolor
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import architectures
import globals as g
import utils as u
from datasets import dataset
from datasets import providers
from sampler import BetterBatchSampler
from utils import Phase, SamplingMode, Subset, Domain, BatchMode

writer: Optional[SummaryWriter] = None


def run_model(args, epoch, loader, model, optimizer, phase: Phase = None, need_buffers=False):
    if phase.is_train():
        model.train()
    else:
        model.eval()
    num_batches = len(loader)
    if num_batches == 0:
        return None, None
    color = phase.get_color()
    print(termcolor.colored("Running model, Phase {} ...".format(phase.to_text()), color=color), end='')

    did_backprop = False
    current_epoch_loss = 0
    current_epoch_ce_loss = 0
    current_epoch_num_target = 0
    current_epoch_num_correct = 0
    current_epoch_num_total = 0
    features_buffer = torch.empty(0)
    logits_buffer = torch.empty(0)
    target_buffer = torch.empty(0, dtype=torch.long)
    mc_buffers = [torch.empty(0) for _ in range(args.mc_iterations)]
    paths_buffer = np.empty(0)
    per_class_acc = defaultdict(partial(defaultdict, int))
    confusion_matrix = np.zeros(shape=(args.num_classes, args.num_classes), dtype=np.float32)
    optimizer.zero_grad()

    with torch.set_grad_enabled(phase.is_train()):
        for batch_num, (img_, target_, image_path_, domain_, rel_feats_, rel_weights_) in enumerate(loader):
            loss, _ce_loss, _feature_loss = [torch.tensor([0]).float().cuda(non_blocking=True) for _ in range(3)]
            target = target_.requires_grad_(False).cuda(non_blocking=True)
            domain = torch.tensor([Domain.SOURCE_AS_INT if x == Domain.SOURCE else Domain.TARGET_AS_INT for x in domain_]).requires_grad_(False).cuda(
                non_blocking=True)
            img = img_.requires_grad_(False).cuda()
            outputs = model(img, phase)

            # Buffers
            if need_buffers:
                features_buffer = torch.cat([features_buffer, outputs['pre_drop_features'].detach().cpu()], dim=0)
                logits_buffer = torch.cat([logits_buffer, outputs['logits'].detach().cpu()], dim=0)
                target_buffer = torch.cat([target_buffer, target.detach().long().cpu()], dim=0)
                paths_buffer = np.concatenate([paths_buffer, image_path_], axis=0)
                if phase == Phase(Phase.MC_DROPOUT_TEST):
                    for mc_idx in range(args.mc_iterations):
                        mc_buffers[mc_idx] = torch.cat([mc_buffers[mc_idx], outputs['mc_logits'][mc_idx].cpu().detach()], dim=0)

            # Xent loss
            if args.use_lsm_for:
                soft_targets = torch.zeros((target_.size(0), outputs['logits'].size(1))).fill_(
                    args.label_smoothing_eps / (outputs['logits'].size(1) - 1)).cuda()
                soft_targets.scatter_(1, target.view(-1, 1), 1. - args.label_smoothing_eps)
                hard_targets = torch.zeros((target_.size(0), outputs['logits'].size(1))).cuda()
                hard_targets.scatter_(1, target.view(-1, 1), 1.)

                if Domain.SOURCE in args.use_lsm_for:
                    # Copy soft targets where element is source
                    indices = torch.nonzero(domain == Domain.SOURCE_AS_INT)
                    hard_targets[indices] = soft_targets[indices]
                if Domain.TARGET in args.use_lsm_for:
                    # Copy soft targets where element is target
                    indices = torch.nonzero(domain == Domain.TARGET_AS_INT)
                    hard_targets[indices] = soft_targets[indices]
                _ce_loss = _ce_loss + u.crossentropy_loss(outputs['logits'], hard_targets, reduction='mean')
            else:
                _ce_loss = _ce_loss + F.cross_entropy(outputs['logits'], target)

            if args.use_feature_loss and phase == Phase(Phase.IMPROVEMENT_TRAIN):
                # Feature metric loss, shape: BxTOPKx2048
                _feats = outputs['pre_drop_features']
                target_indices = torch.nonzero(domain == Domain.TARGET_AS_INT).squeeze()
                tar_cnn_feats = _feats[target_indices].squeeze()
                related_features = rel_feats_[target_indices].cuda(non_blocking=True).requires_grad_(False)
                related_weights = rel_weights_[target_indices].cuda(non_blocking=True).requires_grad_(False)

                # Make target features closer to mean feats
                ex_t_feats = tar_cnn_feats.unsqueeze(1).expand_as(related_features)
                tar_to_mean_feat_dist = (ex_t_feats - related_features).pow(2).sum(dim=2).view(-1, args.top_k)
                _feature_loss = _feature_loss + u.crossentropy_loss(F.softmax(-tar_to_mean_feat_dist, dim=1),
                                                                    related_weights,
                                                                    reduction='mean')

            # Finalize loss
            current_epoch_ce_loss += _ce_loss.item()
            loss += _ce_loss + _feature_loss

            # Update stats
            current_epoch_num_total += img_.size(0)
            current_epoch_num_correct += torch.sum((target == torch.argmax(outputs['logits'], dim=1)).float()).item()
            current_epoch_num_target += domain_.count(Domain.TARGET)
            current_epoch_loss += loss.item()

            # Print stats
            batch_acc = torch.mean((target == torch.argmax(outputs['logits'], dim=1)).float()).item()
            print(termcolor.colored(f"\rEp. {epoch:0>3} ({phase.to_text()}: {batch_num:0>4}/{num_batches - 1:0>4}): "
                                    f"ΣLoss: {loss.item():.2f}, "
                                    f"F-Loss: {_feature_loss.item():.2f}, "
                                    f"Acc.: {batch_acc * 100:.2f}%, "
                                    f"BS: {img.size(0)}, "
                                    f"Tar%: {domain_.count(Domain.TARGET) / img.size(0) * 100:.2f}, "
                                    f"LR: {u.get_learning_rates(optimizer)}, k: {args.top_k}, MC: {args.mc_dropout_rate:.2f}",
                                    color=color), end='')

            # Calculate per class accuracy
            corrects = target[target == torch.argmax(outputs['logits'], dim=1)].cpu().detach().numpy()
            for c in corrects:
                per_class_acc[c]['correct'] += 1
                per_class_acc[c]['total'] += 1

            incorrect_indices = target != torch.argmax(outputs['logits'], dim=1)
            incorrects = target[incorrect_indices].cpu().detach().numpy()
            for c in incorrects:
                per_class_acc[c]['total'] += 1

            # Confusion matrix
            for gt, pred in zip(target_, outputs['logits'].argmax(dim=1).cpu().detach().numpy()):
                confusion_matrix[gt, pred] += 1

            # Do backprop while training, ensure that training is allowed on dataset, especially when using the test data!
            if phase.is_train():
                assert loader.dataset.trainable is True
                loss.backward(), optimizer.step(), optimizer.zero_grad()
                did_backprop = True

        # Normalize confusion matrix
        assert confusion_matrix.sum() == current_epoch_num_total
        norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        __prob_sum = norm_confusion_matrix.sum(axis=1).squeeze()
        if not np.any(np.isnan(__prob_sum)):  # Could be zero, check for NaNs
            assert np.allclose(__prob_sum, np.ones(args.num_classes), rtol=0.02, atol=0.02)

        # Show per class accuracy
        if args.show_per_class_stats and phase.is_test():
            # Prevent modification of diagonal to propagate, create temporary copy
            _norm_confusion_matrix = norm_confusion_matrix.copy()
            # Prevent confusion with the class itself
            np.fill_diagonal(_norm_confusion_matrix, 0)
            # Reset cursor
            print('')
            for class_, class_name in sorted(map(lambda x: (x, loader.dataset.dataset_provider.get_label_for_index(x)), per_class_acc.keys()),
                                             key=lambda x: x[1]):
                most_confused_idx = _norm_confusion_matrix[class_].squeeze().argmax()
                most_confused_prob = _norm_confusion_matrix[class_, most_confused_idx]

                class_acc = per_class_acc[class_]['correct'] / per_class_acc[class_]['total']
                print(termcolor.colored(f'\t {class_name:<20}: {class_acc:.4f} -- '
                                        f'confused for {loader.dataset.dataset_provider.get_label_for_index(most_confused_idx)} ({most_confused_prob:.3f}) '
                                        f'[{per_class_acc[class_]["total"]} samples]',
                                        color=color))

        # Print overall acc, fill with spaces in the end, so old text is overwritten
        overall_acc = current_epoch_num_correct / current_epoch_num_total
        print(termcolor.colored(f'\rOverall {phase.to_text()} acc (epoch {epoch}): '
                                f'{overall_acc * 100:.3f}% (considering {current_epoch_num_total} instances), LR: {u.get_learning_rates(optimizer)} '
                                f'{"[BP enabled]" if did_backprop else ""} {" " * 80}',
                                color=color))
        # Also report mean over class accuracies for VISDA
        if phase.is_test() and 'VISDA' in args.test_dataset.upper():
            visda_acc = norm_confusion_matrix.diagonal().mean()
            print(termcolor.colored(f'Mean over per class mean: {visda_acc * 100:.3f}%', color=color))
            writer.add_scalar('{}/metric/AverageClassAccuracy'.format(phase.get_phase().lower()), visda_acc * 100, epoch)

        # Log to tensorboard
        writer.add_scalar('{}/metric/Accuracy'.format(phase.get_phase().lower()), overall_acc * 100, epoch)
        # Also log the final accuracy as text to prevent tensorboard rounding
        if phase in [Phase(Phase.FINAL_TEST), Phase(Phase.SOURCE_ONLY_TEST)]:
            writer.add_text(f"{phase.get_phase().lower()}/result", str(overall_acc * 100))
            if args.experiment_subdir != "":
                output_file = osp.join(g.TENSORBOARD_RUN_DIR, args.experiment_subdir, 'results', phase.get_phase().lower(),
                                       f'{args.dataset.upper()}-{args.test_dataset.upper()}.txt')
                confusion_matrix_file = osp.join(osp.dirname(output_file), f'confusion_matrix_{args.dataset.upper()}-{args.test_dataset.upper()}')
                os.makedirs(osp.dirname(output_file), exist_ok=True)
                # Save classification accuracy to file
                with open(output_file, 'w+') as results_file:
                    results_file.write(f"{args.dataset.upper()}-{args.test_dataset}: {overall_acc * 100}\n")
                # Save confusion matrix to file
                labelmap = {x: loader.dataset.dataset_provider.get_label_for_index(x) for x in range(args.num_classes)}
                np.savez(confusion_matrix_file, norm_confusion_matrix=norm_confusion_matrix, unnorm_confusion_matrix=confusion_matrix, labelmap=labelmap,
                         task=[str(args.dataset), str(args.test_dataset)])
        writer.add_scalar('{}/loss/Total_Loss'.format(phase.get_phase().lower()), current_epoch_loss, epoch)
        writer.add_scalar('{}/loss/CE_Loss'.format(phase.get_phase().lower()), current_epoch_ce_loss, epoch)
        writer.add_scalar('{}/misc/LR'.format(phase.get_phase().lower()), float([param_group['lr'] for param_group in optimizer.param_groups][0]), epoch)
        writer.add_scalar('{}/misc/Target_Percentage'.format(phase.get_phase().lower()), current_epoch_num_target / current_epoch_num_total * 100, epoch)
        return current_epoch_loss, overall_acc, {'features': features_buffer.cpu().numpy(),
                                                 'logits': logits_buffer.cpu().numpy(),
                                                 'mc_logits': torch.cat([x.unsqueeze(0) for x in mc_buffers], dim=0).cpu().detach(),
                                                 'paths': paths_buffer,
                                                 'overall_loss': current_epoch_loss}


def print_epoch_header(epoch, bg_color='on_white', color='grey'):
    prefix = 'Epoch ' if isinstance(epoch, int) else ''
    num_padding = (110 - len(str(epoch))) // 2
    print(termcolor.colored('{} {}{} {}'.format(' ' * num_padding, prefix, epoch, ' ' * num_padding), color=color, on_color=bg_color, attrs=['bold']))


def main(args):
    # Init random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    snapshot_path = None

    # Datasets and loaders
    train_dataset = dataset.get_dataset(args.dataset, args, Subset(Subset.TRAIN, augmentations=True, domain=Domain.SOURCE))
    args.num_classes = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_dataset = dataset.get_dataset(args.test_dataset or args.dataset, args, Subset(Subset.TEST, domain=Domain.TARGET), immutable=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size or args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Get model, optimizer and LR scheduler
    model, optimizer, scheduler, additional_data = architectures.get_model(args)

    # Start training
    _add_data = None
    for epoch in range(args.start_epoch, args.epochs):
        print_epoch_header(f"Epoch {epoch} ({args.dataset} --> {args.test_dataset}) [{args.log_dir}] [{args.comment or 'No comment.'}]")
        # Train
        run_model(args, epoch, train_loader, model, optimizer, phase=Phase(Phase.TRAIN, train_loader.dataset.dataset_name))
        # Test
        if (epoch + 1) % args.test_every == 0 or epoch == args.epochs - 1:
            run_model(args, epoch, test_loader, model, optimizer, phase=Phase(Phase.TEST, test_loader.dataset.dataset_name))
        # Save model every epoch
        snapshot_path = save_snapshot(locals(), identifier='source-trained', additional_data=_add_data)
        # Take a step for the scheduler, this goes after the optim.step in PyTorch 1.4.0
        scheduler.step()

    # Run source only evaluation
    if args.source_only:
        run_model(args, args.epochs, test_loader, model, optimizer, phase=Phase(Phase.SOURCE_ONLY_TEST, test_loader.dataset.dataset_name))
        return

    # Clean up to prevent usage
    del train_loader, train_dataset, scheduler, optimizer

    ############################################################################################################################################
    # Improvement phase
    ############################################################################################################################################
    optimizer = torch.optim.SGD(model.module.get_parameters(args), lr=args.imp_lr, weight_decay=args.weight_decay, nesterov=True, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.imp_epochs // 3 * 2], gamma=0.1)

    # Keep this out of the loop to not reload big datasets every time
    source_domain_dataset = dataset.get_dataset(args.dataset, args, Subset(Subset.TRAIN, augmentations=True, domain=Domain.SOURCE))
    target_domain_dataset = dataset.get_dataset(args.test_dataset, args, Subset(Subset.TEST, augmentations=True, keep_labels=False, domain=Domain.TARGET))
    concat_dataset = dataset.ConcatDataset(source_domain_dataset, target_domain_dataset)
    for epoch_ctr in range(args.imp_epochs):
        print_epoch_header(
            f"Imp. Epoch {epoch_ctr} ({args.dataset} --> {args.test_dataset}) [{args.log_dir}] [{args.comment or 'No comment.'}]",
            bg_color='on_blue',
            color='white')

        # Generate pseudo-labels
        _, _, pseudo_buffers = run_model(args, epoch_ctr, test_loader, model, optimizer, phase=Phase(Phase.IMPROVEMENT_TEST, test_loader.dataset.dataset_name),
                                         need_buffers=True)
        if args.sampling_mode in [u.SamplingMode.UNCERTAINTY]:
            mc_ind, mc_props = conduct_mc_dropout(args, epoch_ctr, test_loader, model, optimizer, phase=Phase(Phase.MC_DROPOUT_TEST))
        else:
            mc_ind = mc_props = None

        # Update labels of target
        target_domain_dataset.set_labels(np.argmax(pseudo_buffers['logits'], axis=1))

        # Set up batch sampler in loop as args might change
        batch_sampler = BetterBatchSampler(concat_dataset,
                                           args=args,
                                           cur_epoch=epoch_ctr)

        # Inform batch sampler about the new labels and threshold
        batch_sampler.inform(pseudo_buffers['logits'], pseudo_buffers['features'], mc_ind, mc_props)
        shared_domain_loader = DataLoader(concat_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=batch_sampler)

        run_model(args, epoch_ctr, shared_domain_loader, model, optimizer, phase=Phase(Phase.IMPROVEMENT_TRAIN, shared_domain_loader.dataset.dataset_name))
        scheduler.step()
        epoch_ctr += 1

        # Save to snapshot
        snapshot_path = save_snapshot(locals(), epoch=epoch_ctr, identifier='concat-trained')

    # Generate final domain adaptation results on full dataset
    print("Final result: {}".format(args.log_dir))
    args.show_per_class_stats = True
    _, _, buffers = run_model(args, args.imp_epochs, test_loader, model, optimizer, phase=Phase(Phase.FINAL_TEST, test_loader.dataset.dataset_name),
                              need_buffers=args.write_embeddings)

    # Write embeddings to csv file, use last buffer outputs from final test
    if args.write_embeddings:
        assert len(buffers['paths']) == len(test_dataset), len(buffers['paths'])
        write_embeddings(args, buffers, prefix=args.test_dataset)


def conduct_mc_dropout(args, epoch_ctr, loader, model, optimizer, phase):
    _, _, mc_buffers = run_model(args, epoch_ctr, loader, model, optimizer, phase=phase, need_buffers=True)
    props = F.softmax(mc_buffers['mc_logits'], dim=2)
    props = props.mean(dim=0).squeeze()
    res = torch.topk(props, k=args.top_k, dim=1)
    return res.indices.cpu().numpy(), res.values.cpu().numpy()


def write_embeddings(args, buffers, prefix=''):
    output_path = osp.join(args.log_dir, 'embeddings', f'{prefix}{osp.basename(args.log_dir)}.csv')
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    print(f"Writing features to {output_path}")
    with open(output_path, 'w+') as out_file:
        for f, l, p in zip(buffers['features'], buffers['logits'], buffers['paths']):
            out_file.write(f"path,{p},logits,{','.join(map(str, l.tolist()))},features,{','.join(map(str, f.tolist()))}\n")


def save_snapshot(_locals=None, _args=None, model=None, optimizer=None, scheduler=None, epoch=None, identifier="default", additional_data=None):
    model = model or _locals.get('model', None)
    optimizer = optimizer or _locals.get('optimizer', None)
    scheduler = scheduler or _locals.get('scheduler', None)
    epoch = epoch or _locals.get('epoch', None)
    _args = _args or _locals['args']

    snapshot_path = osp.join(_args.log_dir, 'snapshots', f'snapshot_{identifier}.pth')
    os.makedirs(osp.dirname(snapshot_path), exist_ok=True)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'seed': _args.seed,
                'data': {
                    'epoch': epoch,
                    'random_states': {'python': random.getstate(),
                                      'numpy': np.random.get_state(),
                                      'pytorch': torch.get_rng_state()
                                      },
                    'args': _args.__dict__,
                    'additional_data': additional_data,
                }}, snapshot_path)
    return snapshot_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, required=True)
    parser.add_argument('--experiment-subdir', type=str, default="")
    parser.add_argument('--dataset', choices=providers.get_available_providers())
    parser.add_argument('--val-dataset', default=None, choices=providers.get_available_providers())
    parser.add_argument('--test-dataset', choices=providers.get_available_providers())
    parser.add_argument('--architecture', choices=architectures.get_available())

    parser.add_argument('--val-percentage', default=0., type=float, help="When splitting off data from train set.")
    parser.add_argument('--test-every', type=int, default=5)
    parser.add_argument('--show-per-class-stats', default=False, action='store_true', help='Show accuracy for all classes while testing.')

    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--base-lr', type=float, default=0.0005)
    parser.add_argument('--imp-lr', type=float, default=0.00025)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=240)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--start-epoch', type=int, default=0, help="For resuming.")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpus', nargs='+', default=[0, 1, 2, 3], type=int)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--use-lsm-for', default=[Domain.SOURCE], nargs='*', choices=[Domain.SOURCE, Domain.TARGET])
    parser.add_argument('--label-smoothing-eps', type=float, default=0.2)

    parser.add_argument('--use-uncertain-features', default=True, action='store_true')
    parser.add_argument('--no-uncertain-features', action='store_false', dest='use_uncertain_features')
    parser.add_argument('--use-filtering', default=True, action='store_true')
    parser.add_argument('--no-filtering', action='store_false', dest='use_filtering')
    parser.add_argument('--ignore-params', type=str, default=[], nargs='*')
    parser.add_argument('--imp-dropout', type=float, default=0.75)
    parser.add_argument('--imp-epochs', type=int, default=75)
    parser.add_argument('--top-k', type=int, required=True, help="Set this to number of classes.")
    parser.add_argument('--mc-iterations', type=int, default=20)
    parser.add_argument('--mc-temp-scaling', type=float, default=1.)
    parser.add_argument('--regen-every', type=int, default=5)
    parser.add_argument('--feature-size', type=int, default=-1)
    parser.add_argument('--prob-lower-bound', type=float, default=0.05)
    parser.add_argument('--mc-dropout-rate', type=float, default=0.85)
    parser.add_argument('--batch-construction', default=BatchMode.SMART_BATCH_LAYOUT, choices=BatchMode.get_modes())
    parser.add_argument('--use-feature-loss', default=True, action='store_true')
    parser.add_argument('--no-feature-loss', action='store_false', dest='use_feature_loss')
    parser.add_argument('--sample-num-instances', default=[5, 4, 3], nargs='+', type=int)
    parser.add_argument('--sample-num-mini-batches', type=int, default=50)
    parser.add_argument('--sampling-mode', default=SamplingMode.UNCERTAINTY, choices=SamplingMode.get_modes())

    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--source-only', default=False, action='store_true')
    parser.add_argument('--write-embeddings', default=False, action='store_true')
    parser.add_argument('--just-evaluate', default=False, action='store_true')
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()
    args.log_dir = osp.join(g.TENSORBOARD_RUN_DIR, args.experiment_subdir,
                            f"{args.dataset}-{args.test_dataset}_{args.architecture}_{args.comment}_{str(datetime.datetime.now()).replace(' ', '_')}")

    # Warnings
    if torch.__version__ != '1.4.0':
        warnings.warn(f"Code was not tested with this torch version, use 1.4.0 for reproducibility (your version is {torch.__version__}).")
    if len(args.gpus) != 4:
        raise ValueError(f"4 GPUs needed, but only {len(args.gpus)} specified.")

    # Prepare experiment folders
    writer = SummaryWriter(log_dir=args.log_dir, max_queue=1, flush_secs=0.5)
    print("Logdir:", writer.file_writer.get_logdir())

    # Copy code to experiment directory
    for root, dirs, files in os.walk('./src/'):
        if files:
            for file in files:
                if file.endswith('.py') and 'runs' not in root:
                    os.makedirs(osp.join(writer.file_writer.get_logdir(), 'code'), exist_ok=True)
                    out_path = osp.join(writer.file_writer.get_logdir(), 'code', root)
                    os.makedirs(out_path, exist_ok=True)
                    shutil.copy(osp.join(root, file), out_path)

    # Also save args
    with open(osp.join(writer.file_writer.get_logdir(), 'args.json'), 'w+') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        pprint(args.__dict__)
        print(args.log_dir)

    # Also save command line (does not work on Windows, but who cares?)
    with open('/proc/self/cmdline', 'r') as f:
        cmd_line = f.read().replace('\x00', ' ').strip()
    with open(osp.join(writer.file_writer.get_logdir(), 'cmd.sh'), 'w+') as f:
        f.write(cmd_line)

    # Dirty hack to skip training phases
    if args.just_evaluate:
        args.epochs = 0
        args.imp_epochs = 0

    main(args)
    # Make sure the last values are flushed
    writer.close()
    # Print out relevant information once again
    print(f"{'-' * 100}\n{args.log_dir}\n{'-' * 100}\n{args.__dict__}")
