import random
from collections import OrderedDict

import numpy as np
import termcolor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils as u
from models import resnet


class UtilBase:
    def get_parameters(self, args):
        params = list(y for x, y in self.named_parameters() if x.split('.')[1] not in args.ignore_params) if args.ignore_params else self.parameters()
        ignored_params = list(x for x, y in self.named_parameters() if x.split('.')[1] in args.ignore_params)
        for name, tensor in self.named_parameters():
            if name in ignored_params:
                tensor.requires_grad = False
            else:
                tensor.requires_grad = True
        print("Ignoring parameters: ", ignored_params)
        return params


class Resnet50(nn.Module, UtilBase):
    def __init__(self, args):
        super(Resnet50, self).__init__()
        self.args = args
        self.resnet = resnet.resnet50(pretrained=args.pretrained)
        self.embeddings_size = self.resnet.embedding_size
        self.args.feature_size = self.embeddings_size
        self.classifier = nn.Sequential(
            OrderedDict([
                ('classifier_fc', nn.Linear(self.embeddings_size, args.num_classes)),
            ])
        )

    def forward(self, img, phase):
        mc_logits = []
        _, pre_drop_features = self.resnet(img)
        if phase == u.Phase(u.Phase.IMPROVEMENT_TRAIN):
            post_drop_features = F.dropout(pre_drop_features, p=self.args.imp_dropout, training=True)
        elif phase == u.Phase(u.Phase.MC_DROPOUT_TEST):
            for i in range(self.args.mc_iterations):
                post_drop_features = F.dropout(pre_drop_features, p=self.args.mc_dropout_rate, training=True)
                mc_logits.append(self.classifier(post_drop_features.detach()).detach())
        else:
            post_drop_features = pre_drop_features
        logits = self.classifier(post_drop_features)
        return {'logits': logits, 'pre_drop_features': pre_drop_features, 'post_drop_features': post_drop_features, 'mc_logits': mc_logits}


class Resnet101(nn.Module, UtilBase):
    def __init__(self, args):
        super(Resnet101, self).__init__()
        self.args = args
        self.resnet = resnet.resnet101(pretrained=args.pretrained)
        self.embeddings_size = self.resnet.embedding_size
        self.args.feature_size = self.embeddings_size
        self.classifier = nn.Sequential(
            OrderedDict([
                ('classifier_fc', nn.Linear(self.embeddings_size, args.num_classes)),
            ])
        )

    def forward(self, img, phase):
        mc_logits = []
        _, pre_drop_features = self.resnet(img)
        if phase == u.Phase(u.Phase.IMPROVEMENT_TRAIN):
            post_drop_features = F.dropout(pre_drop_features, p=self.args.imp_dropout, training=True)
        elif phase == u.Phase(u.Phase.MC_DROPOUT_TEST):
            for i in range(self.args.mc_iterations):
                post_drop_features = F.dropout(pre_drop_features, p=self.args.mc_dropout_rate, training=True)
                mc_logits.append(self.classifier(post_drop_features.detach()).detach())
        else:
            post_drop_features = pre_drop_features
        logits = self.classifier(post_drop_features)
        return {'logits': logits, 'pre_drop_features': pre_drop_features, 'post_drop_features': post_drop_features, 'mc_logits': mc_logits}


def get_model(args):
    # Look for all defined subclasses of nn.Module
    defined_models = {v.__name__: v for k, v in globals().items() if type(v) == type(type) and issubclass(v, nn.Module)}
    chosen_model = defined_models[args.architecture](args)
    chosen_model = nn.DataParallel(chosen_model, device_ids=args.gpus).cuda()
    # Calculate parameter count
    num_params = 0
    for p in chosen_model.parameters():
        num_params += p.numel()
    print(f"Loaded model {chosen_model.module.__class__.__name__} with {num_params:,} parameters")

    # Restore snapshot state
    additional_data = None
    if args.snapshot:
        print(f"Loading snapshot: {args.snapshot}")
        snapshot = torch.load(args.snapshot)
        state_dict = snapshot['model']
        try:
            chosen_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(termcolor.colored(e, 'red'))
            chosen_model.load_state_dict(state_dict, strict=False)

        args.start_epoch = args.start_epoch or snapshot['data']['epoch'] + 1
        # Reinitialize random states
        if 'random_states' in snapshot['data']:
            random_states = snapshot['data']['random_states']
            random.setstate(random_states['python'])
            torch.set_rng_state(random_states['pytorch'])
            np.random.set_state(random_states['numpy'])

        # Check seed
        if 'seed' in snapshot and snapshot['seed'] != args.seed:
            raise AttributeError(f'Different seeds found in snapshot ({snapshot["seed"]}) and args ({args.seed})!')

        # Get extra context
        additional_data = snapshot['data']['additional_data']

    # Setup optimizer
    parameters = chosen_model.module.get_parameters(args)
    optimizer = torch.optim.SGD(parameters, lr=args.base_lr, weight_decay=args.weight_decay, nesterov=True, momentum=args.momentum)
    if args.snapshot:
        try:
            optimizer.load_state_dict(snapshot['optimizer'])
        except Exception as e:
            termcolor.colored(e, 'red')

    # Static LR
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)

    if args.snapshot and not args.just_evaluate:
        scheduler.load_state_dict(snapshot['scheduler'])

    return chosen_model, optimizer, scheduler, additional_data


def get_available():
    return [v.__name__ for k, v in globals().items() if type(v) == type(type) and issubclass(v, nn.Module)]


def get_model_by_name(name):
    return {v.__name__.lower(): v for k, v in globals().items() if type(v) == type(type) and issubclass(v, nn.Module)}[name.lower()]
