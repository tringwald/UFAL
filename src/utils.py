import numpy as np
import termcolor
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Module


class Domain:
    SOURCE = 'SOURCE'
    TARGET = 'TARGET'

    SOURCE_AS_INT = 0
    TARGET_AS_INT = 1


class Subset:
    TRAIN = 'TRAIN'
    VAL = 'VAL'
    TEST = 'TEST'

    def __init__(self, subset_name: str, augmentations=False, special_augmentations=False, keep_labels=True, domain: Domain = None, warn=True, comment=''):
        assert subset_name.upper() in Phase.__dict__
        self.subset_name = subset_name.upper()
        self.augmentations = augmentations
        self.special_augmentations = special_augmentations
        self.comment = comment
        self.keep_labels = keep_labels
        self.domain = domain
        if self.is_train() and not self.augmentations and warn:
            print(termcolor.colored(f'Train subset without augmentations? {self.__dict__}', 'yellow'))

    def is_train(self):
        return self.subset_name == self.TRAIN

    def is_val(self):
        return self.subset_name == self.VAL

    def is_test(self):
        return self.subset_name == self.TEST

    def to_text(self):
        if self.comment != "":
            return "{} ({})".format(self.subset_name, self.comment)
        else:
            return self.subset_name

    def get_subset_name(self):
        return str(self)

    def __str__(self):
        return self.subset_name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.subset_name == other
        elif isinstance(other, Subset):
            return self.subset_name == other.subset_name
        else:
            raise ValueError()


class Phase:
    TRAIN = 'TRAIN'
    VAL = 'VAL'
    TEST = 'TEST'
    CLUSTER = 'CLUSTER'
    IMPROVEMENT_TRAIN = 'IMPROVEMENT_TRAIN'
    IMPROVEMENT_TEST = 'IMPROVEMENT_TEST'
    SOURCE_ONLY_TEST = 'SOURCE_ONLY_TEST'
    FINAL_TEST = 'FINAL_TEST'
    MC_DROPOUT_TEST = 'MC_DROPOUT_TEST'
    MC_DROPOUT_TEST_OTHER = 'MC_DROPOUT_TEST_OTHER'

    def __init__(self, phase_name: str, dataset_name: str = 'unknown_dataset', comment=''):
        assert phase_name.upper() in Phase.__dict__
        self.phase = phase_name.upper()
        self.comment = comment
        self.dataset_name = dataset_name
        self._phases = {Phase.TRAIN: 0,
                        Phase.VAL: 1,
                        Phase.TEST: 2,
                        Phase.CLUSTER: 3,
                        Phase.IMPROVEMENT_TRAIN: 4,
                        Phase.IMPROVEMENT_TEST: 5,
                        Phase.FINAL_TEST: 6,
                        Phase.MC_DROPOUT_TEST: 7,
                        Phase.SOURCE_ONLY_TEST: 8,
                        Phase.MC_DROPOUT_TEST_OTHER: 9}

    def as_int(self) -> int:
        return self._phases[self.phase]

    def is_train(self):
        return self.phase in [self.TRAIN, self.IMPROVEMENT_TRAIN]

    def is_val(self):
        return self.phase in [self.VAL]

    def is_test(self):
        return self.phase in [self.TEST, self.IMPROVEMENT_TEST, self.FINAL_TEST, self.MC_DROPOUT_TEST, self.SOURCE_ONLY_TEST,
                              self.MC_DROPOUT_TEST_OTHER]

    def is_cluster(self):
        return self.phase == self.CLUSTER

    def to_text(self):
        if self.comment != "":
            return "{} of {} ({})".format(self.phase, self.dataset_name, self.comment)
        else:
            return self.phase

    def get_phase(self):
        return self.phase

    def __eq__(self, other):
        return self.phase == other.phase

    def get_color(self):
        try:
            return {self.TRAIN: None,
                    self.VAL: 'yellow',
                    self.TEST: 'blue',
                    self.IMPROVEMENT_TEST: 'cyan',
                    self.SOURCE_ONLY_TEST: 'magenta',
                    self.FINAL_TEST: 'magenta',
                    self.MC_DROPOUT_TEST: 'green',
                    self.MC_DROPOUT_TEST_OTHER: 'green'}[self.phase]
        except:
            return None


class SamplingMode:
    TARGET_ONLY = 'TARGET_ONLY'
    BASIC_SAMPLING = 'BASIC_SAMPLING'
    UNCERTAINTY = 'UNCERTAINTY'

    @staticmethod
    def get_modes():
        return [x for x in SamplingMode.__dict__ if str(x).isupper()]


class BatchMode:
    SMART_BATCH_LAYOUT = 'SMART_BATCH_LAYOUT'
    RANDOM_BATCH_LAYOUT = 'RANDOM_BATCH_LAYOUT'
    RANDOM = 'RANDOM'
    SOURCE_FIRST = 'SOURCE_FIRST'
    TARGET_FIRST = 'TARGET_FIRST'

    @staticmethod
    def get_modes():
        return [x for x in BatchMode.__dict__ if str(x).isupper()]


class DatasetInstance:
    def __init__(self, path, label, domain, sm_score=None, index=None, class_name=None, rel_features=None, feat_weights=None):
        self.path: str = path
        self.label: int = label
        self.class_name: str = class_name
        assert domain in [Domain.SOURCE, Domain.TARGET]
        self.domain = domain
        self.sm_score: float = sm_score
        self.index: int = index
        self.related_features = rel_features
        self.feature_weights = feat_weights

    def __str__(self):
        return f"<{self.domain}: {self.class_name}, ID {self.index}>"

    def __eq__(self, other):
        if self.domain == other.domain and self.index == other.index:
            return True
        else:
            return False


def get_learning_rates(optimizer):
    return [float(param_group['lr']) for param_group in optimizer.param_groups]


def get_momentum(optimizer):
    return [float(param_group['momentum']) if 'momentum' in param_group else None for param_group in
            optimizer.param_groups]


def stable_softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def crossentropy_loss(logits, target, reduction='mean'):
    """ TF-like crossentropy implementation, as PyTorch only accepts the class index and not a one hot encoded label.
    :param logits: NxC matrix of logits.
    :param target: NxC matrix of probability distribution.
    :return: Average loss over all batch elements.
    """
    loss = torch.sum(- target * F.log_softmax(logits, -1), -1)
    if reduction == 'mean':
        loss = loss.mean()
    return loss
