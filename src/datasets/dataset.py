from utils import Subset
from utils import Phase
from torch.utils.data import Dataset
from datasets import providers
from PIL import Image
import numpy as np
import termcolor
import cv2
import utils as u
import torch


class GenericDataset(Dataset):
    def __init__(self, dataset_name, args, subset: Subset, immutable):
        self.args = args
        self.subset = subset
        self.domain = subset.domain
        self.dataset_name = dataset_name
        # Assure that test data is not tampered with (no resetting/no label setting)
        self.immutable = immutable
        self.dataset_provider = providers.get_provider(dataset_name)(args, subset)
        print("Loaded dataset {} ({}) with {} classes and {} instances.".format(dataset_name, subset.to_text(),
                                                                                self.num_classes,
                                                                                len(self.dataset_provider)))

    @property
    def num_classes(self):
        return self.dataset_provider.num_classes

    @property
    def ordered_labels(self):
        # Check if we are allowed to get the labels
        assert self.trainable
        return self.dataset_provider.ordered_labels

    @property
    def ordered_dataset(self):
        # Check if we are allowed to get the labels
        assert self.trainable
        return self.dataset_provider.ordered_dataset

    @property
    def trainable(self):
        return self.dataset_provider.trainable

    def get_class_name(self, index: int):
        return self.dataset_provider.get_label_for_index(index)

    def set_labels(self, new_labels, paths=None):
        """
        :param new_labels: np.ndarray, set label to -1 to get it removed from the dataset.
        :return: None
        """
        assert not self.immutable
        self.reset_labels()
        print(termcolor.colored('Set {} new labels for {} ({})!'.format(np.count_nonzero(new_labels != -1), self.dataset_name, str(self.subset)), 'red'))
        self.dataset_provider.set_labels(np.array(new_labels), paths=paths)

    def reset_labels(self):
        assert not self.immutable
        print(termcolor.colored('Reset labels for {} ({})!'.format(self.dataset_name, str(self.subset)), 'red'))
        self.dataset_provider.reset_labels()

    def __getitem__(self, item):
        img, label, image_path = self.dataset_provider.load(item)
        try:
            img = self.dataset_provider.transform(img)
        except Exception as e:
            print(f"\n\n{image_path}\n\n")
            raise e

        if self.args.debug:
            cv2.imshow("debug", cv2.cvtColor(
                np.array(255 * (img.numpy().transpose(1, 2, 0) * 0.2 + 0.5), dtype=np.uint8),
                cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)
        return img, label, image_path, self.domain, float('nan'), float('nan')

    def __len__(self):
        return len(self.dataset_provider)


class ConcatDataset(Dataset):
    def __init__(self, dataset_source, dataset_target):
        assert dataset_source.subset == Subset.TRAIN and dataset_target.subset == Subset.TEST and dataset_target.domain.lower() == 'target'

        self.dataset_source = dataset_source
        self.dataset_target = dataset_target
        self.args = self.dataset_source.args
        self.dataset_name = f"<Concatenation of {self.dataset_source.dataset_name} and {self.dataset_target.dataset_name}>"
        print(f"Loaded dataset {self.dataset_name} with {self.num_classes} classes and {len(self)} instances.")

    def set_labels(self, new_labels):
        self.dataset_source.dataset_provider.set_labels(new_labels[:len(self.dataset_source)])
        self.dataset_target.dataset_provider.set_labels(new_labels[len(self.dataset_source):])

    @property
    def trainable(self):
        return self.dataset_source.dataset_provider.trainable and self.dataset_target.dataset_provider.trainable

    @property
    def num_classes(self):
        return self.dataset_source.num_classes

    def get_dataset(self, domain):
        assert domain in [u.Domain.SOURCE, u.Domain.TARGET]
        if domain == u.Domain.SOURCE:
            return self.dataset_source
        elif domain == u.Domain.TARGET:
            return self.dataset_target
        else:
            raise AttributeError

    @property
    def ordered_labels(self):
        return np.concatenate([self.dataset_source.ordered_labels, self.dataset_target.ordered_labels], axis=0)

    def __getitem__(self, item):
        assert self.dataset_target.trainable
        domain = None

        # Allow to index by providing a dict with all the information. This is a really dirty hack, but PyTorch allows it ...
        if isinstance(item, u.DatasetInstance):
            # Conversion necessary as apparently some images are gray scale
            # Note that this bypasses the dataset_provider.load function
            img = Image.open(item.path).convert('RGB')
            img = self.dataset_source.dataset_provider.transform(img) if item.domain == u.Domain.SOURCE else self.dataset_target.dataset_provider.transform(img)
            return img, item.label, item.path, item.domain, torch.from_numpy(item.related_features), torch.from_numpy(item.feature_weights)

        # Item must be int then
        assert isinstance(item, (int, np.int32, np.int64)), type(item)
        if item < len(self.dataset_source):
            img, label, image_path = self.dataset_source.dataset_provider.load(item)
            img = self.dataset_source.dataset_provider.transform(img)
            domain = u.Domain.SOURCE
        else:
            # Index is 0 based for concat dataset
            img, label, image_path = self.dataset_target.dataset_provider.load(item - len(self.dataset_source))
            img = self.dataset_target.dataset_provider.transform(img)
            domain = u.Domain.TARGET

        if self.args.debug:
            cv2.imshow("{}_{}".format(self.dataset_name, label), cv2.cvtColor(
                np.array(255 * (img.numpy().transpose(1, 2, 0) * 0.2 + 0.5), dtype=np.uint8),
                cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return img, label, image_path, domain, float('nan'), float('nan')

    def reset_labels(self):
        self.dataset_source.reset_labels()
        self.dataset_target.reset_labels()

    def __len__(self):
        return len(self.dataset_source) + len(self.dataset_target)


def get_dataset(dataset_name, args, subset: Subset, immutable=False):
    return GenericDataset(dataset_name, args, subset, immutable)
