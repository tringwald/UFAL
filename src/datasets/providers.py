import os
import os.path as osp
from abc import ABC

import PIL
import numpy as np
import torchvision.transforms as TF
from PIL import Image
import termcolor

import globals as g
from utils import Subset


class AvailableProvider(ABC):
    """ Dummy class to flag all providers than can be chosen. """

    def __init__(self, *args, **kwargs):
        pass


class ProviderInterface(ABC):
    def __init__(self, args, subset, *_, **__):
        self.args = args
        self.subset = subset
        self.transforms = None
        self.test_train_allowed = False

        if not self.subset.keep_labels:
            print(termcolor.colored(f"Not keeping labels for {self.subset} ({self.subset.domain}).", color="red"))

    @property
    def trainable(self):
        if self.subset.is_test() or self.subset.is_val():
            return self.test_train_allowed
        else:
            return True

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def ordered_labels(self):
        raise NotImplementedError

    @property
    def ordered_dataset(self):
        raise NotImplementedError

    def transform(self, img: PIL.Image):
        return self.transforms(img)

    def load(self, index: int):
        raise NotImplementedError

    def reset_labels(self):
        raise NotImplementedError

    def set_labels(self, new_labels: np.ndarray):
        raise NotImplementedError

    def get_label_for_index(self, index: int):
        raise NotImplementedError


########################################################################
class OC(ProviderInterface):
    def __init__(self, args, subset, subtype):
        super().__init__(args, subset)
        if self.subset.augmentations:
            self.transforms = TF.Compose([
                TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
                TF.RandomChoice([
                    TF.RandomGrayscale(p=0.2),
                    TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
                ]),
                TF.RandomCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = TF.Compose([
                TF.Resize((256, 256)),
                TF.CenterCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])
        self.subtype = subtype

        self.images_root = osp.join(g.OFFICE_CALTECH_ROOT, subtype)
        self.unique_classes = list(sorted(list(set(os.listdir(self.images_root)))))
        self.class_to_index = dict(zip(self.unique_classes, range(len(self.unique_classes))))
        if args.debug:
            print(self.class_to_index)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        data = []
        for class_ in self.unique_classes:
            for image in sorted(os.listdir(osp.join(self.images_root, class_))):
                if self.subset.keep_labels:
                    label = self.class_to_index[class_]
                else:
                    label = float('nan')
                data.append([osp.join(self.images_root, class_, image), label])
        self.dataset = make_val(deterministic_shuffle(data), args, subset)

    def __getitem__(self, index):
        return self.dataset[index]

    @property
    def num_classes(self):
        return len(set([x[1] for x in self.dataset]))

    @property
    def ordered_labels(self):
        return np.array([x[1] for x in self.dataset])

    @property
    def ordered_dataset(self):
        return self.dataset

    def load(self, index):
        img_path, target = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        return img, int(target), img_path

    def __len__(self):
        return len(self.dataset)

    def set_labels(self, new_labels, paths=None):
        delete_list = []
        for idx, (entry, new_label) in enumerate(zip(self.dataset, new_labels)):
            if paths is not None:
                assert paths[idx] == entry[0], (paths[idx], entry[0])  # Make sure the right entry is set
            entry[1] = new_label
            if new_label == -1:
                delete_list.append(idx)
        # Remove invalid labels
        for del_idx in reversed(sorted(delete_list)):
            del self.dataset[del_idx]

    def reset_labels(self):
        """ Has to be called before training on a test subset dataset to ensure that a model is not trained on GT annotations. """
        for entry in self.dataset:
            entry[1] = None
        assert self.num_classes == 1
        self.test_train_allowed = True

    def get_label_for_index(self, index: int):
        return self.index_to_class[index]

    def set_dataset(self, new_dataset: np.ndarray):
        raise NotImplementedError


class OCAmazon(OC, AvailableProvider):
    def __init__(self, args, subset):
        super(OCAmazon, self).__init__(args, subset, subtype='amazon')


class OCDSLR(OC, AvailableProvider):
    def __init__(self, args, subset):
        super(OCDSLR, self).__init__(args, subset, subtype='dslr')


class OCWebcam(OC, AvailableProvider):
    def __init__(self, args, subset):
        super(OCWebcam, self).__init__(args, subset, subtype='webcam')


class OCCaltech(OC, AvailableProvider):
    def __init__(self, args, subset):
        super(OCCaltech, self).__init__(args, subset, subtype='caltech')


########################################################################
class VisDA17(ProviderInterface, AvailableProvider):
    def __init__(self, args, subset):
        super(VisDA17, self).__init__(args, subset)
        if self.subset.augmentations:
            self.transforms = TF.Compose([
                TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
                TF.RandomChoice([
                    TF.RandomGrayscale(p=0.2),
                    TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
                ]),
                TF.RandomCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = TF.Compose([
                TF.Resize((256, 256)),
                TF.CenterCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])

        if subset.is_train() or subset.is_val():
            subset_root = osp.join(g.VISDA17_ROOT, 'train')
        elif subset.is_test():
            subset_root = osp.join(g.VISDA17_ROOT, 'validation')
            subset.comment += 'Actually val set.'
        else:
            raise NotImplementedError
        self.label_map = {}

        data = []
        with open(osp.join(subset_root, 'image_list.txt')) as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                label = 0 if len(tmp) == 1 else int(tmp[1])
                self.label_map[tmp[0].split('/')[0]] = label
                if not self.subset.keep_labels:
                    label = float('nan')
                full_path = osp.join(subset_root, tmp[0])
                data.append([full_path, label])
        self.dataset = make_val(deterministic_shuffle(data), args, subset)
        self.index_to_label = {v: k for k, v in self.label_map.items()}

    def load(self, index):
        img_path, target = self.dataset[index]
        # Make RGB
        img = Image.open(img_path).convert("RGB")
        return img, int(target), img_path

    def transform(self, img):
        return self.transforms(img)

    def get_label_for_index(self, index):
        return self.index_to_label[int(index)]

    def set_labels(self, new_labels, paths=None):
        delete_list = []
        for idx, (entry, new_label) in enumerate(zip(self.dataset, new_labels)):
            if paths is not None:
                assert paths[idx] == entry[0], (paths[idx], entry[0])  # Make sure the right entry is set
            entry[1] = new_label
            if new_label == -1:
                delete_list.append(idx)
        # Remove invalid labels
        for del_idx in reversed(sorted(delete_list)):
            del self.dataset[del_idx]

    def reset_labels(self):
        for entry in self.dataset:
            entry[1] = None
        assert self.num_classes == 1, self.ordered_labels
        self.test_train_allowed = True

    @property
    def ordered_labels(self):
        return np.array([x[1] for x in self.dataset])

    @property
    def ordered_dataset(self):
        return self.dataset

    @property
    def num_classes(self):
        return len(set([x[1] for x in self.dataset]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class VisDA17Test(ProviderInterface, AvailableProvider):
    def __init__(self, args, subset):
        super(VisDA17Test, self).__init__(args, subset)
        if self.subset.augmentations:
            self.transforms = TF.Compose([
                TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
                TF.RandomChoice([
                    TF.RandomGrayscale(p=0.2),
                    TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
                ]),
                TF.RandomCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = TF.Compose([
                TF.Resize((256, 256)),
                TF.CenterCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])

        subset_root = osp.join(g.VISDA17_ROOT, 'test')
        self.label_map = {"aeroplane": 0,
                          "bicycle": 1,
                          "bus": 2,
                          "car": 3,
                          "horse": 4,
                          "knife": 5,
                          "motorcycle": 6,
                          "person": 7,
                          "plant": 8,
                          "skateboard": 9,
                          "train": 10,
                          "truck": 11,
                          }

        data = []
        with open(osp.join(subset_root, 'ground_truth.txt')) as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                label = int(tmp[1])
                if not self.subset.keep_labels:
                    label = float('nan')
                full_path = osp.join(subset_root, tmp[0])
                data.append([full_path, label])
        self.dataset = make_val(deterministic_shuffle(data), args, subset)
        self.index_to_label = {v: k for k, v in self.label_map.items()}

    def load(self, index):
        img_path, target = self.dataset[index]
        # Make RGB
        img = Image.open(img_path).convert("RGB")
        return img, int(target), img_path

    def transform(self, img):
        return self.transforms(img)

    def get_label_for_index(self, index):
        return self.index_to_label[int(index)]

    def set_labels(self, new_labels, paths=None):
        delete_list = []
        for idx, (entry, new_label) in enumerate(zip(self.dataset, new_labels)):
            if paths is not None:
                assert paths[idx] == entry[0], (paths[idx], entry[0])  # Make sure the right entry is set
            entry[1] = new_label
            if new_label == -1:
                delete_list.append(idx)
        # Remove invalid labels
        for del_idx in reversed(sorted(delete_list)):
            del self.dataset[del_idx]

    def reset_labels(self):
        for entry in self.dataset:
            entry[1] = None
        assert self.num_classes == 1, self.ordered_labels
        self.test_train_allowed = True

    @property
    def ordered_labels(self):
        return np.array([x[1] for x in self.dataset])

    @property
    def ordered_dataset(self):
        return self.dataset

    @property
    def num_classes(self):
        return len(set([x[1] for x in self.dataset]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


########################################################################
class OfficeHome(ProviderInterface):
    def __init__(self, args, subset, subtype):
        super().__init__(args, subset)
        if self.subset.augmentations:
            self.transforms = TF.Compose([
                TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=Image.BICUBIC),
                TF.RandomHorizontalFlip(p=0.5),
                TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, resample=Image.BICUBIC),
                TF.RandomChoice([
                    TF.RandomGrayscale(p=0.2),
                    TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
                ]),
                TF.RandomCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = TF.Compose([
                TF.Resize((256, 256)),
                TF.CenterCrop((224, 224)),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])
        self.subtype = subtype

        self.images_root = osp.join(g.OFFICE_HOME_ROOT, subtype)
        self.unique_classes = list(sorted(list(set(os.listdir(self.images_root)))))
        self.class_to_index = dict(zip(self.unique_classes, range(len(self.unique_classes))))
        if args.debug:
            print(self.class_to_index)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        data = []
        for class_ in self.unique_classes:
            for image in sorted(os.listdir(osp.join(self.images_root, class_))):
                if self.subset.keep_labels:
                    label = self.class_to_index[class_]
                else:
                    label = float('nan')
                data.append([osp.join(self.images_root, class_, image), label])
        self.dataset = make_val(deterministic_shuffle(data), args, subset)

    def __getitem__(self, index):
        return self.dataset[index]

    @property
    def num_classes(self):
        return len(set([x[1] for x in self.dataset]))

    @property
    def ordered_labels(self):
        return np.array([x[1] for x in self.dataset])

    @property
    def ordered_dataset(self):
        return self.dataset

    def load(self, index):
        img_path, target = self.dataset[index]
        img = Image.open(img_path)
        return img, int(target), img_path

    def __len__(self):
        return len(self.dataset)

    def set_labels(self, new_labels, paths=None):
        delete_list = []
        for idx, (entry, new_label) in enumerate(zip(self.dataset, new_labels)):
            if paths is not None:
                assert paths[idx] == entry[0], (paths[idx], entry[0])  # Make sure the right entry is set
            entry[1] = new_label
            if new_label == -1:
                delete_list.append(idx)
        # Remove invalid labels
        for del_idx in reversed(sorted(delete_list)):
            del self.dataset[del_idx]

    def reset_labels(self):
        """ Has to be called before training on a test subset dataset to ensure that a model is not trained on GT annotations. """
        for entry in self.dataset:
            entry[1] = None
        assert self.num_classes == 1
        self.test_train_allowed = True

    def get_label_for_index(self, index: int):
        return self.index_to_class[index]

    def set_dataset(self, new_dataset: np.ndarray):
        raise NotImplementedError


class OfficeHomeArt(OfficeHome, AvailableProvider):
    def __init__(self, args, subset):
        super(OfficeHomeArt, self).__init__(args, subset, subtype='Art')


class OfficeHomeClipart(OfficeHome, AvailableProvider):
    def __init__(self, args, subset):
        super(OfficeHomeClipart, self).__init__(args, subset, subtype='Clipart')


class OfficeHomeProduct(OfficeHome, AvailableProvider):
    def __init__(self, args, subset):
        super(OfficeHomeProduct, self).__init__(args, subset, subtype='Product')


class OfficeHomeReal(OfficeHome, AvailableProvider):
    def __init__(self, args, subset):
        super(OfficeHomeReal, self).__init__(args, subset, subtype='Real World')


########################################################################
def deterministic_shuffle(x):
    _old_rng_state = np.random.get_state()
    np.random.seed(0)
    np.random.shuffle(x)
    np.random.set_state(_old_rng_state)
    return x


def make_val(data, args, subset):
    if subset.is_train():
        deterministic_shuffle(data)
        return data[:int(len(data) * (1 - args.val_percentage))]
    elif subset.is_val():
        deterministic_shuffle(data)
        return data[int(len(data) * (1 - args.val_percentage)):]
    else:
        return data


def get_available_providers():
    return [v.__name__.upper() for k, v in globals().items()
            if
            type(v) == type(ProviderInterface) and issubclass(v, AvailableProvider) and v not in ABC.__subclasses__()]


def get_provider(name: str):
    return {v.__name__.upper(): v for k, v in globals().items()
            if
            type(v) == type(ProviderInterface) and issubclass(v, AvailableProvider) and v not in ABC.__subclasses__()}[
        name]
