import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import BatchSampler
from collections import OrderedDict

import utils as u
from datasets.dataset import ConcatDataset
from utils import SamplingMode


class BetterBatchSampler(BatchSampler):
    """
    Samples differently depending on sampling_mode. Generates batches for training.
    """

    def __init__(self, dataset, args, cur_epoch):
        assert isinstance(dataset, ConcatDataset), "Dataset must be of type ConcatDataset"
        assert args.batch_size % len(args.gpus) == 0
        assert args.batch_size % sum(args.sample_num_instances) == 0
        self.dataset = dataset
        self.args = args
        self.cur_epoch = cur_epoch
        self.eps = 0.00001
        self.target_percentage = 0.5

        super().__init__(self, self.args.batch_size, drop_last=False)
        print("BetterBatchSampler using {} mode.".format(self.args.sampling_mode))

        # Set by resetting the sampler or other methods
        self.__sample_init = False
        self.target_logits = None
        self.tar_mc_affiliation_ind = None
        self.tar_mc_affiliation_probs = None
        self.tar_features = None

    def __iter__(self):
        assert self.__sample_init, "Not initialized yet."
        # Setup mapping for source
        source_dataset = self.dataset.get_dataset(u.Domain.SOURCE)
        target_dataset = self.dataset.get_dataset(u.Domain.TARGET)
        source_mapping = defaultdict(list)
        for idx, (path, label) in enumerate(source_dataset.ordered_dataset):
            source_mapping[label].append(u.DatasetInstance(path=path, label=label, domain=u.Domain.SOURCE,
                                                           index=idx, class_name=source_dataset.get_class_name(label),
                                                           rel_features=np.zeros((self.args.top_k, self.args.feature_size), dtype=np.float32),
                                                           feat_weights=np.zeros((self.args.top_k,), dtype=np.float32)))
        source_keys = list(source_mapping.keys())

        if self.args.sampling_mode in [SamplingMode.TARGET_ONLY]:
            target_softmax_scores = u.stable_softmax(self.target_logits)
            # Construct label --> instances mapping
            target_mapping = defaultdict(list)
            for idx, (data_tuple, sm_score, sm_index) in enumerate(zip(target_dataset.ordered_dataset,
                                                                       np.max(target_softmax_scores, axis=1).squeeze(),
                                                                       np.argmax(self.target_logits, axis=1).squeeze())):
                path, label = data_tuple
                assert label == sm_index
                assert 0. <= sm_score <= 1., sm_score
                target_mapping[sm_index].append(u.DatasetInstance(path=path, label=label, domain=u.Domain.TARGET,
                                                                  sm_score=sm_score, index=idx, class_name=target_dataset.get_class_name(label),
                                                                  rel_features=np.zeros((self.args.top_k, self.args.feature_size), dtype=np.float32),
                                                                  feat_weights=np.zeros((self.args.top_k,), dtype=np.float32)))
            # Sampling loop
            for iteration in range(self.args.sample_num_mini_batches):
                cur_batch = []
                # Sample target instances, oversample in case key is missing
                rand_keys = self.sample_avoid_dupes(source_keys, self.args.batch_size)
                for rand_key in rand_keys:
                    if rand_key in target_mapping:
                        cur_batch.extend(self.sample_avoid_dupes(target_mapping[rand_key], sum(self.args.sample_num_instances)))
                        if len(cur_batch) >= self.args.batch_size:
                            break
                yield self.construct_batch([], cur_batch)
        elif self.args.sampling_mode in [SamplingMode.BASIC_SAMPLING]:
            target_softmax_scores = u.stable_softmax(self.target_logits)
            # Construct label --> instances mapping
            target_mapping = defaultdict(list)
            for idx, (data_tuple, sm_score, sm_index) in enumerate(zip(target_dataset.ordered_dataset,
                                                                       np.max(target_softmax_scores, axis=1).squeeze(),
                                                                       np.argmax(self.target_logits, axis=1).squeeze())):
                path, label = data_tuple
                assert label == sm_index, (label, sm_index, target_softmax_scores[idx])
                assert 0. <= sm_score <= 1., sm_score
                target_mapping[sm_index].append(u.DatasetInstance(path=path, label=label, domain=u.Domain.TARGET,
                                                                  sm_score=sm_score, index=idx, class_name=target_dataset.get_class_name(label),
                                                                  rel_features=np.zeros((self.args.top_k, self.args.feature_size), dtype=np.float32),
                                                                  feat_weights=np.zeros((self.args.top_k,), dtype=np.float32)))
            # Sort within class array
            for k in target_mapping.keys():
                target_mapping[k] = list(sorted(target_mapping[k], key=lambda x: x.sm_score, reverse=True))

            # Sampling loop
            for iteration in range(self.args.sample_num_mini_batches):
                cur_batch_source_indices = []
                cur_batch_target_indices = []

                # Sample target instances
                rand_keys = self.sample_avoid_dupes(source_keys, int((self.args.batch_size * self.target_percentage) // sum(self.args.sample_num_instances)))
                for rand_key in rand_keys:
                    if rand_key not in target_mapping or len(target_mapping[rand_key]) < len(self.args.sample_num_instances):
                        # Put source data in the place of target data
                        cur_batch_target_indices.extend(self.sample_avoid_dupes(source_mapping[rand_key], sum(self.args.sample_num_instances)))
                    else:
                        num_bins = len(self.args.sample_num_instances)
                        num_samples = len(target_mapping[rand_key])
                        bounds = [(int(x / num_bins * num_samples), int((x + 1) / num_bins * num_samples)) for x in range(num_bins)]
                        for bin_idx, (lower, upper) in enumerate(bounds):
                            cur_batch_target_indices.extend(self.sample_avoid_dupes(target_mapping[rand_key][lower:upper],
                                                                                    self.args.sample_num_instances[bin_idx]))

                # Sample source instances based on what targets are already in the batch
                for key in rand_keys:
                    cur_batch_source_indices.extend(self.sample_avoid_dupes(source_mapping[key],
                                                                            int((1. - self.target_percentage) * self.args.batch_size // len(rand_keys))))
                yield self.construct_batch(cur_batch_source_indices, cur_batch_target_indices)
        elif self.args.sampling_mode in [SamplingMode.UNCERTAINTY]:
            # Ignore small probs
            self.tar_mc_affiliation_probs[self.tar_mc_affiliation_probs <= self.args.prob_lower_bound] = self.eps
            # Flag bad entries
            bad_indices = (np.sum(self.tar_mc_affiliation_probs[:, 0:self.args.num_classes // 4], axis=1) <= 0.5).squeeze()
            if not self.args.use_filtering:
                bad_indices = np.zeros_like(bad_indices).astype(np.bool)
            # Renorm target topk probs to sum to 1 again
            self.tar_mc_affiliation_probs = self.tar_mc_affiliation_probs / self.tar_mc_affiliation_probs.sum(axis=1, keepdims=True)

            # Main sampling loop
            tar_class_to_instances = None
            for iteration in range(self.args.sample_num_mini_batches):
                # Regenerate mapping every X iterations
                if iteration % self.args.regen_every == 0:
                    tar_class_to_instances = self.regenerate_structure(bad_indices)
                    if iteration % self.args.sample_num_mini_batches // 2 == 0:
                        print(f"[Sampler] Regenerated structure with {sum(list(map(len, tar_class_to_instances.values())))} items, "
                              f"bad indices: {sum(bad_indices)}.")

                # Construct batch using class affiliation
                cur_batch_source_indices = []
                cur_batch_target_indices = []

                # Sample target instances
                rand_keys = self.sample_avoid_dupes(source_keys, int((self.args.batch_size * self.target_percentage) // sum(self.args.sample_num_instances)))
                for rand_key in rand_keys:
                    if rand_key not in tar_class_to_instances or len(tar_class_to_instances[rand_key]) < len(self.args.sample_num_instances):
                        # Put source data in the place of target data
                        cur_batch_target_indices.extend(self.sample_avoid_dupes(source_mapping[rand_key], sum(self.args.sample_num_instances)))
                    else:
                        num_bins = len(self.args.sample_num_instances)
                        num_samples = len(tar_class_to_instances[rand_key])
                        bounds = [(int(x / num_bins * num_samples), int((x + 1) / num_bins * num_samples)) for x in range(num_bins)]
                        for bin_idx, (lower, upper) in enumerate(bounds):
                            cur_batch_target_indices.extend(self.sample_avoid_dupes(tar_class_to_instances[rand_key][lower:upper],
                                                                                    self.args.sample_num_instances[bin_idx]))

                # Sample source instances based on what targets are already in the batch
                for key in rand_keys:
                    cur_batch_source_indices.extend(self.sample_avoid_dupes(source_mapping[key],
                                                                            int((1. - self.target_percentage) * self.args.batch_size // len(rand_keys))))
                yield self.construct_batch(cur_batch_source_indices, cur_batch_target_indices)

    def construct_batch(self, bs, bt):
        if self.args.batch_construction == u.BatchMode.SMART_BATCH_LAYOUT:
            return self.apply_smart_batch_layout(bs, bt)
        elif self.args.batch_construction == u.BatchMode.RANDOM_BATCH_LAYOUT:
            random.shuffle(bs)
            random.shuffle(bt)
            return self.apply_smart_batch_layout(bs, bt)
        elif self.args.batch_construction == u.BatchMode.SOURCE_FIRST:
            concat = bs + bt
            return list(sorted(concat, key=lambda x: 0 if x.domain == u.Domain.SOURCE else 1))
        elif self.args.batch_construction == u.BatchMode.TARGET_FIRST:
            concat = bt + bs
            return list(sorted(concat, key=lambda x: 0 if x.domain == u.Domain.TARGET else 1))
        elif self.args.batch_construction == u.BatchMode.RANDOM:
            concat = bt + bs
            random.shuffle(concat)
            return concat

    def apply_smart_batch_layout(self, cbsi, cbti):
        # Split equally over GPUs
        cbsi_len, cbti_len = len(cbsi), len(cbti)
        current_batch_indices = []
        factor = 1. / len(self.args.gpus)
        for i in range(len(self.args.gpus)):
            start_s = int(i * factor * cbsi_len)
            end_s = int((i + 1) * factor * cbsi_len)
            start_t = int(i * factor * cbti_len)
            end_t = int((i + 1) * factor * cbti_len)
            current_batch_indices.extend(cbsi[start_s:end_s])
            current_batch_indices.extend(cbti[start_t:end_t])
        return current_batch_indices

    def regenerate_structure(self, bad_bool_flags):
        target_dataset = self.dataset.get_dataset(u.Domain.TARGET)
        certainty_max_preds = np.argmax(self.target_logits, axis=1).squeeze()
        bad_indices = np.nonzero(bad_bool_flags)[0].squeeze()

        feature_means = {}
        if self.args.use_uncertain_features:
            # Generate class assignments for current regen cycle
            current_cycle_max_preds, current_cycle_sampled_probs = self.matrix_random_choice(self.tar_mc_affiliation_ind, self.tar_mc_affiliation_probs)

            # Create new feature means with uncertain predictions
            for c in set(current_cycle_max_preds):
                indices = (current_cycle_max_preds == c) & ~bad_bool_flags
                if np.any(indices):
                    feature_means[c] = self.tar_features[indices].mean(axis=0).squeeze()

        # Check normal max preds for feature if not available in uncertain prediction
        for c in range(self.args.num_classes):
            if c not in feature_means:
                indices = (certainty_max_preds == c)
                if indices.sum() == 0.:
                    # This should never happen, just making sure the feature is still set to something sensible
                    feature_means[c] = self.tar_features.mean(axis=0).squeeze()
                else:
                    feature_means[c] = self.tar_features[indices].mean(axis=0).squeeze()

        # Create target data structure
        tar_class_to_instances = defaultdict(list)
        for dataset_idx, (data_tuple, affiliated_ind, affiliated_probs, certainty_logits) in enumerate(zip(target_dataset.ordered_dataset,
                                                                                                           self.tar_mc_affiliation_ind,
                                                                                                           self.tar_mc_affiliation_probs,
                                                                                                           self.target_logits)):
            path, _ = data_tuple
            # Skip samples
            if dataset_idx in bad_indices:
                continue

            max_pred = int(np.argmax(certainty_logits))
            sample_prob = float(u.stable_softmax(certainty_logits.reshape(1, -1)).squeeze()[max_pred])
            assert 0 <= max_pred <= self.args.num_classes and 0 <= sample_prob <= 1.0, (max_pred, sample_prob)
            related_features = np.concatenate([feature_means[x].reshape(1, -1) for x in affiliated_ind], axis=0)
            tar_class_to_instances[max_pred].append(u.DatasetInstance(path=path,
                                                                      label=int(max_pred),
                                                                      sm_score=sample_prob,
                                                                      domain=u.Domain.TARGET,
                                                                      index=dataset_idx,
                                                                      class_name=target_dataset.get_class_name(int(max_pred)),
                                                                      rel_features=related_features,
                                                                      feat_weights=affiliated_probs))
        # Sort within class array
        for k in tar_class_to_instances.keys():
            tar_class_to_instances[k] = list(sorted(tar_class_to_instances[k], key=lambda x: x.sm_score, reverse=True))
        return tar_class_to_instances

    def matrix_random_choice(self, mat, weight):
        # Sample randomly from every row given the weight
        output_ind = np.zeros(shape=(mat.shape[0],))
        output_prob = np.zeros(shape=(mat.shape[0],))
        for idx, (ind_row, w_row) in enumerate(zip(mat, weight)):
            chosen_idx, chosen_w = random.choices(list(tuple(x) for x in zip(ind_row, w_row)), weights=w_row, k=1)[0]
            output_ind[idx] = chosen_idx
            output_prob[idx] = chosen_w
        return output_ind, output_prob

    def inform(self, certainty_buffer, feats, ind, probs):
        self.target_logits = certainty_buffer
        self.tar_mc_affiliation_ind = ind
        self.tar_mc_affiliation_probs = probs
        self.tar_features = feats
        self.__sample_init = True

    def sample_avoid_dupes(self, array, num):
        if num == 0:
            return []
        elif len(array) >= num:
            return np.random.choice(array, size=num, replace=False).tolist()
        else:
            buffer = []
            while len(buffer) != num:
                buffer.extend(np.random.choice(array, size=min([len(array), num - len(buffer)]), replace=False).tolist())
            assert len(buffer) == num
            return buffer

    def __len__(self):
        return self.args.sample_num_mini_batches
