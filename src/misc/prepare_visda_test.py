import os
import os.path as osp

gt_file = '/cvhci/data/domain_adaption/VisDA-2017/test/ground_truth.txt'
ROOT_DIR = '/cvhci/data/domain_adaption/VisDA-2017/test/'
OUT_ROOT_DIR = '/cvhci/data/domain_adaption/VisDA-2017/test_new/'

labelmap = {0: "aeroplane",
            1: "bicycle",
            2: "bus",
            3: "car",
            4: "horse",
            5: "knife",
            6: "motorcycle",
            7: "person",
            8: "plant",
            9: "skateboard",
            10: "train",
            11: "truck"}

with open(gt_file, 'r') as f:
    for line in f.readlines():
        path, label = line.strip().split(' ')
        full_path = osp.join(ROOT_DIR, path)
        label = int(label)
        assert 0 <= label <= 11

        output_path = osp.join(OUT_ROOT_DIR, labelmap[label], osp.basename(full_path))

        print(full_path, output_path)
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        os.symlink(full_path, output_path)
