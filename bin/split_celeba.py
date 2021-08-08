import os
import argparse
import random
import numpy as np
from utils.misc import get_datetime, str2bool
from utils.file import copy, prepare_dirs, write_record
from tqdm import tqdm

celeba_attr_names = [
    '新双颊胡须', '柳叶眉', '吸引人', '眼袋', '秃头',
    '刘海', '大嘴唇', '大鼻子', '黑发', '金发',
    '模糊', '棕发', '浓眉', '圆胖', '双下巴',
    '戴眼镜', '山羊胡子', '灰发', '浓妆', '高颧骨',
    '男性', '微张嘴巴', '八字胡', '细眼睛', '无胡子',
    '椭圆脸', '苍白皮肤', '尖鼻子', '高发际线', '红润双颊',
    '络腮胡', '微笑', '直发', '卷发', '戴耳环',
    '戴帽子', '涂唇膏', '戴项链', '戴领带', '年轻人']

celeba_attr_names_opposite = [
    '无新双颊胡须', '非柳叶眉', '不吸引人', '无眼袋', '非秃头',
    '无刘海', '非大嘴唇', '非大鼻子', '非黑发', '非金发',
    '清晰', '非棕发', '非浓眉', '非圆胖', '非双下巴',
    '无眼镜', '非山羊胡子', '非灰发', '非浓妆', '非高颧骨',
    '女性', '非微张嘴巴', '非八字胡', '非细眼睛', '有胡子',
    '非椭圆脸', '非苍白皮肤', '非尖鼻子', '非高发际线', '非红润双颊',
    '非络腮胡', '非微笑', '非直发', '非卷发', '无耳环',
    '无帽子', '无唇膏', '无项链', '无领带', '非年轻人']


def process_labels(cfg, attr_file):
    with open(os.path.join(cfg.dataset_path, attr_file)) as f:
        lines = f.read().splitlines()
        sample_num = int(lines[0])
        attr_num = len(lines[1].strip().split(' '))
        attr_matrix = np.zeros((sample_num, attr_num))
        for i in range(sample_num):
            line = lines[i + 2].strip().split()[1:]
            assert len(line) == attr_num
            attr_matrix[i] = line
        attr_matrix = attr_matrix.T
        attr_dict = {}
        for i in range(attr_num):
            key = f"a{i}"
            attr_dict[key] = np.where(attr_matrix[i] == 1)[0]
        return attr_matrix, attr_dict


def main(cfg):
    random.seed(cfg.seed)
    os.makedirs(cfg.output_path)
    if cfg.is_hq:
        attr_file = "CelebAMask-HQ-attribute-anno.txt"
        src_path = "CelebA-HQ-img"
    else:
        attr_file = "list_attr_celeba.txt"
        src_path = "images"

    record_path = os.path.join(cfg.output_path, 'README.txt')
    write_record(str(args.__dict__), record_path)
    attr_matrix, attr_dict = process_labels(cfg, attr_file)
    selected_attrs = [[32], [-32]]  # smiling
    attr_names = []
    if len(attr_names) == 0:
        attr_names = [','.join([(celeba_attr_names[a - 1] if a > 0 else celeba_attr_names_opposite[-a - 1]) for a in x])
                      for x in selected_attrs]
    indices_list = []

    max_sample_num = 0
    for i in range(len(attr_names)):
        selected_attr = selected_attrs[i]
        indices = np.ones((attr_matrix.shape[1],), dtype=bool)
        for a in selected_attr:
            indices *= attr_matrix[abs(a) - 1] == (1 if a > 0 else -1)
        write_record(f"{attr_names[i]}({selected_attrs[i]}): {indices.sum()}", record_path)
        max_sample_num = max(max_sample_num, indices.sum())
        indices_list.append(np.where(indices)[0])
    if cfg.fixed_num:
        max_sample_num = cfg.fixed_num
    input(f"Press enter to start splitting dataset...")

    src_path = os.path.join(cfg.dataset_path, src_path)
    for indices, attr_name in zip(indices_list, attr_names):
        indices = indices[:max_sample_num]
        if cfg.test_num:
            train_num = len(indices) - cfg.test_num
        else:
            train_num = int(len(indices) * (1 - cfg.ratio))
        random.shuffle(indices)
        train_indices = indices[:train_num]
        test_indices = indices[train_num:]
        train_dst_path = os.path.join(cfg.output_path, 'train', attr_name)
        test_dst_path = os.path.join(cfg.output_path, 'test', attr_name)
        prepare_dirs([train_dst_path, test_dst_path])
        for i in tqdm(train_indices):
            if cfg.is_hq:
                filename = f"{i}.jpg"
            else:
                filename = f"{i + 1:06}.jpg"
            copy(filename, src_path, train_dst_path)
        for i in tqdm(test_indices):
            if cfg.is_hq:
                filename = f"{i}.jpg"
            else:
                filename = f"{i + 1:06}.jpg"
            copy(filename, src_path, test_dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"D:\Data\CelebAMask-HQ")
    parser.add_argument('--save_path', type=str, default=r"D:\Data\celeba_splits")
    parser.add_argument('--save_name', type=str, default=get_datetime(True))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--test_num', type=int, default=1412)
    parser.add_argument('--is_hq', type=str2bool, default=True)
    parser.add_argument('--fixed_num', type=int)
    args = parser.parse_args()
    args.output_path = os.path.join(args.save_path, args.save_name)
    main(args)
