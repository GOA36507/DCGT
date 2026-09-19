#!/usr/bin/env python3
import os
import argparse
import random
import shutil

###########################################
# 基础工具函数：读取、写入三元组、目录处理
###########################################

def read_triples(file_path):
    """
    读取单个文件中的三元组（假设以空白符或 tab 分隔，至少3列），返回列表 [(head, relation, tail), ...]
    """
    triples = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            head, relation, tail = parts[0], parts[1], parts[2]
            triples.append((head, relation, tail))
    return triples


def write_triples(triples, file_path):
    """
    将三元组列表写入文件，每行格式：head<tab>relation<tab>tail
    """
    with open(file_path, 'w', encoding='utf-8') as fout:
        for h, r, t in triples:
            fout.write(f"{h}\t{r}\t{t}\n")


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

###########################################
# 数据合并和拆分函数
###########################################

def merge_kg(kg_dir):
    """
    合并 kg_dir 下的 train.txt, valid.txt, test.txt（如果存在）作为整个 KG 的全集
    """
    merged = []
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        file_path = os.path.join(kg_dir, split)
        if os.path.exists(file_path):
            merged += read_triples(file_path)
    return merged


def split_into_two(triples, sub_split_ratio=0.5):
    """
    将合并后的 KG 三元组随机打乱后，分成两部分（比例 sub_split_ratio 与 1-sub_split_ratio）
    """
    random.shuffle(triples)
    n = len(triples)
    n1 = int(n * sub_split_ratio)
    return triples[:n1], triples[n1:]


def split_train_valid_test(triples, train_ratio=0.8, valid_ratio=0.1):
    """
    保证训练集覆盖所有实体和关系的划分：
      1. 贪心收集缺失的实体/关系对应的三元组
      2. 从剩余中随机补齐至期望训练集大小
      3. 剩下的按 valid_ratio 划分验证集和测试集
    """
    total = len(triples)
    # 统计全集实体和关系
    all_entities = set()
    all_relations = set()
    for h, r, t in triples:
        all_entities.update([h, t])
        all_relations.add(r)

    # 贪心收集：新实体或新关系出现时加入训练集
    train = []
    covered_e = set()
    covered_r = set()
    for triple in triples:
        h, r, t = triple
        if (h not in covered_e) or (t not in covered_e) or (r not in covered_r):
            train.append(triple)
            covered_e.update([h, t])
            covered_r.add(r)

    # 剩余候选
    remaining = [tr for tr in triples if tr not in train]
    random.shuffle(remaining)

    # 补齐训练集至期望大小
    desired_train_size = int(train_ratio * total)
    if len(train) < desired_train_size:
        need = desired_train_size - len(train)
        train += remaining[:need]
        remaining = remaining[need:]

    # 划分验证集和测试集
    valid_size = int(valid_ratio * total)
    valid = remaining[:valid_size]
    test = remaining[valid_size:]

    return train, valid, test

###########################################
# 对齐文件生成和过滤函数
###########################################

def generate_same_kg_alignment(train1, train2):
    """
    生成同一KG内的对齐对：根据各自训练集头实体的交集，输出格式 [(entity, entity), ...]
    """
    heads1 = set([triple[0] for triple in train1])
    heads2 = set([triple[0] for triple in train2])
    aligned = heads1.intersection(heads2)
    return [(ent, ent) for ent in sorted(aligned)]


def filter_cross_alignment(alignment_file, kg1_triples, kg2_triples):
    """
    过滤跨KG对齐文件，只保留在 kg1_triples 和 kg2_triples 中出现的实体对
    """
    ents1 = set([h for h,_,_ in kg1_triples] + [t for _,_,t in kg1_triples])
    ents2 = set([h for h,_,_ in kg2_triples] + [t for _,_,t in kg2_triples])
    filtered = []
    with open(alignment_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            e1, e2 = parts[0], parts[1]
            if e1 in ents1 and e2 in ents2:
                filtered.append((e1, e2))
    return filtered

###########################################
# 统计数据指标
###########################################

def count_stats(triples):
    """
    统计三元组列表中的实体数、关系数和三元组数
    """
    entities = set()
    relations = set()
    for (h, r, t) in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return len(entities), len(relations), len(triples)

###########################################
# 主函数
###########################################

def main(args):
    random.seed(args.seed)

    # 1. 合并 WIKI 和 YAGO 原始数据
    print("Merging original WIKI KG ...")
    wiki_all = merge_kg(args.wiki_dir)
    print(f"Total WIKI triples: {len(wiki_all)}")
    wiki_ent, wiki_rel, wiki_tri = count_stats(wiki_all)
    print(f"WIKI All: Entities={wiki_ent}, Relations={wiki_rel}, Triples={wiki_tri}")
    wiki1_all, wiki2_all = split_into_two(wiki_all, sub_split_ratio=args.sub_split_ratio)
    print(f"WIKI1 total: {len(wiki1_all)} ; WIKI2 total: {len(wiki2_all)}")

    print("Merging original YAGO KG ...")
    yago_all = merge_kg(args.yago_dir)
    print(f"Total YAGO triples: {len(yago_all)}")
    yago_ent, yago_rel, yago_tri = count_stats(yago_all)
    print(f"YAGO All: Entities={yago_ent}, Relations={yago_rel}, Triples={yago_tri}")
    yago1_all, yago2_all = split_into_two(yago_all, sub_split_ratio=args.sub_split_ratio)
    print(f"YAGO1 total: {len(yago1_all)} ; YAGO2 total: {len(yago2_all)}")

    # 2. 划分 train/valid/test
    wiki1_train, wiki1_valid, wiki1_test = split_train_valid_test(wiki1_all, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    wiki2_train, wiki2_valid, wiki2_test = split_train_valid_test(wiki2_all, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    yago1_train, yago1_valid, yago1_test = split_train_valid_test(yago1_all, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    yago2_train, yago2_valid, yago2_test = split_train_valid_test(yago2_all, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    # 3. 写入输出目录
    ensure_dir(args.output_dir)
    subdirs = {name: os.path.join(args.output_dir, name) for name in ['WIKI1','WIKI2','YAGO1','YAGO2']}
    for subdir in subdirs.values():
        ensure_dir(subdir)

    write_triples(wiki1_train, os.path.join(subdirs['WIKI1'], "train.txt"))
    write_triples(wiki1_valid, os.path.join(subdirs['WIKI1'], "valid.txt"))
    write_triples(wiki1_test, os.path.join(subdirs['WIKI1'], "test.txt"))
    write_triples(wiki2_train, os.path.join(subdirs['WIKI2'], "train.txt"))
    write_triples(wiki2_valid, os.path.join(subdirs['WIKI2'], "valid.txt"))
    write_triples(wiki2_test, os.path.join(subdirs['WIKI2'], "test.txt"))
    write_triples(yago1_train, os.path.join(subdirs['YAGO1'], "train.txt"))
    write_triples(yago1_valid, os.path.join(subdirs['YAGO1'], "valid.txt"))
    write_triples(yago1_test, os.path.join(subdirs['YAGO1'], "test.txt"))
    write_triples(yago2_train, os.path.join(subdirs['YAGO2'], "train.txt"))
    write_triples(yago2_valid, os.path.join(subdirs['YAGO2'], "valid.txt"))
    write_triples(yago2_test, os.path.join(subdirs['YAGO2'], "test.txt"))

    # 4. 生成对齐文件
    same_dir = os.path.join(args.output_dir, "alignment")
    ensure_dir(same_dir)
    wiki_alignment = generate_same_kg_alignment(wiki1_train, wiki2_train)
    with open(os.path.join(same_dir, "WIKI1-WIKI2_alignment.txt"), 'w', encoding='utf-8') as fout:
        for e1, e2 in wiki_alignment:
            fout.write(f"{e1}\t{e2}\n")
    print(f"WIKI1-WIKI2 alignment pairs: {len(wiki_alignment)}")
    yago_alignment = generate_same_kg_alignment(yago1_train, yago2_train)
    with open(os.path.join(same_dir, "YAGO1-YAGO2_alignment.txt"), 'w', encoding='utf-8') as fout:
        for e1, e2 in yago_alignment:
            fout.write(f"{e1}\t{e2}\n")
    print(f"YAGO1-YAGO2 alignment pairs: {len(yago_alignment)}")

    cross_pairs = [
        (wiki1_all, yago1_all, "WIKI1-YAGO1_alignment.txt"),
        (wiki1_all, yago2_all, "WIKI1-YAGO2_alignment.txt"),
        (wiki2_all, yago1_all, "WIKI2-YAGO1_alignment.txt"),
        (wiki2_all, yago2_all, "WIKI2-YAGO2_alignment.txt"),
    ]
    for kgA, kgB, fname in cross_pairs:
        pairs = filter_cross_alignment(args.known_alignment, kgA, kgB)
        with open(os.path.join(same_dir, fname), 'w', encoding='utf-8') as fout:
            for e1, e2 in pairs:
                fout.write(f"{e1}\t{e2}\n")
        print(f"{fname.split('_')[0]} alignment pairs (filtered): {len(pairs)}")

    # 5. 打印统计
    def print_stats(name, all_triples, train, valid, test):
        oe, or_, ot = count_stats(all_triples)
        te, tr, tt = count_stats(train)
        ve, vr, vt = count_stats(valid)
        se, sr, st = count_stats(test)
        print(f"--- {name} Statistics ---")
        print(f"Overall: Entities={oe}, Relations={or_}, Triples={ot}")
        print(f"Train:   Entities={te}, Relations={tr}, Triples={tt}")
        print(f"Valid:   Entities={ve}, Relations={vr}, Triples={vt}")
        print(f"Test:    Entities={se}, Relations={sr}, Triples={st}")

    print("\n========== Subgraph Statistics ==========\n")
    print_stats("WIKI1", wiki1_all, wiki1_train, wiki1_valid, wiki1_test)
    print_stats("WIKI2", wiki2_all, wiki2_train, wiki2_valid, wiki2_test)
    print_stats("YAGO1", yago1_all, yago1_train, yago1_valid, yago1_test)
    print_stats("YAGO2", yago2_all, yago2_train, yago2_valid, yago2_test)

    print("\nAll splitting and alignment generation done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split two KGs into sub-KGs and generate splits with coverage guarantee."
    )
    parser.add_argument("--wiki_dir", type=str, required=True,
                        help="Directory for original WIKI KG.")
    parser.add_argument("--yago_dir", type=str, required=True,
                        help="Directory for original YAGO KG.")
    parser.add_argument("--known_alignment", type=str, required=True,
                        help="File path for known cross KG alignment.")
    parser.add_argument("--output_dir", type=str, default="output_split",
                        help="Output directory for splits and alignments.")
    parser.add_argument("--sub_split_ratio", type=float, default=0.5,
                        help="Ratio to split merged KG into two parts.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio with full coverage guarantee.")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Validation set ratio; test gets the rest.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)
