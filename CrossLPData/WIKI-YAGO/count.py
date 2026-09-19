#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

ROOT    = "output_split"
DATASET = ["WIKI1", "WIKI2","YAGO1","YAGO2"]
SPLITS  = ["train", "valid", "test"]
COMPARE = ["valid", "test"]  # 要对比 train 的两个 split

def parse_triples(path):
    """
    解析三元组文件，返回：
      - all_ents    : 所有实体（head + tail）
      - head_ents   : 仅 head 实体
      - rels        : 关系集
    支持空格或制表符分隔。
    """
    all_ents, head_ents, rels = set(), set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            all_ents.update([h, t])
            head_ents.add(h)
            rels.add(r)
    return all_ents, head_ents, rels

def count_splits_and_new():
    print("==== 各 split 实体/头实体/关系 & valid/test 相对 train 的新增统计 ====\n")
    for ds in DATASET:
        print(f"--- Dataset: {ds} ---")
        # 1) 收集各 split 的统计数据
        stats = {}
        for split in SPLITS:
            path = os.path.join(ROOT, ds, f"{split}.txt")
            ents, head_ents, rels = parse_triples(path)
            stats[split] = (ents, head_ents, rels)
            print(
                f"{ds}-{split:5s} → "
                f"实体(总):{len(ents):5d}  "
                f"头实体:{len(head_ents):5d}  "
                f"关系数:{len(rels):5d}"
            )

        # 2) train+valid+test 并集
        union_ents  = set().union(*(stats[s][0] for s in SPLITS))
        union_heads = set().union(*(stats[s][1] for s in SPLITS))
        union_rels  = set().union(*(stats[s][2] for s in SPLITS))
        print(
            f"{ds}-all    → "
            f"实体(总):{len(union_ents):5d}  "
            f"头实体:{len(union_heads):5d}  "
            f"关系数:{len(union_rels):5d}"
        )

        # 3) valid/test 相对 train 的新增
        train_ents, train_heads, train_rels = stats["train"]
        print()
        for split in COMPARE:
            ents, heads, rels = stats[split]
            new_ents  = ents - train_ents
            new_heads = heads - train_heads
            new_rels  = rels - train_rels
            print(
                f"{ds}-{split:5s} → "
                f"新实体: {len(new_ents):5d}  "
                f"新头实体: {len(new_heads):5d}  "
                f"新关系: {len(new_rels):5d}"
            )
        print()

if __name__ == "__main__":
    count_splits_and_new()
