# Beyond Alignment: Discovering Cross-graph Triples for Knowledge Graph Integration

## Overview
This repository provides the benchmark datasets.

## Data 

**WIKI-YAGO**:  A Public dataset from  https://github.com/nju-websoft/CLP.
**DBP-FB**:  A Public dataset from  https://github.com/nju-websoft/CLP.

**ES-EN**: A novel benchmark constructed for the cross-graph triple prediction. 
**JA-EN**: A novel benchmark constructed for the cross-graph triple prediction. 

Here, we illustrate the files and folders contained in each benchmark. 

- KG1/KG2: Folders containing the triples (train.txt, valid.txt, test.txt) for each individual Knowledge Graph.
- cross: Folder containing the cross-graph information.
  - known_shared_entities.txt: List of known shared entities used for training.
  - cross_test.txt: List of cross-KG triples for testing.
  - all_shared_entities.txt: List of all shared entities between the KGs.



## Dataset Statistics

We provide the statistics of the benchmark datasets.

| Dataset | KGs | Entities | Relations | Known shared entities | Inner-KG triples | Cross-KG test triples |
|---|---|---|---|---|---|---|
| **WIKI-YAGO** | WIKI | 12,550 | 130 | 2,133 | 137,231 | 36,166 |
| | YAGO | 16,319 | 33 | | 224,434 | |
| **DBP-FB** | DBP | 12,562 | 263 | 1,836 | 76,542 | 26,508 |
| | FB | 11,235 | 236 | | 188,542 | |
| **ES-EN** | ES | 10,108 | 241 | 2,012 | 54,066 | 15,841 |
| | EN | 13,132 | 861 | | 80,167 | |
| **JA-EN** | JA | 7,473 | 172 | 1,565 | 28,774 | 8,477 |
| | EN | 13,132 | 861 | | 80,167 | |





