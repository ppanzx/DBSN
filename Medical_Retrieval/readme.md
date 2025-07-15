## Data
We organize all data used in the experiments in the same manner as [PLIP](https://github.com/PathologyFoundation/plip):

```
data
├── books_set
│   │
│   ├── images   # raw books_set images
│   │      ├── 0a1d36e7-29d8-406a-9e8c-ede16a11a4c2.png
│   │      └── ...
│   │
│   └── captions.json  # mapping from books_set id to image's file name
│   
├── pubmed_set
│   │
│   ├── images   # raw pubmed_set images
│   │      ├── 0a0a7c19-862b-441e-afa7-76295542c576.jpg
│   │      └── ...
│   │
│   └── captions.json  # mapping from pubmed_set id to image's file name
│
└── readme.md
```

## run
```bash
sh zero_shot_retrieval.sh
```
