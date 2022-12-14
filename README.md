# Install environment
```
conda create -n coffea_torch coffea pytorch
conda activate coffea_torch
```

# Run
```
python python/analysis.py
python python/normalize.py # compute norm for threeTag to fourTag in SB
python python/analysis.py # run again, this time normalizing threeTag to fourTag
python python/plots.py
```
