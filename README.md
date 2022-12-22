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

# Train
Train Four vs ThreeTag (FvT) models to do kinematic reweighting of ThreeTag data to look like FourTag data.
This is done using a 3-fold training with one third of the data withheld for validation.
By training three times, each time withholding a different third, we can compute the classifier output for every event without ever using a training sample event.
This procedure avoids the overconfidence/miscalibration of the joint probability distribution estimate that will always be present in the training set.
```
python python/train.py --train --offset 0
python python/train.py --train --offset 1
python python/train.py --train --offset 2
```
We can precompute friend TTrees of the 3-fold model output to make it fast and easy to plot the results in relation to other kinematic quantities.
```
python python/train.py --update --model "models/"
```