# Install environment
```
conda create -n coffea_torch coffea pytorch tensorboard
conda activate coffea_torch
```

# Run
```
python python/analysis.py
python python/normalize.py # compute norm for threeTag to fourTag in SB
python python/analysis.py --normalize --save # run again, this time normalizing threeTag to fourTag. Save coffea files with derived quantities for use in classifier training
python python/plots.py
```
Once you have produced the FvT reweight files (using the training steps below), you can run again and plot the results
```
python python/analysis.py --normalize # Make FvT hists without reweighting
python python/analysis.py --reweight  # Make FvT hists with reweighting
python python/plots.py --hists data/hists_normalized.pkl --plots plots/normalized
python python/plots.py --hists data/hists_reweighted.pkl --plots plots/reweighted
```

# Train
Train Four vs ThreeTag (FvT) models to do kinematic reweighting of ThreeTag data to look like FourTag data.
This is done using a 3-fold training with one third of the data withheld for validation.
By training three times, each time withholding a different third, we can compute the classifier output for every event without ever using a training sample event.
This procedure avoids the overconfidence/miscalibration of the joint probability distribution estimate that will always be present in the training set. With the implementation of SvB classifier, an additional argument ``--task`` has been added to specify the type of classifier to train (FvT or SvB).
```
python python/train.py --train --task FvT --offset 0
python python/train.py --train --task FvT --offset 1
python python/train.py --train --task FvT --offset 2
```
We can precompute friend TTrees of the 3-fold model output to make it fast and easy to plot the results in relation to other kinematic quantities.
```
python python/train.py --model "models/FvT_Basic_CNN_8_offset_*_epoch_20.pkl"
```

# Autoencoder
Train the autoencoder to reproduce samples in output with:
```
python python/train.py --train --task dec --offset 0
python python/train.py --train --task dec --offset 1
python python/train.py --train --task dec --offset 2
```
The reduced-dimension step is specified inside train.py within "Architecture hyperparameters" as ``bottleneck_dim``. Same as for the FvT and SvB classifiers, also friend TTrees can be computed of the 3-fold model:
```
python python/train.py --model "models/dec_fourTag_10x_Basic_CNN_6_offset_*_epoch_25.pkl"
```

## Sampling
Running the model like before generates a file containing the activations in the bottleneck. These can be used to randomly sample a new synthetic dataset. To do it:
```
python python/train.py --generate --model "models/dec_fourTag_10x_Basic_CNN_AE_6_offset_*_epoch_025.pkl"
```
