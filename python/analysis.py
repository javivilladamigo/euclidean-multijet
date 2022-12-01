# conda create -n coffea_torch coffea pytorch
# conda activate coffea_torch

import awkward as ak
import numpy as np

# https://coffeateam.github.io/coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")

