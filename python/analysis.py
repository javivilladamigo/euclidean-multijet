# conda create -n coffea_torch coffea pytorch
# conda activate coffea_torch

import time
import awkward as ak
import numpy as np

import hist # https://hist.readthedocs.io/en/latest/index.html

# https://coffeateam.github.io/coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor, util
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")


def count_nested_dict(nested_dict, c=0):
    for key in nested_dict:
        if isinstance(nested_dict[key], dict):
            c = count_nested_dict(nested_dict[key], c)
        else:
            c += 1
    return c


class analysis(processor.ProcessorABC):
    def __init__(self):

        self._accumulator = processor.dict_accumulator({'cutflow': processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, partial(processor.defaultdict_accumulator, float) ) ),
                                                        'nEvent' : processor.defaultdict_accumulator(int),
                                                        'hists'  : processor.dict_accumulator()})

        dataset_axis = hist.axis.StrCategory([], growth=True, name='dataset', label='Dataset')
        mass_axis = hist.axis.Regular(300, 0, 1500, name='mass', label=r'$m_{4j}$ [GeV]')
        self._accumulator['hists']['passPreSel']['inclusive']['m4j'] = hist.Hist(dataset_axis, mass_axis, storage='weight', label='Events')
        
        self.nHists = count_nested_dict(self._accumulator['hists'])

        print(f'{self.nHists} total histograms')


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, event):
        tstart = time.time()
        np.random.seed(0)
        output = self.accumulator.identity()

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '

        nEvent = len(event)
        output['nEvent'][dataset] += nEvent

        output['cutflow'][dataset]['All Events'] += np.sum(event.weight)

        
