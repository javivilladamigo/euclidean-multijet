import uproot
import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

# convert = [('fourTag_toyTree.root', 'fourTag_picoAOD.root'),
#            ('fourTag_10x_toyTree.root', 'fourTag_10x_picoAOD.root'),
#            ('threeTag_toyTree.root', 'threeTag_picoAOD.root'),
#            ('HH4b_dR04_toyTree.root', 'HH4b_picoAOD.root')]

convert = []
convert.append(('HH4b_toyTree.root', 'HH4b_picoAOD.root'))

for toyFile, picoFile in convert:
    tree = NanoEventsFactory.from_root(toyFile, treepath='Tree', schemaclass=BaseSchema).events()

    with uproot.recreate(picoFile) as outfile:
        # tree['jetMass'] = 0*tree.jetEnergy # jet masses were set to zero when the toy trees were made so we don't need to recalc from pt, eta, phi, energy. Mult by jetEnergy just to get correct dtype
        outfile['Events'] = {'Jet': ak.zip({'pt'  : tree.jetPt,
                                            'eta' : tree.jetEta,
                                            'phi' : tree.jetPhi,
                                            'mass': tree.jetMass}, with_name='PtEtaPhiMLorentzVector'),
                             'GenPart': ak.zip({'pt'  : tree.jetPtGen,
                                                'eta' : tree.jetEtaGen,
                                                'phi' : tree.jetPhiGen,
                                                'mass': tree.jetMassGen,
                                                'pdgId': tree.jetPID}, with_name='PtEtaPhiMLorentzVector'),
                             'weight': tree.weight,
                             'run': np.zeros(len(tree)),
                             'luminosityBlock': np.zeros(len(tree)),
                             'event': np.arange(len(tree))}


# NanoAODSchema.warn_missing_crossrefs = False
# import warnings
# warnings.filterwarnings("ignore")
