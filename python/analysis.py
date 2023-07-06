# conda create -n coffea_torch coffea pytorch
# conda activate coffea_torch

import time, pickle, os, argparse
import awkward as ak
import numpy as np
from functools import partial

# https://hist.readthedocs.io/en/latest/index.html
import hist

# https://coffeateam.github.io/coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor, util
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
from ClassifierSchema import ClassifierSchema

NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")



class analysis(processor.ProcessorABC):
    def __init__(self, save=False, fvt='FvT'):
        self.debug = False
        self.save = save
        self.fvt = fvt
    
    def process(self, event):
        tstart = time.time()
        np.random.seed(0)
        
        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        norm    = event.metadata.get('normalize', None)
        if norm:
            with open(norm, 'rb') as nfile:
                norm = pickle.load(nfile)['norm']
                event['weight'] = norm * event.weight

        fvt_path = dataset.replace('picoAOD', self.fvt)
        fvt_exists = os.path.exists(fvt_path)
        if fvt_exists:
            event['FvT'] = NanoEventsFactory.from_root(fvt_path, entry_start=estart, entry_stop=estop, schemaclass=ClassifierSchema).events().FvT

        if event.metadata.get('reweight', False) and 'threeTag' in dataset:
            event['weight'] = event.FvT.rw * event.weight

        dataset_axis = hist.axis.StrCategory([], growth=True, name='dataset', label='Dataset')
        cut_axis     = hist.axis.StrCategory([], growth=True, name='cut',     label='Cut')
        region_axis  = hist.axis.StrCategory([], growth=True, name='region',  label='Region')

        output = {'hists': {},
                  'cutflow': hist.Hist(dataset_axis, cut_axis, region_axis, storage='weight', label='Events'),
                  'sumw': ak.sum(event.weight),
                  'nEvent': len(event)}

        m4j_axis = hist.axis.Regular(300, 0, 1500, name='mass', label=r'$m_{4j}$ [GeV]')
        output['hists']['m4j'] = hist.Hist(dataset_axis,
                                           cut_axis,
                                           region_axis,
                                           m4j_axis,
                                           storage='weight', label='Events')

        lead_st_m2j_axis = hist.axis.Regular(100, 0, 500, name='lead', label=   r'Lead $S_{T}$ $m_{2j}$ [GeV]')
        subl_st_m2j_axis = hist.axis.Regular(100, 0, 500, name='subl', label=r'Sublead $S_{T}$ $m_{2j}$ [GeV]')
        output['hists']['lead_st_m2j_subl_st_m2j'] = hist.Hist(dataset_axis,
                                                               cut_axis,
                                                               region_axis,
                                                               lead_st_m2j_axis,
                                                               subl_st_m2j_axis,
                                                               storage='weight', label='Events')
        lead_st_dr_axis = hist.axis.Regular(60, 0, 6, name='lead', label=   r'Lead $S_T$ $\Delta R(j,j)$')
        subl_st_dr_axis = hist.axis.Regular(60, 0, 6, name='subl', label=r'Sublead $S_T$ $\Delta R(j,j)$')
        output['hists']['lead_st_dr_subl_st_dr'] = hist.Hist(dataset_axis,
                                                             cut_axis,
                                                             region_axis,
                                                             lead_st_dr_axis,
                                                             subl_st_dr_axis,
                                                             storage='weight', label='Events')

        if fvt_exists:
            fvt_axis = hist.axis.Regular(50, 0, 2.5, name='rw', label='FvT P(D4)/P(D3)')
            output['hists']['FvT_rw'] = hist.Hist(dataset_axis,
                                                  cut_axis,
                                                  region_axis,
                                                  fvt_axis,
                                                  storage='weight', label='Events')

        # compute four-vector of sum of jets, for the toy samples there are always four jets
        v4j = event.Jet.sum(axis=1)
        event['v4j'] = v4j

        output['cutflow'].fill(dataset=dataset, cut='all', region=['inclusive']*len(event), weight=event.weight) # bug in boost_histogram, waiting for fix, https://github.com/scikit-hep/boost-histogram/issues/452
        output['hists']['m4j'].fill(dataset=dataset, cut='all', region='inclusive', mass=event.v4j.mass, weight=event.weight)

        # Jet selection
        event['Jet', 'selected'] = (event.Jet.pt>=40) & (np.abs(event.Jet.eta)<=2.4)
        event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
        event['preselection'] = (event.nJet_selected>=4)
        #event = event[event.nJet_selected>=4]

        selev = event[event.preselection]
        output['cutflow'].fill(dataset=dataset, cut='preselection', region=['inclusive']*len(selev), weight=selev.weight)
        output['hists']['m4j'].fill(dataset=dataset, cut='preselection', region='inclusive', mass=selev.v4j.mass, weight=selev.weight)

        #
        # Build diJets, indexed by diJet[event,pairing,0/1]
        #
        pairing = [([0,2],[0,1],[0,1]),
                   ([1,3],[2,3],[3,2])]
        diJet         = event.Jet[:,pairing[0]]     +   event.Jet[:,pairing[1]]
        diJet['st']   = event.Jet[:,pairing[0]].pt  +   event.Jet[:,pairing[1]].pt
        diJet['dr']   = event.Jet[:,pairing[0]].delta_r(event.Jet[:,pairing[1]])
        diJet['lead'] = event.Jet[:,pairing[0]]
        diJet['subl'] = event.Jet[:,pairing[1]]
        # Sort diJets within pairings to be lead st, subl st
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
        # Now indexed by diJet[event,pairing,lead/subl st]

        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[ 52, 50]]])
        maxDiJetMass = np.array([[[180,173]]])
        diJet['diJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

        # Compute sliding window delta_r criteria (drc)
        min_m4j_scale = np.array([[ 360, 235]])
        min_dr_offset = np.array([[-0.5, 0.0]])
        max_m4j_scale = np.array([[ 650, 650]])
        max_dr_offset = np.array([[ 0.5, 0.7]])
        max_dr        = np.array([[ 1.5, 1.5]])
        m4j = np.repeat(np.reshape(np.array(event.v4j.mass), (-1,1,1)), 2, axis=2)
        diJet['drc'] = (min_m4j_scale/m4j + min_dr_offset < diJet.dr) & (diJet.dr < np.maximum(max_m4j_scale/m4j + max_dr_offset, max_dr))

        # Compute consistency of diJet masses with higgs boson mass
        mH = 125.0
        st_bias = np.array([[[1.02, 0.98]]])
        cH = mH * st_bias
        diJet['xH'] = (diJet.mass - cH)/(0.1*diJet.mass)

        #
        # Build quadJets
        #
        quadJet = ak.zip({'lead': diJet[:,:,0],
                          'subl': diJet[:,:,1],
                          'diJetMass': ak.all(diJet.diJetMass, axis=2),
                          'random': np.random.uniform(low=0.1, high=0.9, size=(diJet.__len__(), 3))
                          })#, with_name='quadJet')
        quadJet['dr'] = quadJet['lead'].delta_r(quadJet['subl'])
        # Compute Region
        quadJet['xHH'] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
        max_xHH = 1.9
        quadJet['SR'] = quadJet.xHH < max_xHH
        quadJet['SB'] = quadJet.diJetMass & ~quadJet.SR
        
        # pick quadJet at random giving preference to ones which pass diJetMass and drc's
        quadJet['rank'] = 10*quadJet.lead.diJetMass + 10*quadJet.subl.diJetMass + quadJet.lead.drc + quadJet.subl.drc + quadJet.random
        quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

        event[  'diJet'] =   diJet
        event['quadJet'] = quadJet
        event['quadJet_selected'] = quadJet[quadJet.selected][:,0]
        event['diJetMass'] = event.quadJet_selected.diJetMass
        event['SB'] = event.quadJet_selected.SB
        event['SR'] = event.quadJet_selected.SR

        mask = event.preselection
        self.fill(output, event[mask], dataset=dataset, cut='preselection', region='inclusive')
        mask = event.preselection & event.diJetMass
        self.fill(output, event[mask], dataset=dataset, cut='preselection', region='diJetMass')
        mask = event.preselection & event.SB
        self.fill(output, event[mask], dataset=dataset, cut='preselection', region='SB')
        mask = event.preselection & event.SR
        self.fill(output, event[mask], dataset=dataset, cut='preselection', region='SR')

        if self.save:
            util.save(event, dataset.replace('.root',f'_{estart:07d}_{estop:07d}.coffea'))
                
        # Done
        elapsed = time.time() - tstart
        if self.debug: print(f'{chunk}{nEvent/elapsed:,.0f} events/s')
        return output

    def fill(self, output, event, dataset='', cut='', region=''):
        output['cutflow'].fill(
            dataset=dataset, cut=cut, region=[region]*len(event),
            weight=event.weight)
        output['hists']['m4j'].fill(
            dataset=dataset, cut=cut, region=region,
            mass=event.v4j.mass, weight=event.weight)
        output['hists']['lead_st_m2j_subl_st_m2j'].fill(
            dataset=dataset, cut=cut, region=region,
            lead=event.quadJet_selected.lead.mass, subl=event.quadJet_selected.subl.mass, weight=event.weight)
        output['hists']['lead_st_dr_subl_st_dr'].fill(
            dataset=dataset, cut=cut, region=region,
            lead=event.quadJet_selected.lead.dr, subl=event.quadJet_selected.subl.dr, weight=event.weight)
        if 'FvT_rw' in output['hists']:
            output['hists']['FvT_rw'].fill(
                dataset=dataset, cut=cut, region=region,
                rw=event.FvT.rw, weight=event.weight)
            
    def postprocess(self, accumulator):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--normalize', default=False, action='store_true', help='Normalize threeTag sample to fourTag in SB')
    parser.add_argument('--normfile', default='data/normalize.pkl', help='Normalize file')
    parser.add_argument('-s', '--save',      default=False, action='store_true', help='Save coffea files')
    parser.add_argument('-r', '--reweight',  default=False, action='store_true', help='Apply FvT kinematic reweight to threeTag sample')
    parser.add_argument('--fvt', default='FvT', help='string to replace picoAOD for FvT reweight file')
    args = parser.parse_args()

    datasets  = []
    datasets += ['data/fourTag_picoAOD.root']
    datasets += ['data/threeTag_picoAOD.root']
    datasets += ['data/HH4b_picoAOD.root']
    datasets += ['data/fourTag_10x_picoAOD.root']

    fileset = {}
    for dataset in datasets:
        fileset[dataset] = {'files': [dataset],
                            'metadata': {}}

    outfile = 'data/hists.pkl'
    if args.normalize or args.reweight:
        print(f'Normalize threeTag with {args.normfile}')
        fileset['data/threeTag_picoAOD.root']['metadata']['normalize'] = args.normfile
        outfile = 'data/hists_normalized.pkl'
    if args.reweight:
        print(f'Apply kinematic reweighting using {args.fvt}')
        fileset['data/threeTag_picoAOD.root']['metadata']['reweight']  = args.reweight
        outfile = 'data/hists_reweighted.pkl'
        
    tstart = time.time()
    output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=analysis(save=args.save, fvt=args.fvt),
                                      executor=processor.futures_executor,
                                      executor_args={'schema': NanoAODSchema, 'workers': 4},
                                      chunksize=100_000,
                                      #maxchunks=1,
                                      )
    elapsed = time.time() - tstart
    nEvent = output['nEvent'] #sum([output['nEvent'][dataset] for dataset in output['nEvent'].keys()])
    print(f'{nEvent/elapsed:,.0f} events/s ({nEvent:,}/{elapsed:,.2f})')


    with open(outfile, 'wb') as hfile:
        print('Save Histograms to',outfile)
        pickle.dump(output, hfile)
