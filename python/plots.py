import pickle, os, argparse
import hist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def mkpath(path, debug=False):
    if os.path.exists(path) and debug:
        print("#",path,"already exists")
        return
        
    thisDir = ''
    for d in path.split('/'):
        thisDir = thisDir+d+"/"
        try:
            os.mkdir(thisDir)
        except FileExistsError:
            if debug: print(f'# {thisDir} already exists')

            

def standard(hdict, var, cut='preselection', region='SB', project='', name='', xlim=[], plotsdir='plots'):
    try:
        h = hdict['hists'][var]
        h3 = h['data/threeTag_picoAOD.root', cut, region, ...]
        h4 = h['data/fourTag_picoAOD.root',  cut, region, ...]
        hh = h['data/HH4b_picoAOD.root',     cut, region, ...]
    except KeyError:
        print(f'Could not find hist(s)', var, cut, region)

    if project:
        h3 = h3.project(project)
        h4 = h4.project(project)
        hh = hh.project(project)

    fig = plt.figure(figsize=(10, 8))
    main_ax_artists, ratio_ax_arists = h4.plot_ratio(
        h3,
        rp_ylabel=r'Four/Three',
        rp_num_label='FourTag',
        rp_denom_label='ThreeTag',
        rp_uncert_draw_type='line',  # line or bar
    )
    axes = fig.get_axes()
    axes[1].set_ylim([0,2])

    if xlim:
        for ax in axes:
            ax.set_xlim(xlim)

    hh.plot1d(ax=axes[0], label=r'SM HH$\rightarrow b\bar{b}b\bar{b}$')
    axes[0].legend()
    
    outdir = f'{plotsdir}/{cut}/{region}'
    mkpath(outdir)
    if name:
        name = f'{outdir}/{name}.pdf'
    else:
        name = f'{outdir}/{var}.pdf'
    print(name)
    fig.savefig(name)
    plt.close()


def sample2D(hdict, sample, var, cut='preselection', region='SB', name='', xlim=[], ylim=[], plotsdir='plots'):
    fig = plt.figure(figsize=(8, 8))

    try:
        h = hdict['hists'][var][f'data/{sample}_picoAOD.root', cut, region, ...]
    except KeyError:
        print(f'Could not find hist hdict["hists"][{var}]["data/{sample}_picoAOD.root", {cut}, {region}, ...]')
        return
    
    h.plot2d_full(
        main_cmap="coolwarm",
        top_ls="--",
        top_color="orange",
        top_lw=2,
        side_ls=":",
        side_lw=2,
        side_color="steelblue",
    )
    axes = fig.get_axes()
    if xlim:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
    if ylim:
        axes[0].set_ylim(ylim)
        axes[2].set_ylim(ylim)

    outdir = f'{plotsdir}/{cut}/{region}'
    mkpath(outdir)
    if name:
        name = f'{outdir}/{name}.pdf'
    else:
        name = f'{outdir}/{sample}_{var}.pdf'
    print(name)
    fig.savefig(name)
    plt.close()


def plot_training_residuals(true_val, reco_val, offset, epoch): # expects [batch, (3) features, (4) jets] shaped tensors
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import torch
    #from fast_histogram import histogram2d

    true_val = true_val.detach()
    reco_val = reco_val.detach()
    res = reco_val - true_val
    res_norm = res / true_val

    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0
    for i, feature in enumerate(["$|S_{T}|\ ({\\rm GeV)}$", "$|\Delta\eta|$", "$|\Delta\phi|\ ({\\rm rad})$"]):
        
        '''
        Implementation of fast histogram is weird: histogram2d produces a 2d plot that makes NO sense in the confrontation of y vs x (the correlation is lost somehow)
        bounds = [(true_val[:, i, :].min(), true_val[:, i, :].max()), (reco_val[:, i, :].min(), reco_val[:, i, :].max())]
        h = histogram2d(true_val[:, i, :], reco_val[:, i, :], range=bounds, bins=100) # get the histogram of the i-th feature for all the events and all the jets
        im = ax[i].imshow(h.T, cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = h.max()), extent= [*bounds[0], *bounds[1]], aspect = 'auto')
        '''
        
        h2d, xbins, ybins, im = ax[i].hist2d(true_val[:, i, :].flatten().numpy(), res[:, i, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))

        ax[i].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[i].minorticks_on()
        ax[i].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)

        ax[i].set_xlabel(f'True {feature}')
        ax[i].set_ylabel(f'Reco - true {feature}')

        #ax[i].plot(xbins, xbins, lw = 2., c = 'grey', ls = '-.')
        ax[i].axhline(y = 0, lw = 2., c = 'grey', ls = '-.')
        
        if h2d.max() > vmax_mob:
            im_vmax = im
            vmax_mob = h2d.max()
    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.3)
    fig.suptitle(f'Epoch {epoch}')
    path = "plots/autoencoder/reco_vs_true/relative_features/"
    mkpath(path)
    fig.savefig(f'{path}residuals_offset_{offset}_epoch_{epoch:02d}.pdf')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hists', default='data/hists_normalized.pkl', help='File containing hists to be plotted')
    parser.add_argument('--plots', default='plots', help='base directory to save plots')
    args = parser.parse_args()

    with open(args.hists, 'rb') as hfile:
        hdict = pickle.load(hfile)

    for cut in ['preselection']:
        for region in ['inclusive', 'diJetMass', 'SB', 'SR']:
            standard(hdict, 'm4j',    cut=cut, region=region, plotsdir=args.plots)
            standard(hdict, 'FvT_rw', cut=cut, region=region, plotsdir=args.plots)

            xlim = [40,200] if region!='inclusive' else []
            ylim = [40,200] if region!='inclusive' else []
            standard(hdict, 'lead_st_m2j_subl_st_m2j', project='lead', name='lead_st_m2j', cut=cut, region=region, xlim=xlim, plotsdir=args.plots)
            standard(hdict, 'lead_st_m2j_subl_st_m2j', project='subl', name='subl_st_m2j', cut=cut, region=region, xlim=xlim, plotsdir=args.plots)

            standard(hdict, 'lead_st_dr_subl_st_dr', project='lead', name='lead_st_dr', cut=cut, region=region, plotsdir=args.plots)
            standard(hdict, 'lead_st_dr_subl_st_dr', project='subl', name='subl_st_dr', cut=cut, region=region, plotsdir=args.plots)
            
            for sample in ['threeTag', 'fourTag', 'HH4b']:
                sample2D(hdict, sample, 'lead_st_m2j_subl_st_m2j', cut=cut, region=region, xlim=xlim, ylim=ylim, plotsdir=args.plots)

                sample2D(hdict, sample, 'lead_st_dr_subl_st_dr', cut=cut, region=region, plotsdir=args.plots)
