import pickle, os, argparse
import hist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networks


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
        top_ls="-", # before was "--"
        top_color="orange",
        top_lw=2,
        side_ls="-", # before was ":"
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


def plot_training_residuals_PxPyPzEm2jm4jPt(true_val, reco_val, phi_rot, offset, epoch, sample, network_name): # expects [batch, (3) features, (4) jets] shaped tensors
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    plot_masses = True
    if plot_masses:
        d_rot, dPxPyPzE_rot = networks.addFourVectors(  networks.PtEtaPhiM(true_val)[:,:,(0,2,0,1,0,1)], 
                                                        networks.PtEtaPhiM(true_val)[:,:,(1,3,2,3,3,2)])
        q_rot, qPxPyPzE_rot = networks.addFourVectors(  d_rot[:,:,(0,2,4)],
                                                        d_rot[:,:,(1,3,5)], 
                                                        v1PxPyPzE = dPxPyPzE_rot[:,:,(0,2,4)],
                                                        v2PxPyPzE = dPxPyPzE_rot[:,:,(1,3,5)])
        m2j = d_rot[:, 3:4, :]
        m4j = q_rot[:, 3:4, 0]

        rec_d, rec_dPxPyPzE = networks.addFourVectors(  networks.PtEtaPhiM(reco_val)[:,:,(0,2,0,1,0,1)], 
                                                        networks.PtEtaPhiM(reco_val)[:,:,(1,3,2,3,3,2)])
        rec_q, rec_qPxPyPzE = networks.addFourVectors(  rec_d[:,:,(0,2,4)],
                                                        rec_d[:,:,(1,3,5)])
        rec_m2j = rec_d[:, 3:4, :]
        rec_m4j = rec_q[:, 3:4, 0]

        true_m2j = m2j.detach()
        reco_m2j = rec_m2j.detach()
        res_m2j = reco_m2j - true_m2j

        true_m4j = m4j.detach()
        reco_m4j = rec_m4j.detach()
        res_m4j = reco_m4j - true_m4j

    true_val = true_val.detach()
    reco_val = reco_val.detach()
    res = reco_val - true_val
    res_norm = res / true_val



    true_pt = ((true_val[:, 0:1, :]**2 + true_val[:, 1:2, :]**2).sqrt()).detach()
    reco_pt = ((reco_val[:, 0:1, :]**2 + reco_val[:, 1:2, :]**2).sqrt()).detach()
    res_pt = reco_pt - true_pt

    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(2, 4, figsize = (15, 5))
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0
    i, j = 0, 0
    for feature in ["$p_{x}\ ({\\rm GeV)}$", "$p_{y}\ ({\\rm GeV)}$", "$p_{z}\ ({\\rm GeV)}$", "$E\ ({\\rm GeV)}$", "$m_{2j}\ ({\\rm GeV)}$", "$m_{4j}\ ({\\rm GeV)}$", "$p_{T}\ ({\\rm GeV)}$"]:
        if j > 3:
            i += 1
            j -= 4

        '''
        Implementation of fast histogram is weird: histogram2d produces a 2d plot that makes NO sense in the confrontation of y vs x (the correlation is lost somehow)
        bounds = [(true_val[:, i, :].min(), true_val[:, i, :].max()), (reco_val[:, i, :].min(), reco_val[:, i, :].max())]
        h = histogram2d(true_val[:, i, :], reco_val[:, i, :], range=bounds, bins=100) # get the histogram of the i-th feature for all the events and all the jets
        im = ax[i].imshow(h.T, cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = h.max()), extent= [*bounds[0], *bounds[1]], aspect = 'auto')
        '''
        
        if i == 0:
            if j == 1:
                if phi_rot:
                    h2d, xbins, ybins, im = ax[i, j].hist2d(true_val[:, j, 1:4].flatten().numpy(), res[:, j, 1:4].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
                else:
                    h2d, xbins, ybins, im = ax[i, j].hist2d(true_val[:, j, :].flatten().numpy(), res[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
            else:
                h2d, xbins, ybins, im = ax[i, j].hist2d(true_val[:, j, :].flatten().numpy(), res[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
        elif i == 1:
            h2d, xbins, ybins, im = ax[1, 0].hist2d(true_m2j[:, :, :].flatten().numpy(), res_m2j[:, :, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
            h2d, xbins, ybins, im = ax[1, 1].hist2d(true_m4j[:, :].flatten().numpy(), res_m4j[:, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50)) 
            h2d, xbins, ybins, im = ax[1, 2].hist2d(true_pt.flatten().numpy(), res_pt.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))

        ax[i, j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[i, j].minorticks_on()
        ax[i, j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)

        ax[i, j].set_xlabel(f'True {feature}')
        ax[i, j].set_ylabel(f'Reco - true {feature}')

        #ax[i].plot(xbins, xbins, lw = 2., c = 'grey', ls = '-.')
        ax[i, j].axhline(y = 0, lw = 2., c = 'grey', ls = '-.')
        
        if h2d.max() > vmax_mob:
            im_vmax = im
            vmax_mob = h2d.max()
        
        j+=1

        
    
    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_residuals_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Residuals saved to {path}')
    plt.close()


def plot_training_residuals_PtEtaPhiEm2jm4j(true_val, reco_val, offset, epoch, sample, network_name): # expects [batch, (3) features, (4) jets] shaped tensors
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d
    plot_masses = True
    if plot_masses:
        d_rot, dPxPyPzE_rot = networks.addFourVectors(  true_val[:,:,(0,2,0,1,0,1)], 
                                                        true_val[:,:,(1,3,2,3,3,2)])
        q_rot, qPxPyPzE_rot = networks.addFourVectors(  d_rot[:,:,(0,2,4)],
                                                        d_rot[:,:,(1,3,5)], 
                                                        v1PxPyPzE = dPxPyPzE_rot[:,:,(0,2,4)],
                                                        v2PxPyPzE = dPxPyPzE_rot[:,:,(1,3,5)])
        m2j = d_rot[:, 3:4, :]
        m4j = q_rot[:, 3:4, 0]

        rec_d, rec_dPxPyPzE = networks.addFourVectors(  reco_val[:,:,(0,2,0,1,0,1)], 
                                                        reco_val[:,:,(1,3,2,3,3,2)])
        rec_q, rec_qPxPyPzE = networks.addFourVectors(  rec_d[:,:,(0,2,4)],
                                                        rec_d[:,:,(1,3,5)])
        rec_m2j = rec_d[:, 3:4, :]
        rec_m4j = rec_q[:, 3:4, 0]

        true_m2j = m2j.detach()
        reco_m2j = rec_m2j.detach()
        res_m2j = reco_m2j - true_m2j

        true_m4j = m4j.detach()
        reco_m4j = rec_m4j.detach()
        res_m4j = reco_m4j - true_m4j


    true_val = true_val.detach()
    reco_val = reco_val.detach()
    res = reco_val - true_val
    res_norm = res / true_val

    

    

    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(2, 4, figsize = (15, 5))
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0
    i, j = 0, 0
    for feature in ["$p_{T}\ ({\\rm GeV)}$", "$\eta$", "$\phi$", "$m_{j}\ ({\\rm GeV)}$", "$m_{2j}\ ({\\rm GeV)}$", "$m_{4j}\ ({\\rm GeV)}$"]:
        if j > 3:
            i += 1
            j -= 4

        '''
        Implementation of fast histogram is weird: histogram2d produces a 2d plot that makes NO sense in the confrontation of y vs x (the correlation is lost somehow)
        bounds = [(true_val[:, i, :].min(), true_val[:, i, :].max()), (reco_val[:, i, :].min(), reco_val[:, i, :].max())]
        h = histogram2d(true_val[:, i, :], reco_val[:, i, :], range=bounds, bins=100) # get the histogram of the i-th feature for all the events and all the jets
        im = ax[i].imshow(h.T, cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = h.max()), extent= [*bounds[0], *bounds[1]], aspect = 'auto')
        '''
        
        if i ==0:
            h2d, xbins, ybins, im = ax[i, j].hist2d(true_val[:, j, :].flatten().numpy(), res[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
        elif i == 1 and plot_masses:
            h2d, xbins, ybins, im = ax[1, 0].hist2d(true_m2j[:, :, :].flatten().numpy(), res_m2j[:, :, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
            h2d, xbins, ybins, im = ax[1, 1].hist2d(true_m4j[:, :].flatten().numpy(), res_m4j[:, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))

        ax[i, j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[i, j].minorticks_on()
        ax[i, j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)

        ax[i, j].set_xlabel(f'True {feature}')
        ax[i, j].set_ylabel(f'Reco - true {feature}')

        #ax[i].plot(xbins, xbins, lw = 2., c = 'grey', ls = '-.')
        ax[i, j].axhline(y = 0, lw = 2., c = 'grey', ls = '-.')
        
        if h2d.max() > vmax_mob:
            im_vmax = im
            vmax_mob = h2d.max()
        
        j+=1

        
    
    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/VAE/residualsPtEtaPhi_notfms/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_residuals_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Residuals saved to {path}')
    plt.close()


def plot_training_residuals_VAE(true_val, reco_val, offset, epoch, sample, network_name): # expects [batch, (3) features, (4) jets] shaped tensors
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    true_val = true_val.detach()
    reco_val = reco_val.detach()
    res = reco_val - true_val
    res_norm = res / true_val
    '''
    true_m2j = m2j.detach()
    reco_m2j = rec_m2j.detach()
    res_m2j = reco_m2j - true_m2j

    true_m4j = m4j.detach()
    reco_m4j = rec_m4j.detach()
    res_m4j = reco_m4j - true_m4j
    '''

    true_pt = ((true_val[:, 0:1, :]**2 + true_val[:, 1:2, :]**2).sqrt()).detach()
    reco_pt = ((reco_val[:, 0:1, :]**2 + reco_val[:, 1:2, :]**2).sqrt()).detach()
    res_pt = true_pt - reco_pt

    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(2, 4, figsize = (15, 5))
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0
    i, j = 0, 0
    for feature in ["$p_{T}\ ({\\rm GeV)}$", "$\eta$", "$\phi$"]:
        if j > 3:
            i += 1
            j -= 4

        '''
        Implementation of fast histogram is weird: histogram2d produces a 2d plot that makes NO sense in the confrontation of y vs x (the correlation is lost somehow)
        bounds = [(true_val[:, i, :].min(), true_val[:, i, :].max()), (reco_val[:, i, :].min(), reco_val[:, i, :].max())]
        h = histogram2d(true_val[:, i, :], reco_val[:, i, :], range=bounds, bins=100) # get the histogram of the i-th feature for all the events and all the jets
        im = ax[i].imshow(h.T, cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = h.max()), extent= [*bounds[0], *bounds[1]], aspect = 'auto')
        '''
        
        if i ==0:
            h2d, xbins, ybins, im = ax[i, j].hist2d(true_val[:, j, :].flatten().numpy(), res[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))

        ax[i, j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[i, j].minorticks_on()
        ax[i, j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)

        ax[i, j].set_xlabel(f'True {feature}')
        ax[i, j].set_ylabel(f'Reco - true {feature}')

        #ax[i].plot(xbins, xbins, lw = 2., c = 'grey', ls = '-.')
        ax[i, j].axhline(y = 0, lw = 2., c = 'grey', ls = '-.')
        
        if h2d.max() > vmax_mob:
            im_vmax = im
            vmax_mob = h2d.max()
        
        j+=1

    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/VAE/residualsPtEtaPhi_notfms/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_residuals_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Residuals saved to {path}')
    plt.close()


def plot_lossVAE(loss, offset, epoch, sample, network_name):
    fig, ax = plt.subplots(figsize = (15, 5))
    ax.set_yscale("log")
    ax.plot(loss["train"]["total"], color = "r", label = "Train loss")
    ax.plot(loss["train"]["kl"], color = "r", linestyle = "dashed", label = "Train kl loss")
    ax.plot(loss["train"]["reco"], color = "r", linestyle = "dashdot", label = "Train reco loss")
    ax.plot(loss["val"]["total"], color = "b", label = "Val loss")
    ax.plot(loss["val"]["kl"], color = "b", linestyle = "dashed", label = "Val kl loss")
    ax.plot(loss["val"]["reco"], color = "b", linestyle = "dashdot", label = "Val reco loss")

    ax.legend(loc = "best")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(np.arange(0, len(loss["train"]["total"]) + 1, len(loss["train"]["total"]) // 20)) if len(loss["train"]["total"]) >= 20 else ax.set_xticks(np.arange(0, len(loss["train"]["total"]) + 1, 2))
    fig.tight_layout()
    path = f"plots/VAE/residualsPtEtaPhi_notfms/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_loss_{network_name}_offset_{offset}_{epoch:03d}epochs.pdf')
    print(f'Losses saved to {path}')
    plt.close()


def plot_loss(loss, offset, epoch, sample, network_name):
    fig, ax = plt.subplots(figsize = (15, 5))
    ax.set_yscale("log")
    ax.plot(loss["train"], color = "r", label = "Train loss")
    ax.plot(loss["val"], color = "b", label = "Val loss")

    ax.legend(loc = "best")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (GeV)')
    ax.set_xticks(np.arange(0, len(loss["train"]) + 1, len(loss["train"]) // 20)) if len(loss["train"]) >= 20 else ax.set_xticks(np.arange(0, len(loss["train"]) + 1, 2))
    fig.tight_layout()
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_loss_{network_name}_offset_{offset}_{epoch:03d}epochs.pdf')
    print(f'Losses saved to {path}')
    plt.close()

def plot_PxPyPzE(true_val, reco_val, phi_rot, offset, epoch, sample, network_name):
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    true_pt = ((true_val[:, 0:1, :]**2 + true_val[:, 1:2, :]**2).sqrt()).detach()
    reco_pt = ((reco_val[:, 0:1, :]**2 + reco_val[:, 1:2, :]**2).sqrt()).detach()
    res_pt = true_pt - reco_pt

    true_val = true_val.detach()
    reco_val = reco_val.detach()

    width = 10 # GeV
    fig, ax = plt.subplots(1, 4, figsize = (15, 5))
    for j, feature in enumerate(["$p_{x}\ ({\\rm GeV)}$", "$p_{y}\ ({\\rm GeV)}$", "$p_{z}\ ({\\rm GeV)}$", "$E\ ({\\rm GeV)}$"]):
        if j == 1:
            if phi_rot:
                nbins = int(round(max(true_val[:, j, 1:4].flatten().numpy()) - min(true_val[:, j, 1:4].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
                h, bins1, _ = ax[j].hist(true_val[:, j, 1:4].flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
                nbins = int(round(max(reco_val[:, j, 1:4].flatten().numpy()) - min(reco_val[:, j, 1:4].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
                ax[j].hist(reco_val[:, j, 1:4].flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)
            else:
                nbins = int(round(max(true_val[:, j, :].flatten().numpy()) - min(true_val[:, j, :].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
                h, bins1, _ = ax[j].hist(true_val[:, j, :].flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
                nbins = int(round(max(reco_val[:, j, :].flatten().numpy()) - min(reco_val[:, j, :].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
                ax[j].hist(reco_val[:, j, :].flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)
        else:
            nbins = int(round(max(true_val[:, j, :].flatten().numpy()) - min(true_val[:, j, :].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_val[:, j, :].flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
            nbins = int(round(max(reco_val[:, j, :].flatten().numpy()) - min(reco_val[:, j, :].flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            ax[j].hist(reco_val[:, j, :].flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)
        
        ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV')
        
        ax[j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[j].minorticks_on()
        ax[j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)

        ax[j].set_xlabel(f'{feature}')
        
    
    #ax[0].hist(true_val[:, 0, :].flatten().numpy() - true_val[:, 1, :].flatten().numpy(), color = "darkorange", label = "px - py", histtype = "step", bins = nbins)
    #ax[0].hist(true_val[:, 0, :].flatten().numpy() + true_val[:, 1, :].flatten().numpy(), color = "forestgreen", label = "px + py", histtype = "step", bins = nbins)
    
    ax[2].set_yscale("log")
    ax[3].set_yscale("log")
    ax[0].set_xlim(-250, 250)
    ax[1].set_xlim(-250, 250)
    #ax[3].set_xlim(-100, 1000)
    ax[0].legend(loc = "best")
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.3, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_PxPyPzPt_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'PxPyPz saved to {path}')
    plt.close()

def plot_PtEtaPhiE(true_val, reco_val, theta, rec_theta, logpt, rec_logpt, offset, epoch, sample, network_name):
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    #true_E = (networks.PxPyPzE(true_val)[:, 3:4, :]).detach()
    #reco_E = (networks.PxPyPzE(reco_val)[:, 3:4, :]).detach()

    true_val = true_val.detach()
    reco_val = reco_val.detach()
    
    true_theta = theta.detach()
    reco_theta = rec_theta.detach()

    true_logpt = logpt.detach()
    reco_logpt = rec_logpt.detach()

    
    fig, ax = plt.subplots(1, 7, figsize = (15, 5))
    for j, feature in enumerate(["$p_{T}\ ({\\rm GeV)}$", "$\eta$", "$\phi$", "$\\theta$", "$\log{(p_{T}\ [{\\rm GeV}])}$"]):
        if j == 3:
            width = 0.25 # GeV
            nbins = int(round(max(true_theta.flatten().numpy()) - min(true_theta.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_theta.flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
            ax[j].hist(reco_theta.flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)
            ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
            ax[j].set_yscale("log")
        elif j == 4:
            width = 0.25 # GeV
            nbins = int(round(max(true_logpt.flatten().numpy()) - min(true_logpt.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_logpt.flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
            nbins = int(round(max(reco_logpt.flatten().numpy()) - min(reco_logpt.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            ax[j].hist(reco_logpt.flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)
            ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
            ax[j].set_yscale("log")
        else:
            width = 20 if j == 0 else 0.25
            nbins = int(round(max(true_val[:, j, :].flatten().numpy()) - min(true_val[:, j, :].flatten().numpy())) /  width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_val[:, j, :].flatten().numpy(), color = "firebrick", label = "true", histtype = "step", bins = nbins)
            nbins = int(round(max(reco_val[:, j, :].flatten().numpy()) - min(reco_val[:, j, :].flatten().numpy())) /  width) + 1 if max(reco_val[:, j, :].flatten().numpy()) < 1e4 else 8192
            ax[j].hist(reco_val[:, j, :].flatten().numpy(), color = "blue", label = "reco", histtype = "step", bins = bins1)

            ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV') if j == 0 else ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
            
        
        ax[j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
        ax[j].minorticks_on()
        ax[j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)
        ax[j].set_xlabel(f'{feature}')

    ax[0].set_yscale('log')
    ax[0].legend(loc = "best")
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/VAE/residualsPtEtaPhi_notfms/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_PtEtaPhiE_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'PtEtaPhi saved to {path}')
    plt.close()

def plot_loss_distr(j, loss_distr, loss_weights, offset, epoch, sample, network_name):
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    loss_weights = loss_weights.detach()
    loss_distr = loss_distr.detach()
    true_j = j.detach()



    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(2, 3, figsize = (15, 5))
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0

    for j, feature in enumerate(["$p_{T}\ ({\\rm GeV)}$", "$\eta$", "$\phi$"]):


        '''
        Implementation of fast histogram is weird: histogram2d produces a 2d plot that makes NO sense in the confrontation of y vs x (the correlation is lost somehow)
        bounds = [(true_val[:, i, :].min(), true_val[:, i, :].max()), (reco_val[:, i, :].min(), reco_val[:, i, :].max())]
        h = histogram2d(true_val[:, i, :], reco_val[:, i, :], range=bounds, bins=100) # get the histogram of the i-th feature for all the events and all the jets
        im = ax[i].imshow(h.T, cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = h.max()), extent= [*bounds[0], *bounds[1]], aspect = 'auto')
        '''

        h2d, xbins, ybins, im = ax[0, j].hist2d(true_j[:, j, :].flatten().numpy(), loss_distr[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))
        
        if j == 0 or j == 1:
            h2d, xbins, ybins, im = ax[1, j].hist2d(true_j[:, j, :].flatten().numpy(), loss_weights[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 2000), bins = (50, 50))

        for i in range(2):
            ax[i, j].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
            ax[i, j].minorticks_on()
            ax[i, j].tick_params(which = 'minor', axis = 'both', direction='in', length = 0)
            ax[i, j].set_xlabel(f'True {feature}')
            
        ax[0, j].set_ylabel(f'Loss for {feature}')
        ax[1, j].set_ylabel(f'$\omega_L$ for {feature}')

        #ax[i].plot(xbins, xbins, lw = 2., c = 'grey', ls = '-.')
        ax[i, j].axhline(y = 0, lw = 2., c = 'grey', ls = '-.')
        
        if h2d.max() > vmax_mob:
            im_vmax = im
            vmax_mob = h2d.max()
        
        j+=1

    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/VAE/residualsPtEtaPhi_notfms/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_lossdistr_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Loss distribution saved to {path}')
    plt.close()

def plot_etaPhi_plane(true_val, reco_val, offset, epoch, sample, network_name): # expects [batch, (3) features, (4) jets] shaped tensors
    import matplotlib
    #matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    #from fast_histogram import histogram2d

    true_j = networks.PtEtaPhiM(true_val)
    rec_j = networks.PtEtaPhiM(reco_val)
    pos = int(round(np.random.uniform()*true_j.shape[0]))

    pt_true = true_j[pos, 0, :].detach().flatten().numpy()
    pt_rec = rec_j[pos, 0, :].detach().flatten().numpy()

    eta_true = true_j[pos, 1, :].detach().flatten().numpy()
    eta_rec = rec_j[pos, 1, :].detach().flatten().numpy()
    phi_true = true_j[pos, 2, :].detach().flatten().numpy()
    phi_rec = rec_j[pos, 2, :].detach().flatten().numpy()
    
    coords = [eta_true, phi_true, eta_rec, phi_rec]         
    #cmap = cm.get_cmap("bwr")
    cmap = cm.get_cmap("viridis")
    #cc.cm["CET_L17"].copy()
    
    fig, ax = plt.subplots(1, figsize = (10, 10))
    for j in range(4):
        ax.plot((eta_true[j], eta_rec[j]), (phi_true[j], phi_rec[j]), lw = 2, ls = 'dashed', color = 'grey')
    ax.scatter(eta_true, phi_true, s=pt_true, color = 'red', label = f'True {pos} event')
    ax.scatter(eta_rec, phi_rec, s=pt_rec, color = 'blue', label = f'Reco {pos} event')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-np.pi, np.pi)
    ax.tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 10)
    ax.minorticks_on()
    ax.tick_params(which = 'minor', axis = 'both', direction='in', length = 0)
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$\phi$')

    #fig.subplots_adjust(top = 0.9, bottom=0.1, left = 0.06, right=0.94, wspace=0.4, hspace = 0.4)
    fig.tight_layout()
    fig.suptitle(f'Epoch {epoch}')
    ax.legend(loc='best')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_etaPhiplane_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'EtaPhiplane saved to {path}')
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
