import pickle, os, argparse
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networks
import matplotlib.cm as cm


# style
def update_rcparams():
    plt.rcParams.update({
        "figure.figsize": [40, 20],
        "font.weight": "bold",
        'figure.titlesize': 50,
        'axes.titlesize': 50, # changes the axes titles (figure title when you have only one)
        'figure.titleweight': 'bold',
        "text.usetex": True,
        "font.family": "serif",
        
        'legend.fontsize': 30,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        'xtick.major.width': 1.,
        "ytick.major.size": 8,
        'ytick.major.width': 1.,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 4,
        'xtick.minor.width': 0.8,
        "ytick.minor.size": 4,
        'ytick.minor.width': 0.8,
        'axes.labelpad': 5.0,
        #"xtick.major.pad": 7,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "font.size": 40, # affects axes title size
        "grid.color": "0.5",
        #"grid.linestyle": "-",
        #"grid.linewidth": 5,
        "lines.linewidth": 5,
        #"lines.color": "g",
        })
    
# some colors
mainblue = ["#42719f", "#6ab4ff"]
coralgreen = ["#117a65", "#138D75"]
reddish = ["#c82929", "#e42f2f"]
orangish = ["#d78939", "#e48a2f"]
blueish = ["#4387cb", "#2f89e4"]

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




########

# upscale a bit the size of the markers on the plots
EtaPhi_zscale = 2
PxPy_zscale = 2

def plot_training_residuals_PxPyPzEm2jm4jPtm2jvsm4j(jPxPyPzE, rec_jPxPyPzE, phi_rot, m2j=None, m4j=None, rec_m2j=None, rec_m4j=None, **kwargs):
    
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    # PxPyPzE
    jPxPyPzE = jPxPyPzE.detach()
    rec_jPxPyPzE = rec_jPxPyPzE.detach()
    res = rec_jPxPyPzE - jPxPyPzE
    
    # Pt
    pt = ((jPxPyPzE[:, 0:1, :]**2 + jPxPyPzE[:, 1:2, :]**2).sqrt()).detach()
    rec_pt = ((rec_jPxPyPzE[:, 0:1, :]**2 + rec_jPxPyPzE[:, 1:2, :]**2).sqrt()).detach()
    res_pt = rec_pt - pt
    
    # m2j, m4j
    plot_masses = False
    if m2j is not None and m4j is not None and rec_m2j is not None and rec_m4j is not None:
        plot_masses = True
        m2j = m2j.detach()
        rec_m2j = rec_m2j.detach()
        res_m2j = rec_m2j - m2j

        m4j = m4j.detach()
        rec_m4j = rec_m4j.detach()
        res_m4j = rec_m4j - m4j


    # plots
    fig, ax = plt.subplots(3, 3)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.84])
    cmap = cm.get_cmap("viridis") # cmap = cm.get_cmap("bwr")

    h2d_list = []
    im_list = []
    # plot PxPyPz on first row and E on second row first element
    for j in range(4):
        if j < 3:
            h2d, xbins, ybins, im = ax[0, j].hist2d(jPxPyPzE[:, j, 1:4].flatten().numpy(), res[:, j, 1:4].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50)) if phi_rot else ax[0, j].hist2d(jPxPyPzE[:, j, :].flatten().numpy(), res[:, j, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
            h2d_list.append(h2d); im_list.append(im)
        else:
            h2d, xbins, ybins, im = ax[1, 0].hist2d(jPxPyPzE[:, 3, :].flatten().numpy(), res[:, 3, :].flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
            h2d_list.append(h2d); im_list.append(im)

    # plot Pt
    h2d, xbins, ybins, im = ax[1, 1].hist2d(pt.flatten().numpy(), res_pt.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
    h2d_list.append(h2d); im_list.append(im)

    if plot_masses:
        # plot m2j
        h2d, xbins, ybins, im = ax[1, 2].hist2d(m2j.flatten().numpy(), res_m2j.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
        h2d_list.append(h2d); im_list.append(im)

        # plot m4j
        h2d, xbins, ybins, im = ax[2, 0].hist2d(m4j.flatten().numpy(), res_m4j.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50)) 
        h2d_list.append(h2d); im_list.append(im)

        # plot m2j vs m4j
        h2d, xbins, ybins, im = ax[2, 1].hist2d(np.repeat(m4j.flatten().numpy(), 2), m2j.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
        h2d_list.append(h2d); im_list.append(im)

        # plot rec_m2j vs rec_m4j
        h2d, xbins, ybins, im = ax[2, 2].hist2d(np.repeat(rec_m4j.flatten().numpy(), 2), rec_m2j.flatten().numpy(), cmap=cmap, norm = matplotlib.colors.LogNorm(vmax = 20000), bins = (50, 50))
        h2d_list.append(h2d); im_list.append(im)


    for i in range(3):
        for j in range(3):
            if i < 2:
                ax[i, j].axhline(y = 0, lw = 2., c = 'grey', ls = '-.') # ._ grey line for residuals plots (except the last 2)
            
            
    ax[2, 0].axhline(y = 0, lw = 2., c = 'grey', ls = '-.') # add the ._ grey line on the third to last plot which is residuals


    # axis labels
    ax[0, 0].set_xlabel('True $p_{x}\ ({\mathrm{GeV}})$');    ax[0, 0].set_ylabel('Reco - true $p_{x}\ ({\mathrm{GeV}})$')
    ax[0, 1].set_xlabel('True $p_{y}\ ({\mathrm{GeV}})$');    ax[0, 1].set_ylabel('Reco - true $p_{y}\ ({\mathrm{GeV}})$')
    ax[0, 2].set_xlabel('True $p_{z}\ ({\mathrm{GeV}})$');    ax[0, 2].set_ylabel('Reco - true $p_{z}\ ({\mathrm{GeV}})$')
    ax[1, 0].set_xlabel('True $E\ ({\\rm GeV)}$');        ax[1, 0].set_ylabel('Reco - true $E\ ({\\rm GeV)}$')
    ax[1, 1].set_xlabel('True $p_{T}\ ({\mathrm{GeV}})$');    ax[1, 1].set_ylabel('Reco - true $p_{T}\ ({\mathrm{GeV}})$')
    ax[1, 2].set_xlabel('True $m_{2j}\ ({\mathrm{GeV}})$');   ax[1, 2].set_ylabel('Reco - true $m_{2j}\ ({\mathrm{GeV}})$')
    ax[2, 0].set_xlabel('True $m_{4j}\ ({\mathrm{GeV}})$');   ax[2, 0].set_ylabel('Reco - true $m_{4j}\ ({\mathrm{GeV}})$')
    ax[2, 1].set_xlabel('True $m_{4j}\ ({\mathrm{GeV}})$')
    ax[2, 1].set_ylabel('True $m_{2j}\ ({\mathrm{GeV}})$')
    ax[2, 2].set_xlabel('Reco $m_{4j}\ ({\mathrm{GeV}})$')
    ax[2, 2].set_ylabel('Reco $m_{2j}\ ({\mathrm{GeV}})$')
    

    # get the im that contains the h2d with the highest value and make the colorbar map that im
    h2d_np = np.array(h2d_list)
    im_np = np.array(im_list)
    position_of_maximum = np.unravel_index(np.argmax(h2d_np), h2d_np.shape)
    #vmax = h2d_np[position_of_maximum]
    im_vmax = im_np[position_of_maximum[0]]

        
    fig.colorbar(im_vmax, cax=cbar_ax)
    fig.subplots_adjust(left=0.07 , bottom=0.1, right=0.93, top=0.94, wspace=0.3, hspace = 0.5)
    fig.suptitle(f'Epoch {epoch}')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_residuals_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Residuals saved to {path}')
    plt.close()

def plot_PxPyPzEPtm2jm4j(jPxPyPzE, rec_jPxPyPzE, phi_rot, m2j=None, m4j=None, rec_m2j=None, rec_m4j=None, **kwargs):

    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    # PxPyPzE
    jPxPyPzE = jPxPyPzE.detach()
    rec_jPxPyPzE = rec_jPxPyPzE.detach()
    res = rec_jPxPyPzE - jPxPyPzE
    
    # Pt
    pt = ((jPxPyPzE[:, 0:1, :]**2 + jPxPyPzE[:, 1:2, :]**2).sqrt()).detach()
    rec_pt = ((rec_jPxPyPzE[:, 0:1, :]**2 + rec_jPxPyPzE[:, 1:2, :]**2).sqrt()).detach()
    res_pt = rec_pt - pt

    # deltaR
    DeltaR = networks.calcDeltaR(       networks.PtEtaPhiM(jPxPyPzE)[:,:,(0,2,0,1,0,1)],
                                        networks.PtEtaPhiM(jPxPyPzE)[:,:,(1, 3,2,3,3,2)])        # obtain the DeltaR between the 6 combinations of jets: this DeltaR should never be < 0.4 (in reality is hehe)
    rec_DeltaR = networks.calcDeltaR(   networks.PtEtaPhiM(rec_jPxPyPzE)[:,:,(0,2,0,1,0,1)],
                                        networks.PtEtaPhiM(rec_jPxPyPzE)[:,:,(1, 3,2,3,3,2)])    # obtain the DeltaR between the 6 combinations of jets: this DeltaR should never be < 0.4
    
    # m2j, m4j
    plot_masses = False
    if m2j is not None and m4j is not None and rec_m2j is not None and rec_m4j is not None:
        plot_masses = True
        m2j = m2j.detach()
        rec_m2j = rec_m2j.detach()
        res_m2j = rec_m2j - m2j

        m4j = m4j.detach()
        rec_m4j = rec_m4j.detach()
        res_m4j = rec_m4j - m4j


    # plots
    fig, ax = plt.subplots(2, 4)

    # plot PxPyPzE on first row
    width = 10 # GeV
    for j, feature in enumerate(["$p_{x}\ ({\mathrm{GeV}})$", "$p_{y}\ ({\mathrm{GeV}})$", "$p_{z}\ ({\mathrm{GeV}})$", "$E\ ({\\rm GeV)}$"]):
        if j == 3: width = 50
        nbins = int(round(max(jPxPyPzE[:, j, :].flatten().numpy()) - min(jPxPyPzE[:, j, :].flatten().numpy())) / width) + 1
        h, bins1, _ = ax[0, j].hist(jPxPyPzE[:, j, :].flatten().numpy(), color = "firebrick", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
        nbins = int(round(max(rec_jPxPyPzE[:, j, :].flatten().numpy()) - min(rec_jPxPyPzE[:, j, :].flatten().numpy())) / width) + 1
        ax[0, j].hist(rec_jPxPyPzE[:, j, :].flatten().numpy(), color = "blue", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
        ax[0, j].set_xlabel(feature)
        ax[0, j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV')
        ax[0, j].set_yscale('log')

        if j < 2:
            ax[0, j].set_xlim(-500, 500) # PxPy
            
        

    # plot Pt
    width = 10
    nbins = int(round(max(pt.flatten().numpy()) - min(pt.flatten().numpy())) / width) + 1
    h, bins1, _ = ax[1, 0].hist(pt.flatten().numpy(), color = "firebrick", label = "True", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
    nbins = int(round(max(rec_pt.flatten().numpy()) - min(rec_pt.flatten().numpy())) / width) + 1
    ax[1, 0].hist(rec_pt.flatten().numpy(), color = "blue", label = "Reco", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
    ax[1, 0].set_xlabel('$p_{T}\ ({\mathrm{GeV}})$')
    ax[1, 0].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV')
    ax[1, 0].set_yscale('log')

    if plot_masses:
        # plot m2j
        width = 10
        nbins = int(round(max(m2j.flatten().numpy()) - min(m2j.flatten().numpy())) / width) + 1
        h, bins1, _ = ax[1, 1].hist(m2j.flatten().numpy(), color = "firebrick", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
        nbins = int(round(max(rec_m2j.flatten().numpy()) - min(rec_m2j.flatten().numpy())) / width) + 1
        ax[1, 1].hist(rec_m2j.flatten().numpy(), color = "blue", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
        ax[1, 1].set_xlabel('$m_{2j}\ ({\mathrm{GeV}})$')
        ax[1, 1].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV')
        ax[1, 1].set_xlim(-100, 2500)
        ax[1, 1].set_yscale('log')

        # plot m4j
        width = 10
        nbins = int(round(max(m4j.flatten().numpy()) - min(m4j.flatten().numpy())) / width) + 1
        h, bins1, _ = ax[1, 2].hist(m4j.flatten().numpy(), color = "firebrick", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
        nbins = int(round(max(rec_m4j.flatten().numpy()) - min(rec_m4j.flatten().numpy())) / width) + 1
        ax[1, 2].hist(rec_m4j.flatten().numpy(), color = "blue", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
        ax[1, 2].set_xlabel('$m_{4j}\ ({\mathrm{GeV}})$')
        ax[1, 2].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f} GeV')
        ax[1, 2].set_xlim(-100, 2500)
        ax[1, 2].set_yscale('log')

    # plot deltaR
    width = 0.1
    nbins = int(round(max(DeltaR.flatten().numpy()) - min(DeltaR.flatten().numpy())) / width) + 1
    h, bins1, _ = ax[1, 3].hist(DeltaR.flatten().numpy(), color = "firebrick", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
    nbins = int(round(max(rec_DeltaR.flatten().numpy()) - min(rec_DeltaR.flatten().numpy())) / width) + 1
    ax[1, 3].hist(rec_DeltaR.flatten().numpy(), color = "blue", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
    ax[1, 3].set_xlabel('$\Delta R$')
    ax[1, 3].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
    
    # layout and save
    fig.suptitle(f'Epoch {epoch}')
    fig.legend(loc = 'upper right', bbox_to_anchor = (0.97, 0.9))
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.94, wspace=0.35, hspace = 0.3)
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_PxPyPzPt_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'PxPyPz saved to {path}')
    plt.close()

def plot_etaPhi_plane(jPxPyPzE, rec_jPxPyPzE, **kwargs):
    
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    # get the PtEtaPhiM representation of j
    j = networks.PtEtaPhiM(jPxPyPzE)
    rec_j = networks.PtEtaPhiM(rec_jPxPyPzE)
    event_number = int(round(np.random.uniform()*j.shape[0])) # get the event number to plot the same for PxPyplane

    # Pt
    pt_true = j[event_number, 0, :].detach().flatten().numpy()
    pt_rec = rec_j[event_number, 0, :].detach().flatten().numpy()
    # Eta
    eta_true = j[event_number, 1, :].detach().flatten().numpy()
    eta_rec = rec_j[event_number, 1, :].detach().flatten().numpy()
    # Phi
    phi_true = j[event_number, 2, :].detach().flatten().numpy()
    phi_rec = rec_j[event_number, 2, :].detach().flatten().numpy()
    
    # plots
    fig, ax = plt.subplots(1, figsize=(20, 10))
    for j in range(4):
        ax.plot((eta_true[j], eta_rec[j]), (phi_true[j], phi_rec[j]), ls = 'dashed', color = 'grey', zorder = 1)
    scatter_true = ax.scatter(eta_true, phi_true, s=EtaPhi_zscale * pt_true, color=reddish[0], label=f'True event {event_number}', zorder=2)
    scatter_reco = ax.scatter(eta_rec, phi_rec, s=EtaPhi_zscale * pt_rec, color=mainblue[0], label=f'Reco event {event_number}', zorder=2)
    ax.legend(handles=[scatter_true, scatter_reco], loc='best')

    # Create the second legend for pt sizes
    legend_sizes = [125, 250, 500, 0]
    pt_legend_handles = []
    for size in legend_sizes:
        handle = ax.scatter([], [], s=size * EtaPhi_zscale, fc='none', ec="black")
        pt_legend_handles.append(handle)

    # Add the markersize legend as a separate artist
    fig.add_artist(fig.legend(handles=pt_legend_handles, labels=[f'{size}' for size in legend_sizes[:-1]]+ ['$p_{T}\, ({\mathrm{GeV}})$'], ncol=len(legend_sizes), bbox_to_anchor=(1, 1.005)))

    # format
    ax.set_xlim(-3, 3)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('$\eta$', fontsize = plt.rcParams["font.size"]*0.8)
    ax.set_ylabel('$\phi$', fontsize = plt.rcParams["font.size"]*0.8)
    ax.tick_params(which = 'major', axis = 'both', direction=plt.rcParams["xtick.direction"], length=plt.rcParams["xtick.major.size"], labelsize=plt.rcParams["font.size"]*0.8)
    
    # layout and plot
    fig.tight_layout() 
    fig.subplots_adjust(top = 0.875, bottom=0.15, left = 0.09, right=0.92, wspace=0.3, hspace = 0.4)
    ax.legend(loc='best')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_etaPhiplane_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'EtaPhiplane saved to {path}')
    plt.close()
    return event_number

def plot_PxPy_plane(jPxPyPzE, rec_jPxPyPzE, event_number = 0, **kwargs):
    
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    # Px
    px_true = jPxPyPzE[event_number, 0, :].detach().flatten().numpy()
    px_rec = rec_jPxPyPzE[event_number, 0, :].detach().flatten().numpy()
    # Py
    py_true = jPxPyPzE[event_number, 1, :].detach().flatten().numpy()
    py_rec = rec_jPxPyPzE[event_number, 1, :].detach().flatten().numpy()    
    # Pz
    # pz_true = jPxPyPzE[event_number, 2, :].detach().flatten().numpy()
    # pz_rec = rec_jPxPyPzE[event_number, 2, :].detach().flatten().numpy()
    # E
    E_true = jPxPyPzE[event_number, 3, :].detach().flatten().numpy()
    E_rec = rec_jPxPyPzE[event_number, 3, :].detach().flatten().numpy()
    
    # plots
    fig, ax = plt.subplots(1, figsize = (20, 10))
    for j in range(4):
        ax.plot((px_true[j], px_rec[j]), (py_true[j], py_rec[j]), ls = 'dashed', color = 'grey', zorder = 1)
    scatter_true = ax.scatter(px_true, py_true, s=PxPy_zscale * E_true, color = reddish[0], label =f'True event {event_number}', zorder = 2)
    scatter_reco = ax.scatter(px_rec, py_rec, s=PxPy_zscale*E_rec, color = mainblue[0], label =f'Reco event {event_number}', zorder = 2)
    ax.legend(handles=[scatter_true, scatter_reco], loc='best')

    # Create the second legend for pt sizes
    legend_sizes = [100, 500, 1000, 0]
    E_legend_handles = []
    for size in legend_sizes:
        handle = ax.scatter([], [], s=size * PxPy_zscale, fc='none', ec="black")
        E_legend_handles.append(handle)

    # Add the markersize legend as a separate artist
    fig.add_artist(fig.legend(handles=E_legend_handles, labels=[f'{size}' for size in legend_sizes[:-1]]+ ['$E\, ({\mathrm{GeV}})$'], ncol=len(legend_sizes), bbox_to_anchor=(1, 1.005)))

    # format
    ax.set_xlabel('$p_{x}\ ({\mathrm{GeV}})$', fontsize = plt.rcParams["font.size"]*0.8)
    ax.set_ylabel('$p_{y}\ ({\mathrm{GeV}})$', fontsize = plt.rcParams["font.size"]*0.8)
    ax.tick_params(which = 'major', axis = 'both', direction=plt.rcParams["xtick.direction"], length=plt.rcParams["xtick.major.size"], labelsize=plt.rcParams["font.size"]*0.8)
    
    # layout and plot
    fig.tight_layout() 
    fig.subplots_adjust(top = 0.875, bottom=0.15, left = 0.09, right=0.92, wspace=0.3, hspace = 0.4)
    ax.legend(loc='best')
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_PxPyplane_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'PxPyplane saved to {path}')
    plt.close()

def plot_activations_embedded_space(z, **kwargs):
    
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')
    
    z = z.numpy()
    # Get the number of features
    dimension = z.shape[1]

    # Determine the grid layout based on the number of features
    if dimension <= 4:
        num_rows = 2
        num_cols = 2
    elif dimension <= 6:
        num_rows = 2
        num_cols = 3
    elif dimension <= 9:
        num_rows = 3
        num_cols = 3 
    elif dimension <= 12:
        num_rows = 3
        num_cols = 4
    elif dimension <= 16:
        num_rows = 4
        num_cols = 4
    else:
        raise ValueError("The number of features is too high to display in a reasonable way.")

    # plots
    fig, axs = plt.subplots(num_rows, num_cols)
    #fig.suptitle(f"Histograms of {dimension} Different Features", fontsize=16)
    axs = axs.flatten()
    if (dimension < num_rows * num_cols):
        for j in range(1, num_rows*num_cols - dimension + 1):
            axs[-j].axis('off')  # Hide any empty subplots

    # Plot histograms for each feature
    for i in range(dimension):
        ax = axs[i]
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        h, bins, _ = ax.hist(z[:, i], bins = 32, alpha=1, color='blue', density = True, histtype='stepfilled', linewidth = plt.rcParams["lines.linewidth"], edgecolor = coralgreen[0], facecolor = coralgreen[1])
        ax.set_title(f"Feature {i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel(f"Counts / {(bins[1]-bins[0]):.1f}")

    # layout and save 
    fig.suptitle(f'Epoch {epoch}')
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace = 1.)
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_activations_{network_name}_offset_{offset}_epoch_{epoch:03d}.pdf')
    print(f'Activations saved to {path}')
    plt.close()

def plot_loss(loss, **kwargs):
    
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    # plot
    fig, ax = plt.subplots(1)
    x = np.arange(1, epoch+1)
    ax.plot(x,loss["train"], color = "r", label = "Train loss", lw = plt.rcParams["lines.linewidth"])
    ax.plot(x,loss["val"], color = "b", label = "Validation loss", lw = plt.rcParams["lines.linewidth"])

    # format
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, len(loss["train"]) + 1, len(loss["train"]) // 20)) if len(loss["train"]) >= 20 else ax.set_xticks(np.arange(0, len(loss["train"]) + 1, 2))
    ax.tick_params(which = 'minor', axis = 'x' , direction='in', length = 0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (GeV)')
    
    # layout and save
    fig.tight_layout()
    ax.legend(loc = "best", fontsize = plt.rcParams["legend.fontsize"])
    path = f"plots/redec/{sample}/"
    mkpath(path)
    fig.savefig(f'{path}{sample}_loss_{network_name}_offset_{offset}_{epoch:03d}epochs.pdf')
    print(f'Losses saved to {path}')
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










### not used right now ###
def plot_PtEtaPhiE(true_val, reco_val, theta, rec_theta, logpt, rec_logpt, **kwargs):
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')

    #true_E = (networks.PxPyPzE(true_val)[:, 3:4, :]).detach()
    #reco_E = (networks.PxPyPzE(reco_val)[:, 3:4, :]).detach()

    true_val = true_val.detach()
    reco_val = reco_val.detach()
    
    true_theta = theta.detach()
    reco_theta = rec_theta.detach()

    true_logpt = logpt.detach()
    reco_logpt = rec_logpt.detach()

    
    fig, ax = plt.subplots(1, 7)
    for j, feature in enumerate(["$p_{T}\ ({\mathrm{GeV}})$", "$\eta$", "$\phi$", "$\\theta$", "$\log{(p_{T}\ [{\\rm GeV}])}$"]):
        if j == 3:
            width = 0.25 # GeV
            nbins = int(round(max(true_theta.flatten().numpy()) - min(true_theta.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_theta.flatten().numpy(), color = "firebrick", label = "true", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
            ax[j].hist(reco_theta.flatten().numpy(), color = "blue", label = "reco", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
            ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
            ax[j].set_yscale("log")
        elif j == 4:
            width = 0.25 # GeV
            nbins = int(round(max(true_logpt.flatten().numpy()) - min(true_logpt.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_logpt.flatten().numpy(), color = "firebrick", label = "true", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
            nbins = int(round(max(reco_logpt.flatten().numpy()) - min(reco_logpt.flatten().numpy())) / width) + 1 # have 20 GeV bins in each histo
            ax[j].hist(reco_logpt.flatten().numpy(), color = "blue", label = "reco", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)
            ax[j].set_ylabel(f'Events / {(bins1[1]-bins1[0]):.1f}')
            ax[j].set_yscale("log")
        else:
            width = 20 if j == 0 else 0.25
            nbins = int(round(max(true_val[:, j, :].flatten().numpy()) - min(true_val[:, j, :].flatten().numpy())) /  width) + 1 # have 20 GeV bins in each histo
            h, bins1, _ = ax[j].hist(true_val[:, j, :].flatten().numpy(), color = "firebrick", label = "true", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = nbins)
            nbins = int(round(max(reco_val[:, j, :].flatten().numpy()) - min(reco_val[:, j, :].flatten().numpy())) /  width) + 1 if max(reco_val[:, j, :].flatten().numpy()) < 1e4 else 8192
            ax[j].hist(reco_val[:, j, :].flatten().numpy(), color = "blue", label = "reco", histtype = "step", linewidth = plt.rcParams["lines.linewidth"], bins = bins1)

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

def plot_training_residuals_PtEtaPhiEm2jm4j(true_val, reco_val, **kwargs): # expects [batch, (3) features, (4) jets] shaped tensors 
    # Get additional keyword arguments using kwargs.get()
    offset = kwargs.get('offset')
    epoch = kwargs.get('epoch')
    sample = kwargs.get('sample')
    network_name = kwargs.get('network_name')
    
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
    
    fig, ax = plt.subplots(2, 4)
    cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
    vmax_mob = 0
    i, j = 0, 0
    for feature in ["$p_{T}\ ({\mathrm{GeV}})$", "$\eta$", "$\phi$", "$m_{j}\ ({\mathrm{GeV}})$", "$m_{2j}\ ({\mathrm{GeV}})$", "$m_{4j}\ ({\mathrm{GeV}})$"]:
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
