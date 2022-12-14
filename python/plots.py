import pickle
import hist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


with open('data/hists_normalized.pkl', 'rb') as hfile:
    hdict = pickle.load(hfile)

m4j = hdict['hists']['m4j']
h3 = m4j['data/threeTag_picoAOD.root', 'preselection', 'SB', :]
h4 = m4j['data/fourTag_picoAOD.root',  'preselection', 'SB', :]
    
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
fig.savefig('test.pdf')
