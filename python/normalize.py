import pickle
import hist

with open('data/hists.pkl', 'rb') as hfile:
    hdict = pickle.load(hfile)

cutflow = hdict['cutflow']

w4b = cutflow['data/fourTag_picoAOD.root', 'preselection', 'SB']
w3b = cutflow['data/threeTag_picoAOD.root', 'preselection', 'SB']
norm = w4b.value/w3b.value
print(f'{w4b.value:,.7}/{w3b.value:,.7} = {norm}')

output = {'w4b': w4b,
          'w3b': w3b,
          'norm': norm}

with open('data/normalize.pkl', 'wb') as nfile:
    pickle.dump(output, nfile)

# def normalize(hdict):
#     for key, value in hdict.items():
#         if isinstance(value, dict):
#             normalize(value)
#         elif isinstance(value, hist.Hist):
#             h3b = norm*value['data/threeTag_picoAOD.root', ...]
#             value['data/threeTag_picoAOD.root', ...] = h3b

# normalize(hdict)

# with open('data/hists_normalized.root', 'wb') as hfile:
#     pickle.dump(hdict, hfile)
