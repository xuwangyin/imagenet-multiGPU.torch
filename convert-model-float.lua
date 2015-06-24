require('cunn')
modelfile = arg[1]
m = torch.load(modelfile)
n = m:float()
i, j = string.find(modelfile, '.t7')
savename = string.sub(modelfile, 1, i - 1) .. '_float.t7'
torch.save(savename, n)
