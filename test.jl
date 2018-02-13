include("smc.jl")
## Generate some data for testing
n_obs = Int64(150)
K = 2
d     = [rand(2:5) for k = 1:K]
dataTypes = [gaussianCluster for k = 1:K]
data      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]


using RDatasets
data = [Matrix(dataset("datasets", "iris")[:, 1:4])]

model, smcio = pMDImodel(data, dataTypes, 3, 10, 1, false, 0.5)
@time smc!(model, smcio)
