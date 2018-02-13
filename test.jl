include("smc.jl")
## Generate some data for testing
n_obs = Int64(150)
K = 2
d     = [rand(2:5) for k = 1:K]
dataTypes = [gaussianCluster for k = 1:K]
data      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]
s = [repeat(1:2, inner = Int(n_obs * 0.5)) for k = 1:K]
model, smcio = pMDImodel(data, dataTypes, 3, 32, s, 0.25, 1, false, 0.5)
@time smc!(model, smcio)


# Using the iris data
using RDatasets
K = 1
data = [Matrix(dataset("datasets", "iris")[:, 1:4])]
dataTypes = [gaussianCluster]
s = [repeat(1:3,  inner = 50)]
model, smcio = pMDImodel(data, dataTypes, 3, 32, s, 0.25, 1, false, 0.5)
@time smc!(model, smcio)
