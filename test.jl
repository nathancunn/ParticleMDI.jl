include("smc.jl")
## Generate some data for testing
n_obs = Int64(100)
K = 2
d     = [rand(2:5) for k = 1:K]
# Generate some simple univariate gaussian data with v. distinct clusters
N = 10
dataTypes = [gaussianCluster for k = 1:K]
data      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]




model = SMCModel(M!, lG, n_obs, mdiParticle{K}, Void)

smcio = SMCIO{model.particle, model.pScratch}(10, n_obs, 1, false, 0.5)

@time smc!(model, smcio)
