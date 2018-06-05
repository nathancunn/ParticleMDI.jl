include("particle_clust.jl")

# An example use of pMDImodel
# Takes arguments
# data - the datafiles, specified in a Matrix of data arrays
# dataTypes - a concatenation of the data type structs (only gaussianCluster now)
# C - the maximum number of clusters to be fit
# N - the number of particles
# ρ - the proportion fo data to be fixed
## Generate some data for testing

n_obs = Int64(50)
K = 1
d = [rand(2:5) for k = 1:K]
dataTypes = [gaussianCluster for k = 1:K]
dataFiles      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]
for k = 1:K
    gaussian_normalise!(dataFiles[k])
end
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.99, 1, "output/sim.csv", true, 1)
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.25, 1000, "output/sim.csv", true, 1)

using RDatasets
data = [Matrix(dataset("datasets", "iris")[:, 1:4])]
gaussian_normalise!(data[1])
dataTypes = [gaussianCluster]
@time out = pmdi(data, dataTypes, 10, 32, 0.99, 1, "output/iris_2.csv", true, 1)
srand(1234)
@time out = pmdi(data, dataTypes, 10, 32, 0.25, 1000, "output/iris_2.csv", true, 1)
Profile.clear()
@profile out = pmdi(data, dataTypes, 10, 32, 0.05, 1000, "output/iris_2.csv", true, 1)
Juno.profiler()

KIRC_mRNA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_mRNA_exp_subset.csv", datarow = 2)
KIRC_miRNA  = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_miRNA_exp_subset.csv", datarow = 2)
KIRC_RPPA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_protein_subset.csv", datarow = 2)
KIRC_DNA    = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_DNA_subset.csv", datarow = 2)

KIRC = [Matrix{Float64}(KIRC_mRNA[:, 2:101]),
        Matrix{Float64}(KIRC_miRNA[:, 2:101]),
        Matrix{Float64}(KIRC_RPPA[:, 2:101]),
        Matrix{Float64}(KIRC_DNA[:, 2:101])]
for k in 1:length(KIRC)
    gaussian_normalise!(KIRC[k])
end

dataTypes = [gaussianCluster for k in 1:4]

@time out = pmdi(KIRC, dataTypes, 10, 32, 0.99, 1, "output/KIRC.csv", true, 1)
@time out = pmdi(KIRC, dataTypes, 10, 128, 0.25, 100000, "output/KIRC.csv", true, 1)




dataFile = data[1]

a = gaussianCluster{150, 4, 10}()
@time for i = 1:100
    obs = dataFile[i, :]
    if i <= 50
        a = particle_add(obs, i, 1, a)
    else
        a = particle_add(obs, i, 2, a)
    end
end
a = gaussianCluster{150, 4, 10}()
for i = 1:100
    obs = dataFile[i, :]
    if i <= 50
        particle_add!(obs, i, 1, a)
    else
        particle_add!(obs, i, 2, a)
    end
end
@benchmark particle_reset!(a)
@benchmark particle_reset_2!(a)


@time ID_match(a, b, 10)


A = sample(1:10, 1000)
findin(A, 2)
findindices(A, 2)
