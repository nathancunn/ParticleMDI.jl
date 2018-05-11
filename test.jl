include("particle_clust.jl")

# An example use of pMDImodel
# Takes arguments
# data - the datafiles, specified in a Matrix of data arrays
# dataTypes - a concatenation of the data type structs (only gaussianCluster now)
# C - the maximum number of clusters to be fit
# N - the number of particles
# ρ - the proportion fo data to be fixed
## Generate some data for testing

n_obs = Int64(150)
K = 1
d = [rand(2:5) for k = 1:K]
dataTypes = [gaussianCluster for k = 1:K]
dataFiles      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.95, 1, "iris_2.csv", true, 1)
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.05, 1000, "iris_2.csv", true, 1)

using RDatasets
data = [Matrix(dataset("datasets", "iris")[:, 1:4])]
gaussian_normalise!(data[1])
dataTypes = [gaussianCluster]
@time out = pmdi(data, dataTypes, 10, 32, 0.95, 1, "output/iris_2.csv", true, 1)
srand(1234)
@time out = pmdi(data, dataTypes, 10, 32, 0.25, 1000, "output/iris_2.csv", true, 1)
Profile.clear()
@profile out = pmdi(data, dataTypes, 10, 32, 0.05, 500, "output/iris_2.csv", true, 1)
Juno.profiler()

dataFile = data[1]
a = gaussianCluster{150, 4, 10}()
for i = 1:100
    obs = dataFile[i, :]
    if i <= 50
        a = particle_add(obs, i, 1, a)
    else
        a = particle_add(obs, i, 2, a)
    end
end
