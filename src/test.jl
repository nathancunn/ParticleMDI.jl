include("pmdi.jl")
using particleMDI
using RDatasets


using CSV
data = [Matrix{Float64}(CSV.read("data/synth_gaussian$i.csv", datarow = 2)[:, 2:end]) for i = 1:2]
dataTypes = [particleMDI.GaussianCluster for i = 1:2]
gaussian_normalise!(data[1])
gaussian_normalise!(data[2])
@time out = pmdi(data, dataTypes, 10, 32, 0.99, 1, "output/iris_2.csv", true, 1)
@time out = pmdi(data, dataTypes, 10, 32, 0.25, 1000, "output/iris_2.csv", true, 1)

data = [Matrix(dataset("datasets", "iris")[:, 1:4])]
gaussian_normalise!(data[1])
dataTypes = [particleMDI.GaussianCluster]
@profiler out = pmdi(data, dataTypes, 10, 32, 0.99, 1, "output/iris_2.csv", true, 1)
@profiler out = pmdi(data, dataTypes, 10, 32, 0.25, 1000, "output/iris_2.csv", true, 1)


using CSV
KIRC_SCNA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_SCNA_numeric.csv", datarow = 2)
KIRC = [Matrix{Int64}(KIRC_SCNA[:, 2:70])]
KIRCtype = [particleMDI.CategoricalCluster]

cl = particleMDI.CategoricalCluster(KIRC[1], 10)
logprob = zeros(10)
@profiler for i = 1:100
     particleMDI.calc_logprob!(logprob, KIRC[1][i, :], cl)
 end
@time particleMDI.particle_add!(cl, KIRC[1][1, :], )
logprob
@time out = pmdi(KIRC, KIRCtype, 10, 32, 0.99, 1, "output/iris_2.csv", true, 1)
@time out = pmdi(KIRC, KIRCtype, 10, 32, 0.25, 1000, "output/iris_2.csv", true, 1)
@profiler out = pmdi(KIRC, KIRCtype, 10, 32, 0.25, 100, "output/iris_2.csv", true, 1)

# An example use of pMDImodel
# Takes arguments
# data - the datafiles, specified in a Matrix of data arrays
# dataTypes - a concatenation of the data type structs (only GaussianCluster now)
# C - the maximum number of clusters to be fit
# N - the number of particles
# ρ - the proportion fo data to be fixed
## Generate some data for testing
srand(1234)
n_obs = Int64(100)
K = 1
d = [rand(2:5) for k = 1:K]
dataTypes = [particleMDI.GaussianCluster for k = 1:K]
dataFiles      = [[randn(Int64(n_obs * 0.5), d[k]) - 3; randn(Int64(n_obs * 0.5), d[k]) + 3] for k = 1:K]
for k = 1:K
    gaussian_normalise!(dataFiles[k])
end
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.99, 1, "output/sim.csv", true, 1)
@time out = pmdi(dataFiles, dataTypes, 10, 128, 0.25, 1000, "output/sim.csv", true, 1)

out = zeros(10)
for i = 1:10
    for j = 1:10
        dataFiles = [[randn(Int64(n_obs * 0.5), 2 ^ (i - 1)) - 3; randn(Int64(n_obs * 0.5), 2 ^ (i - 1)) + 3]]
        gaussian_normalise!(dataFiles[1])
        out[i] += @elapsed pmdi(dataFiles, dataTypes, 10, 32, 0.25, 1000, "output/sim.csv", true, 1)
    end
    out[i] /= 10
    print(i)
end

draw(SVGJS("acf_test.js.svg"), clustacf(Array(output)))
,
plot(x = 2 .^ collect(0:9), y = out, Geom.point, Geom.line,
    Guide.xlabel("No. Features"),
    Guide.ylabel("Elapsed time (s)"))
)


using CSV
KIRC_mRNA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_mRNA_exp_subset.csv", datarow = 2)
KIRC_miRNA  = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_miRNA_exp_subset.csv", datarow = 2)
KIRC_RPPA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_protein_subset.csv", datarow = 2)
KIRC_DNA    = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_DNA_subset.csv", datarow = 2)
KIRC_SCNA   = CSV.read("../particleMDI_analysis/Yuan_et_al/preproc_data/KIRC_SCNA_numeric.csv", datarow = 2)



KIRC = Vector{Any}(5)
KIRC[1] = Matrix{Float64}(KIRC_mRNA[:, 2:101])
KIRC[2] = Matrix{Float64}(KIRC_miRNA[:, 2:101])
KIRC[3] = Matrix{Float64}(KIRC_RPPA[:, 2:101])
KIRC[4] = Matrix{Float64}(KIRC_DNA[:, 2:101])
KIRC[5] = Matrix{Int64}(KIRC_SCNA[:, 2:70])
for k in 1:4
    gaussian_normalise!(KIRC[k])
end

dataTypes = [particleMDI.GaussianCluster, particleMDI.GaussianCluster,
            particleMDI.GaussianCluster, particleMDI.GaussianCluster,
            particleMDI.CategoricalCluster]
@time out = pmdi(KIRC, dataTypes, 10, 2, 0.99, 1, "output/KIRC_test.csv", true, 1)
@time out = pmdi(KIRC, dataTypes, 10, 32, 0.25, 1000, "output/KIRC_test.csv", true, 1)

Profile.clear()
srand(112)
@profiler out = pmdi(KIRC, dataTypes, 10, 32, 0.25, 10, "output/KIRC.csv", true, 1)
Juno.profiler()
