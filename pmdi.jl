__precompile__()

module particleMDI
# require("*.jl") makes function available to all workers
import CSVFiles
import Gadfly
import Iterators
import NonUniformRandomVariateGeneration
import StatsBase


using Distributions
using CSVFiles
using Iterators
using StatsBase


# Core code
include("particle_clust.jl")
include("misc.jl")
include("update_hypers.jl")

# Datatype specific
include("datatypes/gaussian_cluster.jl")
include("datatypes/categorical_cluster.jl")

# Output analysis
include("output_analysis/acf_plots.jl")
include("output_analysis/phi_plots.jl")



export pmdi, clustrand, clustacf, gaussian_normalise!, plot_phichain, plot_phimatrix

end # module
