__precompile__()

module particleMDI
# require("*.jl") makes function available to all workers
import Clustering
import CSVFiles
import Gadfly
import Iterators
import StatsBase


using Distributions
using CSVFiles
using Iterators
using StatsBase


# Core code
include("pmdi.jl")
include("misc.jl")
include("update_hypers.jl")

# Datatype specific
include("datatypes/gaussian_cluster.jl")
include("datatypes/categorical_cluster.jl")

# Output analysis
include("output_analysis/acf_plots.jl")
include("output_analysis/phi_plots.jl")
include("output_analysis/consensus_map.jl")



export pmdi, clustrand, clustacf, gaussian_normalise!, plot_phichain, plot_phimatrix, generate_psm, consensus_map

end # module
