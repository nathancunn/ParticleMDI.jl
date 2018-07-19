__precompile__()

module particleMDI

import Distributions
import CSVFiles
import Gadfly
import Iterators
import StatsBase

# Core code
include("particle_clust.jl")
include("misc.jl")
include("update_hypers.jl")

# Datatype specific
include("datatypes/gaussian_cluster.jl")
include("datatypes/multinomial_cluster.jl")

# Output analysis
include("output_analysis/acf_plots.jl")


export pmdi, clustrand, gaussian_normalise!

end # module
