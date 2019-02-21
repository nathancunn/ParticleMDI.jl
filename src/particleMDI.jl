module particleMDI
import Clustering.hclust
import Distributions
import NonUniformRandomVariateGeneration
import Plots
import StatsBase


# Core code
include("pmdi.jl")
include("misc.jl")
include("update_hypers.jl")

# Datatype specific
include("datatypes/gaussian_cluster.jl")
include("datatypes/categorical_cluster.jl")
include("datatypes/binom_cluster.jl")

# Output analysis
include("output_analysis/acf_plots.jl")
include("output_analysis/phi_plots.jl")
include("output_analysis/nclust_plots.jl")
include("output_analysis/consensus_map.jl")



export pmdi,
        gaussian_normalise!,
        plot_phi_chain, plot_phi_matrix,
        plot_nclust_chain, plot_nclust_hist,
         generate_psm, consensus_map

end # module
