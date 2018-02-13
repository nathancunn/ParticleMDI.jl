# Load in the relevant cluster definitions
include("gaussian_cluster.jl")

mutable struct mdiParticle{K}
  clusters::Vector{Any}
  logW::Float64
  mdiParticle{K}() where {K} = new(Vector{Any}(K), Float64(0))
end
