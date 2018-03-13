# Load in the relevant cluster definitions
include("gaussian_cluster.jl")

mutable struct mdiParticle{K}
  c::Vector{Int64}
  clusters::Vector{Any}
  logW::Float64
  ID::Int64
  mdiParticle{K}() where {K} = new(Vector{Int64}(K), Vector{Any}(K), Float64(0), Int64(1))
end

mutable struct mdiScratch{K}
    currentID::Int64
    newID::Int64
    logprob::Vector{Any}
    iter::Int64
    mdiScratch{K}() where K = mdiScratch{K}(Int64(1),
                              Int64(1),
                              Vector{Any}(K),
                              Int64(1))
end
