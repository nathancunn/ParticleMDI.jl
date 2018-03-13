library(XRJulia)
Sys.setenv(JULIA_BIN = "C:/Users/ncunningham/AppData/Local/Julia-0.6.0/bin/julia.exe")
juliaEval("
using SequentialMonteCarlo")
juliaEval("using StaticArrays")

juliaEval("
mutable struct gaussianCluster{n_obs, d, N}
  n::SVector{N, Int64}
  mu::MMatrix{N, d, Float64}
  sigma::MMatrix{N, d, Float64}
  lambda::MMatrix{N, d,  Float64}
  beta::MMatrix{N, d, Float64}
gaussianCluster{n_obs, d, N}() where {n_obs, d, N} = new(
SVector{N, Int64}(zeros(Int64, N)),
MMatrix{N, d, Float64}(zeros(N, d)),
MMatrix{N, d, Float64}(zeros(N, d)),
MMatrix{N, d, Float64}(ones(N, d)),
MMatrix{N, d, Float64}(ones(N, d)))
end
")
juliaEval("
mutable struct mdiParticle{K}
c::Vector{Int64}
clusters::Vector{Any}
logW::Float64
ID::Int64
mdiParticle{K}() where {K} = new(Vector{Int64}(K), Vector{Any}(K), Float64(0), Int64(1))
end
")
juliaEval("
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
")

juliaEval("
function calc_logprob(obs, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
# Initialise with logprob for empty clusters
logprob = MVector{N, Float64}(repmat(sum( lgamma(0.75) -
lgamma(0.25) +
0.5 * log(1 / (0.5 * pi)) -
0.75 * log.(1 + 2 * (obs) .^ 2),
1),
N))
# Identify all non-empty clusters
ind = SVector{N, Bool}(cl.n .!= 0)
logprob[ind] = sum(lgamma.((cl.n[ind] + 2) / 2) .-
lgamma.((1 + cl.n[ind]) / 2) .+
0.5 .* log.(cl.lambda[ind, :] ./ ((0.5 + 0.5 .* cl.n[ind]) .* pi)) -
((cl.n[ind] + 2) ./ 2) .* log.(1 + (1 ./ (0.5 + cl.n[ind] .* 0.5)) .*
(-cl.mu[ind, :] .+ transpose(obs)) .^ 2 .* cl.lambda[ind, :])
, 2)
return logprob
end
")

juliaEval("
function particle_add!(obs, sstar::Int64, p::Int64, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
# cl.c            = sstar
cl.beta[sstar, :]  -= 0.5 .* (- 2 * cl.sigma[sstar, :] .* cl.mu[sstar, :] +
cl.n[sstar] .* cl.mu[sstar, :] .^ 2) .+
(cl.n[sstar] / (2 * (cl.n[sstar] + 1))) .* cl.mu[sstar, :] .^ 2
cl.n            = setindex(cl.n, cl.n[sstar] + 1, sstar)
cl.sigma[sstar, :]  += obs
cl.mu[sstar, :]  = cl.sigma[sstar, :] ./ (cl.n[sstar] + 1)
cl.beta[sstar, :]  += 0.5 * ((obs .^ 2) - 2 * cl.sigma[sstar, :] .* cl.mu[sstar, :] .+
cl.n[sstar] .* cl.mu[sstar, :] .^ 2) .+
(cl.n[sstar] / (2 * (cl.n[sstar] + 1))) .* cl.mu[sstar, :] .^ 2
cl.lambda[sstar, :]    = ((0.5 * cl.n[sstar] + 0.5) .* (cl.n[sstar] + 1)) ./
(cl.beta[sstar, :] .* (cl.n[sstar] + 2))
return
end
")

juliaEval("
function pMDImodel(data, dataTypes, N, particles, s, rho, nThreads, fullOutput, essThreshold)
   K       = length(data)
   n_obs   = size(data[1])[1]
   d       = [size(data[k])[2] for k = 1:K]
   @assert length(dataTypes) == K
   @assert all(x->x==n_obs, [size(data[k])[1] for k = 1:K])
   # The mutation function
   function M!(newParticle::mdiParticle{K}, rng::SMCRNG, p::Int64,
   particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
   i_1 = Int64(floor(rho * n_obs))
   K = length(data)
   n_obs   = size(data[1])[1]
   if p <= i_1
   if p == 1
   for k = 1:K
   newParticle.clusters[k] = dataTypes[k]{n_obs, d[k], N}()
   end
   end
   newParticle.c      = [s[k][p] for k = 1:K]
   for k = 1:K
   particle_add!(data[k][1 + i_1, :], newParticle.c[k], 1 + i_1, newParticle.clusters[k])
   end
   # Set up scratch space
   # All particles have same ID initially
   scratch.currentID = 1
   scratch.newID     = 1
   scratch.logprob   = [zeros(N) for k in 1:K]
   scratch.iter      = 1
   newParticle.ID    = 1
   else
   if smcio.resample[p - 1] == true
   # Reset particle weights
   newParticle.logW = 0
   end
   if scratch.iter == p & scratch.currentID != particle.ID
   scratch.logprob         = [calc_logprob(data[k][p, :], particle.clusters[k]) for k = 1:K]
   scratch.newID           += 1
   scratch.currentID       = particle.ID
   elseif scratch.iter != p
   scratch.iter        = p
   scratch.newID       = 1
   scratch.currentID   = particle.ID
   scratch.logprob     = [calc_logprob(data[k][p, :], particle.clusters[k]) for k = 1:K]
   end
   logprob           = scratch.logprob
   newParticle.ID    = scratch.newID
   fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
   newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
   fprob             = fprob ./ last.(fprob)
   newParticle.c     = [find(fprob[k] .> rand())[1] for k = 1:K]
   for k = 1:K
   particle_add!(data[k][p, :], newParticle.c[k], p, newParticle.clusters[k])
   end

   for k = 1:K
   newParticle.ID = Int64(newParticle.ID * N + newParticle.c[k])
   end
   end
   end
   # The particle weight
   function lG(p::Int64, particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
   return particle.logW
   end
   model = SMCModel(M!, lG, n_obs, mdiParticle{K}, mdiScratch{K})
   return model
   end")


msmc <- juliaEval("
function setup(data, datatypes, nComponents, particles, s, nThreads, essThreshold)
  model = pMDImodel(data, dataTypes, nComponents, particles, s, 0.05, nThreads, true, essThreshold)
  smcio = SMCIO{model.particle, model.pScratch}(particles, n_obs, nThreads, true, essThreshold)
  return model, smcio
  end")
msmc_jl <- JuliaFunction(msmc)
msmc_jl(data_files, data_types, 10, 8, (rep(1:3, each = 50)), 4, 2L)


test <- juliaEval("
                  function doThis(data)
                  return(data)
                  end")
test_jl <- JuliaFunction(test)
a <- test_jl(data_files)
juliaGet(a)
