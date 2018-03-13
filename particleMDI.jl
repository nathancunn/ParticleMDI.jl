using RDatasets
K = 1
data = [Matrix(dataset("datasets", "iris")[:, 1:4])]
dataTypes = [gaussianCluster]
s = [repeat(1:3,  inner = 50)]
K       = length(data) # No. of datasets
n_obs   = size(data[1])[1]
d       = [size(data[k])[2] for k = 1:K]

obs_order = randperm(n_obs)
# The mutation function
function M!(newParticle::mdiParticle{K}, rng::SMCRNG, p::Int64,
  particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
  print(particle.ID, " start ", "\n")
  ρ = 0.05
  n_obs = 150
  N = 3
  i_1 = Int64(floor(0.05 * 150))
  # K = length(data)
  n_obs   = size(data[1])[1]
  dataTypes = [gaussianCluster]
  obs_order = 1:150


  if p == 1
      # Then add this first free observation to a cluster
      for k = 1:K
          newParticle.clusters[k] = dataTypes[k]{n_obs, d[k], N}()
      end


      for k = 1:K
          particle_add(data[k][obs_order[1], :], 1, obs_order[1], newParticle.clusters[k])
      end
      # Set up scratch space
      # All particles have same ID initially
      scratch.currentID = 1
      scratch.newID     = 1
      scratch.logprob   = [zeros(N) for k in 1:K]
      scratch.iter      = 1
  else
      newParticle = particle
      if smcio.resample[p - 1] == true
          # Reset particle weights
          newParticle.logW = 0
    end
        scratch.logprob         = [calc_logprob(data[k][obs_order[p], :], newParticle.clusters[k]) for k = 1:K]
      # logprob           = [calc_logprob(data[k][obs_order[p + i_1], :], newParticle.clusters[k]) for k = 1:K]
      logprob           = scratch.logprob
      # newParticle.ID    = scratch.newID
      fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
      newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
      fprob             = fprob ./ last.(fprob)
      sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]
      for k = 1:K
          particle_add(data[k][obs_order[p], :], sstar[k], obs_order[p], newParticle.clusters[k])
      end

      for k = 1:K
          newParticle.ID = Int64(newParticle.ID * N + sstar[k])
      end
      scratch.currentID       = newParticle.ID
      newParticle.ID = p
  end
  print("\n", newParticle.ID, " end ", "\n")
end

# The particle weight
function lG(p::Int64, particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
  return particle.logW
end

model = SMCModel(M!, lG, 150, mdiParticle{K}, mdiScratch{K})
smcio = smcio = SMCIO{model.particle, model.pScratch}(2, 150, 1, true, 0.2)

smc!(model, smcio)
