# Implement particle cluster analysis for univariate Gaussian data
## Load the packages
using SequentialMonteCarlo
using StaticArrays

include("particles.jl")

function pMDImodel(data, dataTypes, N, particles, s, ρ, nThreads, fullOutput, essThreshold)
    K       = length(data) # No. of datasets
    n_obs   = size(data[1])[1]
    d       = [size(data[k])[2] for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(data[k])[1] for k = 1:K])
    obs_order = randperm(n_obs)
    # The mutation function
    function M!(newParticle::mdiParticle{K}, rng::SMCRNG, p::Int64,
      particle::mdiParticle{K}, scratch::Void) where {K}
      i_1 = Int64(floor(ρ * n_obs))
      K = length(data)
      n_obs   = size(data[1])[1]

      if p == 1
      # Move the fixed observations to their cluster
      for k = 1:K
          newParticle.clusters[k] = dataTypes[k]{n_obs, d[k], N}()
          for i = 1:i_1
              i = obs_order[Int64(i)]
              particle_add(data[k][i, :], Int64(s[k][i]), i, newParticle.clusters[k])
          end
      end
      # Then add this first free observation to a cluster
      logprob           = [calc_logprob(data[k][obs_order[1 + i_1], :], newParticle.clusters[k]) for k = 1:K]
      fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
      newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
      fprob             = fprob ./ last.(fprob)
      sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]

      for k = 1:K
          particle_add(data[k][obs_order[1 + i_1], :], sstar[k], obs_order[1 + i_1], newParticle.clusters[k])
      end
      else
          newParticle = particle
          if smcio.resample[p - 1] == true
              # Reset particle weights
              newParticle.logW = 0
              # Reallocate previous observations
              for j = 1:(p - 1)
                  sstar             = [newParticle.clusters[k].c[obs_order[j + i_1]] for k = 1:K]
                  for k = 1:K
                      particle_remove(data[k][obs_order[j + i_1], :], sstar[k], obs_order[j + i_1], newParticle.clusters[k])
                  end
                  logprob           = [calc_logprob(data[k][obs_order[j + i_1], :], newParticle.clusters[k]) for k = 1:K]
                  fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
                  fprob             = fprob ./ last.(fprob)
                  sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]
                  for k = 1:K
                      particle_add(data[k][obs_order[j + i_1], :], sstar[k], obs_order[j + i_1], newParticle.clusters[k])
                  end
              end
        end
          logprob           = [calc_logprob(data[k][obs_order[p + i_1], :], newParticle.clusters[k]) for k = 1:K]
          fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
          newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
          fprob             = fprob ./ last.(fprob)
          sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]

          for k = 1:K
              particle_add(data[k][obs_order[p + i_1], :], sstar[k], obs_order[p + i_1], newParticle.clusters[k])
          end

      end
    end

    # The particle weight
    function lG(p::Int64, particle::mdiParticle{K}, ::Void) where {K}
      return particle.logW
    end

    model = SMCModel(M!, lG, Int64(n_obs - floor(ρ * n_obs)), mdiParticle{K}, Void)
    smcio = smcio = SMCIO{model.particle, model.pScratch}(particles,Int64(n_obs - floor(ρ * n_obs)), nThreads, fullOutput, essThreshold)
    return model, smcio

end
