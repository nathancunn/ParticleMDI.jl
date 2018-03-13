using StatsBase
using Distributions
using Iterators

using StaticArrays
include("gaussian_cluster.jl")
include("update_hypers.jl")
include("misc.jl")

function pmdi(dataFiles::Vector, dataTypes::Vector, N::Int64, particles::Int64,
    ρ::Float16, iter::Int64, output_file::String, initialise::Bool, output_freq::Int64)

    K       = length(dataFiles) # No. of datasets
    n_obs   = size(dataFiles[1])[1]
    d       = [size(dataFiles[k])[2] for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(dataFiles[k])[1] for k = 1:K])


    M = ones(K) * 2
    γ = [rand(Gamma(2 / N, 1), N) + realmin(Float64) for k = 1:K]
    s = [sample(1:N, Weights(γ[k]), n_obs) for k = 1:K]
    Φ = K > 1 ? rand(Gamma(1, 0.2), Int64(K * (K - 1) *0.5)) : 0

    γ_combn = Matrix{Int64}(N ^ K, K)
    for (i, p) in enumerate(product([1:N for k = 1:K]...))
        γ_combn[i, :] = collect(p)
    end

    Φ_index = K > 1 ? Matrix{Int64}(N ^ K, Int64(K * (K - 1) / 2)) : ones(N, 1)
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_index[:, i] = (γ_combn[:, k1] .== γ_combn[:, k2])
                i += 1
            end
        end
    end

    Γ = Matrix{Float64}(N ^ K, K)
    for k = 1:K
        Γ[:, k] = log.(γ[k][γ_combn[:, k]])
    end
    Z = update_Z!(Float64(0), Φ, Φ_index, Γ)
    Φ_lab = calculate_Φ_lab(K)


    logprob = [[MVector{N, Float64}() for p = 1:particles] for k = 1:K]

    # Initialise the particles
    particle = [[dataTypes[k]{n_obs, d[k], N}() for p in 1:particles] for k in 1:K]
    particle_IDs = [ones(1:particles) for k in 1:K]
    for it in 1:iter
        # Reset particles and IDs
        for k = 1:K
            particle_reset!.(particle[k])
        end
        fill!.(particle_IDs, 1)


        v = update_v!(n_obs, Z)
        update_M!(M, γ, N)

        Π = map(x -> x / sum(x), γ)

        order_obs = randperm(n_obs)

        for k = 1:K
            for i_ind in 1:floor(ρ * n_obs)
                i = order_obs[Int64(i_ind)]
                for p in unique(particle_IDs[k])
                    particle[k][findin(particle_IDs[k], p)] = particle_add!(dataFiles[k][i, :], s[k][i], i, particle[k][p])
                end
            end
        end

        ## The CPF
        for i_ind in (floor(ρ * n_obs) + 1):n_obs
            i = order_obs[Int64(i_ind)]

            @time calc_logprob!(logprob[1][1], dataFiles[k][i, :], particle[k][1])
            for k = 1:K
                calc_logprob!.(())
                broadcast(x -> calc_logprob!.((logprob[k][x]), dataFiles[k][i, :], (particle[k][x])), unique(particle_IDs[k]))
                broadc
                calc_logprob!.((logprob[k][unique(particle_IDs[k])]), dataFiles[k][i, :], (particle[k][unique(particle_IDs[k])]))
            @time calc_logprob(dataFiles[k][i, :], particle[k][1])
            @time [map(x -> calc_logprob(logprob dataFiles[k][i, :], x), particle[k][unique(particle_IDs[k])]) for k = 1:K]




particle_add!(obs, sstar::Int64, p::Int64, cl::gaussianCluster{n_obs, d, N})
end
