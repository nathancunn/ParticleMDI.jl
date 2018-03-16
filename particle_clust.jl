using CSVFiles
using Distributions
using Iterators
using StaticArrays
using StatsBase

include("gaussian_cluster.jl")
include("update_hypers.jl")
include("misc.jl")

N = 10
ρ = 0.33
particles = 16
n_obs = Int64(150)
K = 2
d     = [rand(2:5) for k = 1:K]
dataTypes = [gaussianCluster for k = 1:K]
dataFiles      = [[randn(Int64(n_obs * 0.5), d[k]) - 10; randn(Int64(n_obs * 0.5), d[k]) + 10] for k = 1:K]

srand(1234)
@time out = pmdi(dataFiles, dataTypes, 10, 32, 0.05, 100, "iris.csv", true, 1)

function pmdi(dataFiles, dataTypes, N::Int64, particles::Int64,
    ρ::Float64, iter::Int64, outputFile::String, initialise::Bool, output_freq::Int64)

    K       = length(dataFiles) # No. of datasets
    n_obs   = size(dataFiles[1])[1]
    d       = [size(dataFiles[k])[2] for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(dataFiles[k])[1] for k = 1:K])


    # Initialise the hyperparameters
    M = ones(K) * 2
    γ = rand(Gamma(2 / N, 1), N, K) .+ realmin(Float64)
    Φ = K > 1 ? rand(Gamma(1, 5), Int64(K * (K - 1) *0.5)) : zeros(1)

    # Initialise the allocations randomly according to γ
    s = Matrix{Int64}(zeros(n_obs, K))
    for k = 1:K
        s[:, k] = sample(1:N, Weights(γ[:, k]), n_obs)
    end

    # Get a matrix of all combinations of gammas
    γ_combn = Matrix{Int64}(N ^ K, K)
    for (i, p) in enumerate(product([1:N for k = 1:K]...))
        γ_combn[i, :] = collect(p)
    end

    # Which Φ value is activated by each of the above combinations
    Φ_index = K > 1 ? Matrix{Int64}(N ^ K, Int64(K * (K - 1) / 2)) : ones(Matrix{Bool}(N, 1), Int64)

    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_index[:, i] = (γ_combn[:, k1] .== γ_combn[:, k2])
                i += 1
            end
        end
    end

    # Get all the combinations of Φ values
    # Φ_lab = calculate_Φ_lab(K)

    Γ = Matrix{Float64}(N ^ K, K)
    for k = 1:K
        Γ[:, k] = log.(γ[:, k][γ_combn[:, k]])
    end
    Z = update_Z(Float64(0), Φ, Φ_index, Γ)


    # logprob = zeros(N, particles, K)
    # fprob = copy(logprob)
    logweight = zeros(particles)
    sstar = zeros(Matrix{Bool}(particles, K), Int64)

    # Initialise the particles
    particle = [Vector{dataTypes[k]{n_obs, d[k], N}}(particles) for k = 1:K]
    particle_IDs = ones(Matrix{Bool}(particles, K), Int64)
    for k = 1:K
        particle[k][1] = dataTypes[k]{n_obs, d[k], N}()
    end

    # Save information to file
    if K > 1
        out = vcat(map(x -> @sprintf("mass_%d", x), 1:K), map((x, y) -> @sprintf("phi_%d_%d", x, y), calculate_Φ_lab(K)[:, 1], calculate_Φ_lab(K)[:, 2]), map((x, y) -> @sprintf("K%d_n%d", x, y), repeat(1:K, inner = n_obs), repeat(1:n_obs, outer = K)))
    else
        out = vcat(map(x -> @sprintf("mass_%d", x), 1:K), map((x, y) -> @sprintf("K%d_n%d", x, y), repeat(1:K, inner = n_obs), repeat(1:n_obs, outer = K)))
    end
    out =  reshape(out, 1, length(out))
    writecsv(outputFile, out)

    fileid = open(outputFile, "a")



    # allocations = Array{Int64}(zeros(n_obs, particles, K))
    # particle = [[dataTypes[k]{n_obs, d[k], N}() for p in 1:particles] for k in 1:K]
    for it in 1:iter
        # Reset particles and IDs
        for k = 1:K
            for p in unique(particle_IDs[:, k])
                particle[k][p] = particle_reset!(particle[k][p])
            end
        end
        particle_IDs .= 1

        v = update_v!(n_obs, Z)
        update_M!(M, γ, K, N)

        Π = γ ./ sum(γ, 1)

        order_obs = randperm(n_obs)

        for k = 1:K
            for i in order_obs[1:Int64(floor(ρ * n_obs))]
                #i = order_obs[Int64(i_ind)]
                # allocations[i, :, k] = s[i, k]
                particle[k][1] = particle_add!(dataFiles[k][i, :], i, s[i, k], particle[k][1])
            end
        end


        ## The CPF
        for i in order_obs[Int64((floor(ρ * n_obs) + 1)):n_obs]
            for k = 1:K
                for p in unique(particle_IDs[:, k])
                    calc_logprob!(dataFiles[k][i, :], particle[k][p])
                end
            end

            # calc_fprob!(fprob, logprob, particle_IDs, K, Π)
            update_logweight!(logweight, particle, particle_IDs, Π, K, N)

            draw_sstar!(sstar, particle, particle_IDs, Π, K, particles, N)
            # Set first particle to reference trajectory
            for k = 1:K
                sstar[1, k] = s[i, k]
            end

            # Update the particle IDs
            new_particle_IDs = update_particleIDs!(particle_IDs, sstar, K, particles, N)


            # Add observation to cluster
            for k = 1:K
                particle_k = deepcopy(particle[k])
                for p in unique(new_particle_IDs[:, k])
                    # println(p, " ", particle_IDs[findin(new_particle_IDs[:, k], p)[1], k])
                    # println(sstar[findin(new_particle_IDs[:, k], p)[1], k])
                    # println(particle_IDs[findin(new_particle_IDs[:, k], p)[1], k])
                    particle[k][p] = particle_add!(dataFiles[k][i, :], i, sstar[findin(new_particle_IDs[:, k], p)[1], k], particle_k[particle_IDs[findin(new_particle_IDs[:, k], p)[1], k]])
                end
            end
            for k = 1:K
                # for p = 1:particles
                    # allocations[i, :, k] = sstar[:, k]
                # end
            end
            particle_IDs = deepcopy(new_particle_IDs)

            # Φ upweighting
             Φ_upweight!(logweight, particle_IDs, sstar, K, Φ)


            if calc_ESS(logweight) <= 0.5 * particles
                partstar = draw_partstar(logweight, particles)
                partstar[1] = 1
                logweight .= 1.0

                for k = 1:K
                        particle_IDs[:, k] = particle_IDs[partstar, k]
                        # allocations[:, :, k] = allocations[:, partstar, k]
                end
            end
        end
        # Select a single particle
        p_star = sample(1:particles, Weights(logweight))
            for k = 1:K
                s[:, k] = particle[k][particle_IDs[:, k][p_star]].c
            end
            align_labels!(s, Φ, γ, N, K)
            update_Φ!(Φ, v, s, Φ_index, γ, K, Γ)
            update_γ!(γ, Φ, v, s, Φ_index, γ_combn, Γ, N, K)
            for k = 1:K
                Γ[:, k] = log.(γ[γ_combn[:, k], k])
            end
            Z = update_Z(Float64(0), Φ, Φ_index, Γ)
            if K > 1
                writecsv(fileid, [M; Φ; s[1:(n_obs * K)]]')
            else
                writecsv(fileid, [M; s[1:(n_obs * K)]]')
            end
        end
        close(fileid)
        return s
end
