using CSVFiles
using Distributions.Gamma
using Distributions.logpdf
using Distributions.sample
using Iterators
using StatsBase

"""
`pmdi(dataFiles, dataTypes, N::Int64, particles::Int64,
ρ::Float64, iter::Int64, outputFile::String, initialise::Bool,
output_freq::Int64)`

Runs particleMDI on specified datasets
## Input
- `dataFiles::Vector` a vector of K data matrices to be analysed
- `dataTypes::Vector` a vector of K datatypes. Independent multivariate normals can be
specified with `particleMDI.gaussianCluster`
- `N::Int64` the maximum number of clusters to fit
- `particles::Int64` the number of particles
- `ρ::Float64` proportion of allocations assumed known in each MCMC iteration
- `iter::Int64` number of iterations to run
- `outputFile::String` specification of a CSV file to store output
- `initialise::Bool` if false, the algorithm begins at last output recorded in
`outputFile` otherwise begin fresh.
- `output_freq` how often to write output to file (may be removed as time unaffected)

## Output
Outputs a .csv file, each row containing:
- Mass parameter for datasets `1:K`
- Φ value for `(n * (n - 1) / 2) pairs of datasets`
- c cluster allocations for observations `1:n` in datasets `1:k`

Returns a `n × K` matrix of cluster allocations.
"""
function pmdi(dataFiles, dataTypes, N::Int64, particles::Int64,
    ρ::Float64, iter::Int64, outputFile::String, initialise::Bool, output_freq::Int64)
    K       = Int64(length(dataFiles)) # No. of datasets
    n_obs   = Int64(size(dataFiles[1])[1])
    d       = [Int64(size(dataFiles[k])[2]) for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(dataFiles[k])[1] for k = 1:K])

    # Initialise the hyperparameters
    M = ones(K) .* 2 # Mass parameter
    γ = rand(Gamma(2.0 / N, 1), N, K) .+ eps(Float64) # Component weights
    Φ = K > 1 ? rand(Gamma(1, 5), Int64(K * (K - 1) * 0.5)) : zeros(1) # Dataset concordance measure

    # Initialise the allocations randomly according to γ
    s = Matrix{Int}(zeros(n_obs, K))
    for k = 1:K
        s[:, k] = sample(1:N, Weights(γ[:, k]), n_obs)
    end

    # Get a matrix of all combinations of gammas
    γ_combn = Matrix{Int64}(N ^ K, K)
    for (i, p) in enumerate(product([1:N for k = 1:K]...))
        γ_combn[i, :] = collect(p)
    end


    # Which Φ value is activated by each of the above combinations
    Φ_index = K > 1 ? Matrix{Int64}(N ^ K, Int64(K * (K - 1) / 2)) : ones(Matrix{Int64}(N, 1), Int64)
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_index[:, i] = (γ_combn[:, k1] .== γ_combn[:, k2])
                i += 1
            end
        end
    end

    Γ = Matrix{Float64}(N ^ K, K) # The actual gamma values for the gamma combinations
    for k = 1:K
        Γ[:, k] = log.(γ[:, k][γ_combn[:, k]])
    end
    Z = update_Z(Φ, Φ_index, Γ)

    logweight = zeros(Float64, particles)
    ancestor_weights = zeros(Float64, particles)
    sstar = zeros(Matrix{Int64}(particles, K), Int64)

    logprob = [zeros(Float64, N, particles) for k = 1:K]

    # Initialise the particles
    particle = [Vector{dataTypes[k]}(particles) for k = 1:K]
    particle_IDs = ones(Matrix{Int64}(particles, K), Int64)
    for k = 1:K
        particle[k][1] = dataTypes[k](dataFiles[k], N)
    end

    # Save information to file
    out = vcat(map(x -> @sprintf("MassParameter_%d", x), 1:K), map((x, y) -> @sprintf("phi_%d_%d", x, y), calculate_Φ_lab(K)[:, 1], calculate_Φ_lab(K)[:, 2]), map((x, y) -> @sprintf("K%d_n%d", x, y), repeat(1:K, inner = n_obs), repeat(1:n_obs, outer = K)))
    out =  reshape(out, 1, length(out))
    writecsv(outputFile, out)
    fileid = open(outputFile, "a")
    writecsv(fileid, [M; Φ; s[1:(n_obs * K)]]')


    for it in 1:iter
        # Reset particles and IDs
        for k = 1:K
                particle_reset!(particle[k][1])
        end
        particle_IDs .= Int64(1)
        #println(Z - sum(γ))

        v = update_v(n_obs, Z)
        update_M!(M, γ, K, N)

        Π = γ ./ sum(γ, 1)

        order_obs = randperm(n_obs)
        # Generate the new
        for k = 1:K
            for i in order_obs[1:floor(Int64, ρ * n_obs)]
                particle_add!(dataFiles[k][i, :], i, s[i, k], particle[k][1])
            end
        end

        ## The conditional particle filter
        for i in order_obs[(floor(Int64, ρ * n_obs) + 1):n_obs]
             for k = 1:K
                for p in 1:maximum(particle_IDs[:, k])
                    @inbounds calc_logprob!(dataFiles[k][i, :], particle[k][p], view(logprob[k], :, p))
                end
            end

            # update_logweight!(logweight, particle, particle_IDs, Π, K, N)
            ancestor_weights .= logweight .+ 0

            draw_sstar!(sstar, logprob, particle, particle_IDs, Π, K, N, ancestor_weights, logweight, s[i, :])


            # Set first particle to reference trajectory
            sstar[1, :] = s[i, :]

            #########################
            # Ancestor sampling step
            ########################
            # Take the fprob from each data at the ref trajectory's value
            # Add this to the particles current weight
            # Multinomially select index according to this
            ancestor_index = sample(1:particles, Weights(exp.(ancestor_weights .- maximum(ancestor_weights))))
            for k = 1:K
                 particle_IDs[1, k] .= particle_IDs[ancestor_index, k]
            end
            logweight[1] = logweight[ancestor_index]

            # Update the particle IDs
            new_particle_IDs = update_particleIDs(particle_IDs, sstar, K, particles, N)


            # Add observation to cluster
            for k = 1:K
                obs = dataFiles[k][i, :]
                id_match = ID_match(particle_IDs[:, k], new_particle_IDs[:, k], particles)
                # Doing this in reverse order means we don't need to copy the particles
                # Selecting the max value is quicker than checking for unique IDs
                # If the largest ID is n, then all IDs 1:n exist
                # If new_ID_1 > new_ID_2, then ID_2 ≧ ID_1
                for p in maximum(new_particle_IDs[:, k]):-1:1

                    @inbounds p_ind = findindex(new_particle_IDs[:, k], p)
                    if p == particle_IDs[p_ind, k]
                        particle_add!(obs, i, sstar[p_ind, k], particle[k][particle_IDs[p_ind, k]])
                    elseif p == id_match[particle_IDs[p_ind, k]]
                        particle_add!(obs, i, sstar[p_ind, k], particle[k][particle_IDs[p_ind, k]])
                        particle[k][p] = particle[k][particle_IDs[p_ind, k]]
                    else
                        particle[k][p] = deepcopy(particle[k][particle_IDs[p_ind, k]])
                        particle_add!(obs, i, sstar[p_ind, k], particle[k][p])
                    end
                end
            end


            particle_IDs = deepcopy(new_particle_IDs)
            Φ_upweight!(logweight, sstar, K, Φ)




            if calc_ESS(logweight) <= 0.5 * particles
                partstar = draw_partstar(logweight, particles)
                partstar[1] = Int64(1)
                logweight .= 1.0
                for k = 1:K
                    particle_IDs[:, k] = particle_IDs[partstar, k]
                end
                sstar .= Int64(1)
                particle_IDs = update_particleIDs(particle_IDs, 1, K, particles, N)
            end
        end
        # Select a single particle
        p_star = sample(1:particles, Weights(exp.(logweight .- maximum(logweight))))
        logweight .= 1.0
            for k = 1:K
                @inbounds s[:, k] = particle[k][particle_IDs[p_star, k]].c
                # println(s)
            end
            # Match up labels across datasets
            align_labels!(s, Φ, γ, N, K)
            # s[:, 1] = 11 - s[:, 2]
            update_Φ!(Φ, v, s, Φ_index, γ, K, Γ)
            update_γ!(γ, Φ, v, s, Φ_index, γ_combn, Γ, N, K)

            for k = 1:K
                for i = 1:(N ^ K)
                    Γ[i, k] = log(γ[γ_combn[i, k], k])
                end
            end

            Z = update_Z(Φ, Φ_index, Γ)
            writecsv(fileid, [M; Φ; s[1:(n_obs * K)]]')

        end
        close(fileid)
        return s
end
