function calculate_Φ_lab(K::Int64)
    Φ_lab = K > 1 ? Matrix{Int64}(Int64(K * (K - 1) / 2), 2) : ones(1)
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_lab[i, :] = [k1, k2]
                i += 1
            end
        end
    end
    return Φ_lab
end

# Not needed anymore
function calc_fprob!(fprob::Array, logprob, particle_IDs, K, Π)
    for k = 1:K
        for p in unique(particle_IDs[:, k])
            fprob[:, p, k] = Π[:, k] .* exp.(logprob[:, p, k] .- maximum(logprob[:, p, k]))
        end
    end
    return
end

function update_logweight!(logweight::Array, particle::Array, particle_IDs::Array, Π::Array, K::Int64, N::Int64)
    for k = 1:K
        particle_k = particle[k]
        for p in unique(particle_IDs[:, k])
            fprob = Π[:, k] .* exp.(particle_k[p].ζ .- maximum(particle_k[p].ζ))
            logweight[findin(particle_IDs[:, k], p)] .+= log(sum(fprob)) + maximum(particle_k[p].ζ)
        end
    end
    return
end

function draw_sstar!(sstar::Array, particle::Array, particle_IDs::Array, Π::Array, K::Int64, N::Int64)
    for k = 1:K
        π_k = Π[:, k]
        ids_k = particle_IDs[:, k]
        for p = 1:maximum(ids_k)
            particle_flag = findin(ids_k, p)
            fprob = π_k .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ))
            sstar[particle_flag, k] = sample(1:N, Weights(fprob), length(particle_flag))
        end
    end
end


function update_particleIDs!(particle_IDs, sstar, K, particles, N)
    return ID_to_canonical!(particle_IDs * (particles * N) + sstar)
end

function ID_to_canonical!(x)
    for k in 1:size(x, 2)
        x[:, k] = indexin(x[:, k], unique(x[:, k]))
    end
    return x
end


function calc_ESS(logweight)
    return sum(exp.(logweight .- maximum(logweight))) .^ 2 / sum(exp.(logweight .- maximum(logweight)) .^ 2)
end


function draw_partstar(logweight, particles)
    pprob = cumsum(exp.(logweight .- maximum(logweight)))
    pprob ./= last(pprob)
    u = collect(0:(1 / particles):(particles -1) / particles) .+ (rand() / particles)
    return map(x -> indmax(pprob .> x), u)
end

function Φ_upweight!(logweight::Array, sstar::Array, K::Int64, Φ::Array)
    if K == 1
        return logweight
    else
        Φ_lab = calculate_Φ_lab(K)
        for i = 1:Int64((K * (K - 1) * 0.5))
            logweight .+= (sstar[:, Φ_lab[i, 1]] .== sstar[:, Φ_lab[i, 2]]) .* log(1 + Φ[i])
        end
    end
end


function Φ_involvement(K::Int64, k::Int64)
    Φ_lab = calculate_Φ_lab(K)
    Φ_involved = (Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)
    return Φ_involved
end


function align_labels!(s::Array, Φ::Array, γ::Array, N::Int64, K::Int64)
    # current_labels = unique(s)
    if K == 1
        return
    else
        Φ_lab = calculate_Φ_lab(K)
        Φ_log = log.(1 + Φ)
        unique_s = unique(s)
        #if length(unique_s) > 1
            for k = 1:K
                unique_sk = unique(s[:, k])
                relevant_Φs = Φ_log[(Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)]
                # for i in 1:(length(unique_sk))
                for label in 1:N
                    # label = unique_sk[i]
                    new_label = sample(setdiff(unique_s, label))

                    label_ind = findin(s[:, k], label)
                    new_label_ind = findin(s[:, k], new_label)

                    label_rows      = s[label_ind, setdiff(1:K, k)]
                    new_label_rows  = s[new_label_ind, setdiff(1:K, k)]
                    log_phi_sum     = sum(sum(label_rows .== label, 1) .* relevant_Φs + sum(new_label_rows .== new_label, 1) .* relevant_Φs)
                    log_phi_sum_swap = sum(sum(label_rows .== new_label, 1) .* relevant_Φs + sum(new_label_rows .== label, 1) .* relevant_Φs)

                    accept = min(1, exp(log_phi_sum_swap - log_phi_sum))

                    if rand() < accept
                        s[label_ind, k]        .= new_label
                        s[new_label_ind, k]    .= label
                        γ_temp = γ[new_label, k]
                        γ[new_label, k] = γ[label, k]
                        γ[label, k] = γ_temp
                    end
                end
            end
        # end
    end
end
