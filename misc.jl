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


function calc_fprob!(fprob, logprob, particle_IDs, K, Π)
    for k = 1:K
        for p in unique(particle_IDs[:, k])
            fprob[:, p, k] = Π[:, k] .* exp.(logprob[:, p, k] .- maximum(logprob[:, p, k]))
        end
    end
    return
end

function update_logweight!(logweight, particle, particle_IDs, Π, K, N)
    for k = 1:K
        particle_k = particle[k]
        for p in unique(particle_IDs[:, k])
            fprob = Π[:, k] .* exp.(particle_k[p].ζ .- maximum(particle_k[p].ζ))
            # logweight[findin(particle_IDs[:, k], p)] += log(sum(fprob[:, p, k])) + maximum(logprob[:, p, k])
            logweight[findin(particle_IDs[:, k], p)] += log(sum(fprob)) + maximum(particle_k[p].ζ)
            # logweight[findin(particle_IDs[:, k], p)] .= logweight[p]
        end
    end
    return
end

function draw_sstar!(sstar, particle, particle_IDs, Π, K, particles, N)
    for k = 1:K
        particle_k = particle[k]
        for p = 1:particles
            fprob = Π[:, k] .* exp.(particle_k[particle_IDs[p, k]].ζ .- maximum(particle_k[particle_IDs[p, k]].ζ))
            sstar[p, k] = sample(1:N, Weights(fprob), 1)[1]
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
    return sum(exp.(logweight .- maximum(logweight))) ^ 2 / sum(exp.(logweight .- maximum(logweight)) .^ 2)
end


function draw_partstar(logweight, particles)
    pprob = cumsum(exp.(logweight .- maximum(logweight)))
    pprob /= last(pprob)
    u = collect(0:(1 / particles):(particles -1) / particles) .+ rand() / particles

    return map(x -> indmax(pprob .> x), u)
end

function Φ_upweight!(logweight, particle_IDs, sstar, K, Φ)
    if K == 1
        return logweight
    else
        Φ_lab = calculate_Φ_lab(K)
        for i = 1:Int64((K * (K - 1) * 0.5))
            logweight += (sstar[:, Φ_lab[i, 1]] .== sstar[:, Φ_lab[i, 2]]) * log(1 + Φ[i])
        end
    end
end

function Φ_involvement(K::Int64, k::Int64)
    Φ_lab = calculate_Φ_lab(K)
    Φ_involved = (Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)
    return Φ_involved
end


function align_labels!(s, Φ, γ, N, K)
    # current_labels = unique(s)
    if K == 1
        return
    else
        Φ_lab = calculate_Φ_lab(K)
        Φ_log = log.(1 + Φ)
        for k = 1:K
            if length(unique(s[:, k])) > 1
                relevant_Φs = Φ_log[(Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)]
                # relevant_Φs = sort(union(findin(Φ_lab[:, 1], k),  findin(Φ_lab[:, 2], k)))
                for i in 1:(length(unique(s[:, k])) - 1)
                    label = unique(s[:, k][i])[1]
                    new_label = sample(unique(s[:, k])[(i+1):end])

                    # new_label = rand(setdiff(unique(s[:, k]), label))

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
        end
    end
end
