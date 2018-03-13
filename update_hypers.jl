function update_v!(n_obs, Z)
    return rand(Gamma(n_obs, Z))
end

function update_M!(M, γ, N)
    # Update the mass parameter
    prior = [2, 0.5]
    for k in 1:length(γ)
        current_γ = γ[k]
        current_M = Float64(M[k])
        log_likelihood = - sum(logpdf.(Gamma(current_M / N, 1), current_γ))
        log_likelihood_0 = - sum(logpdf.(Gamma(prior[1], prior[2]), current_M))
        proposed_mass = (current_M + rand(Normal()) / 10)
        if proposed_mass <= 0.0
            alpha <- 0.0
        else
            new_log_likelihood = - sum(logpdf.(Gamma(proposed_mass / N, 1), current_γ))
            new_log_likelihood_0 = - sum(logpdf(Gamma.(prior[1], prior[2]), proposed_mass))
            alpha = exp(-new_log_likelihood - new_log_likelihood_0 + log_likelihood + log_likelihood_0)
        end
        if rand() < alpha
            M[k] = proposed_mass
        end
    end
    return
end


function update_Z!(Z, Φ, Φ_index, Γ)
    # Update the normalising constant
    Z = Float64(sum(exp.(Φ_index * log.(1 + Φ) + sum(Γ, 2))))
    return(Z)
end

function update_γ!(γ, Φ, v, s, Φ_index, γ_combn, Γ, N, K)
    α_0 = 1 / N
    β_0 = 1
    α_star = [α_0 + ([sum(s[k] .== n) for n = 1:N]) for k = 1:K]

    for n in 1:N
        for k in 1:K
            pertinent_rows = γ_combn[:, k] .== n
            β_star = β_0 + v * sum(exp.(Φ_index[pertinent_rows, :] * log.(1 + Φ) + sum(Γ[pertinent_rows, :], 2))) ./ γ[k][n]
            γ[k][n] = rand(Gamma(α_star[k][n], β_star)) + realmin(Float64)
        end
    end
    return
end


function update_Φ!(Φ, v, s, Φ_index, Φ_lab, γ, K)
    # Prior parameters
    α_0 = 1
    β_0 = 0.2
    if K == 1
        return 0
    end
    for i in 1:length(Φ)
        # Get relevant allocations
        current_allocations = [s[Φ_lab[i, 1]], s[Φ_lab[i, 2]]]
        Φ_current = Φ[i]
        n_agree = sum(current_allocations[1] .== current_allocations[2])
        # Get relevant terms in the normalisation constant Terms that include the current phi
        pertinent_rows = Φ_index[:, i] .== 1
        β_star = β_0 + v * sum(exp.(Φ_index[pertinent_rows, :] * log.(1 + Φ) + sum(Γ[pertinent_rows, :], 2))) ./ Φ_current

        weights = cumsum(log.(α_0 .+ [1, 1:n_agree...]))  # Add initial 1 to account for zero case
        weights += logpdf.(Binomial(n_agree, 0.5), 0:n_agree)
        weights -= cumsum((ones(1 +  n_agree) .* log(β_star)))
        weights -= maximum(weights)
        weights = exp.(weights)
        α_star = α_0 + sample(1:length(weights), Weights(weights)) - 1
        Φ[i] = rand(Gamma(α_star, β_star))
    end
    return
end
