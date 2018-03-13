function calculate_Φ_lab(K::Int64)
    Φ_lab = K > 1 ? Matrix{Int64}(Int64(K * (K - 1) / 2), 2) : 1
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
