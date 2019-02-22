using Clustering
using Compose
using LinearAlgebra
using Plots

mutable struct Posterior_similarity_matrix
    psm::Vector
    names::Vector{String}
    Posterior_similarity_matrix(K, n_obs) = new([Matrix(1.0I, n_obs, n_obs) for k in 1:(K + (K > 1))],
                                                Vector{String}(undef, K + (K > 1)))
end


"""
`generate_psm(outputFile::String, burnin::Int64 = 0, thin::Int64 = 1)`

Generates a posterior similarity matrix from pmdi output
## Input
- `outputFile::S    tring` a string referring to the location on disk of pmdi output
- `burnin::Int64` an integer of the number of initial iterations to discard as burn-in
- `thin::Int64` an integer for the rate at which to thin output, `thin = 2` discards
every second iteration

## Output
A struct of class `Posterior_similarity_matrix` containing:
    - `psm` K n × n matrices where element (i, j) measures how frequently observation
    i and j are co-clustered in dataset k. If `K > 1` an overall consensus matrix
    is also included which is an element-wise average of the K other matrices
    - `names` a vector of the dataset names
"""
function generate_psm(outputFile::String, burnin::Int64 = 0, thin::Int64 = 1)
    outputNames = split(readline(outputFile), ',')
    output = readdlm(outputFile, ',', header = false, skipstart = burnin + 1)
    K = sum(map(outputNames) do str
        occursin(r"MassParameter", str)
    end)

    output = output[1:thin:end, (K + binomial(K, 2) + (K == 1) + 2):end]

    n_obs = size(output, 2) / K
    @assert mod(n_obs, 1) == 0 "Error: Datasets have different number of observations"
    n_obs = Int64(n_obs)
    n_iter = size(output, 1)

    psm = Posterior_similarity_matrix(K, n_obs)
    psm.names[1:K] = unique(map(x -> split(x, '_')[1], outputNames[(K + (K == 1) + binomial(K, 2) + 2):end]))
    if K > 1
        psm.names[K + 1] = "Overall"
    end
    for k = 1:K
        for j = 1:(n_obs - 1)
            for i = (j + 1):n_obs
                psm.psm[k][i, j] = sum(output[:, i + n_obs * (k - 1)] .== output[:, j + n_obs * (k - 1)]) / n_iter
            end
        end
    end
    if K > 1
        for k = 1:K
            psm.psm[K + 1] += psm.psm[k] / K
        end
            psm.psm[K + 1][diagind(psm.psm[K + 1])] .= 1.0
    end

    return psm
end

function generate_psm2(output, K, n_obs)
    output = output[(K + binomial(K, 2) + (K == 1) + 2):end]

    n_iter = size(output, 1)

    psm = Posterior_similarity_matrix(K, n_obs)
    for k = 1:K
        for j = 1:(n_obs - 1)
            for i = (j + 1):n_obs
                psm.psm[k][i, j] = sum(output[i + n_obs * (k - 1)] .== output[j + n_obs * (k - 1)]) / n_iter
            end
        end
    end
    if K > 1
        for k = 1:K
            psm.psm[K + 1] += psm.psm[k] / K
        end
            psm.psm[K + 1][diagind(psm.psm[K + 1])] .= 1.0
    end
    return psm
end

"""
`get_consensus_allocations(psm::Posterior_similarity_matrix, nclust::Int64, orderby::Int64 = 0)`
"""
function get_consensus_allocations(psm::Posterior_similarity_matrix, nclust::Int64, orderby::Int64 = 0)
    if orderby == 0
        orderby = size(psm.psm, 1)
    end
    hc = hclust(1 .- Symmetric(psm.psm[orderby], :L), linkage = :complete, uplo = :L)
    return cutree(hc, k = nclust)
end



"""
`consensus_map(psm::Posterior_similarity_matrix, nclust::Int64, orderby::Int64 = 0)`

Generates a consensus map plot for a given posterior similarity matrix
## Input
- `psm` a posterior similarity matrix struct as output from `generate_psm`
- `nclust` the number of clusters to highlight in the data
- `orderby` observations in all plots are ordered in a way to separate clusters as well
as possible. `orderby` specifies which dataset should be used to inform this ordering.
`orderby = 0` will let the overall consensus dictate the ordering.
## Output
- K + 1 (or K if K == 1) consensus maps illustrating the clustering output in each
of K datasets.

**Note: the output of this needs to be saved to file to be viewed. This can be done
as follows:**
`using Gadfly
draw(SVG("path/to/file.svg"), consensus_map(psm, nclust, orderby))`
**This should be fixed pending a bug fix in the Gadfly packge**
"""
function consensus_map(psm::Posterior_similarity_matrix, nclust::Int64, orderby::Int64 = 0)
    if orderby == 0
        orderby = size(psm.psm, 1)
    end
    hc = hclust(1 .- Symmetric(psm.psm[orderby], :L), linkage = :complete, uplo = :L)
    cuts = cutree(hc, k = nclust)[hc.order]
    ticks = indexin(1:nclust, cuts) .- 0.5
    sort!(ticks)
    append!(ticks, size(psm.psm[1], 1) + 0.5)
    println(ticks)
    # Surely a better way of doing this
    ticks1 = [[ticks[i], ticks[i+1], NaN][j] for i = 1:nclust for j = 1:3]
    ticks2 = [[ticks[i], ticks[i], NaN][j] for i = 1:nclust for j = 1:3]
    ticks3 = [[ticks[i], ticks[i], NaN][j] for i = 2:(nclust + 1) for j = 1:3]
    order = sortperm(hc.order)
    K = length(psm.psm)
    plots = [Plots.heatmap(Symmetric(psm.psm[k], :L)[hc.order, hc.order],
                        ticks = false,
                        yflip = true,
                        title = psm.names[k],
                        legend = false,
                        aspect_ratio = 1,
                        c = :viridis,
                        titlefont = Plots.font(family = "serif", pointsize = 12)) for k in [K; 1:(K - 1)]]
    l = @layout [a{0.8w} grid((K - 1), 1)]
    Plots.plot(plots..., layout = l,
               left_margin= -5px,
               right_margin = -5px,
               bottom_margin = -5px,
               top_margin = -5px,
               title_location = :left)
    plot!(ticks1, ticks2, seriestype = :path, c = "#FFFFFF", linestyle = :dash)
    plot!(ticks1, ticks3, seriestype = :path, c = "#FFFFFF", linestyle = :dash)
    plot!(ticks2, ticks1, seriestype = :path, c = "#FFFFFF", linestyle = :dash)
    plot!(ticks3, ticks1, seriestype = :path, c = "#FFFFFF", linestyle = :dash,
    widen = false)

end
