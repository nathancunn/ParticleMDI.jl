"""
`plot_phichain(outputfile::String, burnin::Int64, thin::Int64)`

Plots the phi values at each iteration resulting from pmdi output
## Input
- `outputFile::String` a string referring to the location on disk of pmdi output
- `burnin::Int64` an integer of the number of initial iterations to discard as burn-in
- `thin::Int64` an integer for the rate at which to thin output, `thin = 2` discards
every second iteration

## Output
Outputs a line plot of phi values resulting from pmdi output
"""
function plot_nclust_hist(outputFile::String, burnin::Int64 = 0, thin::Int64 = 1)
    outputNames = split(readline(outputFile), ',')
    K = sum(map(x -> occursin(r"MassParameter", x), outputNames))
    output = readdlm(outputFile, ',', header = false, skipstart = burnin + 1)
    hyperCols = Int64(K * (K - 1) / 2 + K + 1 + (K == 1))
    output = output[1:thin:end, (hyperCols + 1):end]
    n_obs = Int64((size(output, 2)) / K)
    dataNames = unique(map(x -> split(x, "_")[1], outputNames[(hyperCols + 1):end]))

    out = Matrix(undef, size(output, 1) * K, 3)

    endCol = 0
    for k in 1:K
        startCol = endCol + 1
        endCol = startCol + n_obs - 1
        for i in 1:size(output, 1)
            out[i + size(output, 1) * (k - 1), 1] = dataNames[k]
            out[i + size(output, 1) * (k - 1), 2] = i
            out[i + size(output, 1) * (k - 1), 3] = length(unique(output[i, startCol:endCol]))
        end
    end

    plotdf = DataFrame(out)
    names!(plotdf, [:Dataset, :Iteration, :nclust])

    plot(plotdf, x = :nclust,
    color = :Dataset,
    alpha = 0.3,
    Geom.histogram,
    Guide.ylabel("Frequency"),
    Guide.xlabel("Number of clusters"))
end
