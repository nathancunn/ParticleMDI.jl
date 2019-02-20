using DataFrames
using Gadfly

"""
`plot_phimatrix(outputfile::String, burnin::Int64, thin::Int64)`

Plots the mean phi values resulting from pmdi output
## Input
- `outputFile::String` a string referring to the location on disk of pmdi output
- `burnin::Int64` an integer of the number of initial iterations to discard as burn-in
- `thin::Int64` an integer for the rate at which to thin output, `thin = 2` discards
every second iteration

## Output
Outputs a heatmap of pairwise dataset Φ values
"""
function plot_phi_matrix(outputFile::String, burnin::Int64, thin::Int64)
    outputNames = split(readline(outputFile), ',')
    phiColumns = map(outputNames) do str
        occursin(r"phi_", str)
    end
    # phiColumns = occursin.(r"phi_", outputNames)
    output = readdlm(outputFile, ',', header = false, skipstart = burnin + 1)

    phiValues = DataFrame(output[1:thin:end, phiColumns])
    names!(phiValues, Symbol.(outputNames[phiColumns]))
    phiValues[:Iteration] = collect((1):thin:size(output, 1))
    nphis = sum(phiColumns)

    K = Int64(0.5 + sqrt(8 * nphis + 1) * 0.5)
    @assert K > 1 "Φ not inferred for no. of datasets = 1"
    phiMatrix = Matrix{Float64}(undef, K, K)
    phiMatrix .= NaN

    i = 1
    for k1 = 1:(K - 1)
        for k2 = (k1 + 1):K
            phiMatrix[k1, k2] = phiMatrix[k2, k1] = Statistics.mean(output[:, K + i])
            i += 1
        end
    end
    spy(phiMatrix,
    Guide.xticks(ticks = [1:K;]),
    Guide.yticks(ticks = [1:K;]),
    Guide.xlabel("Φ(x, ⋅)"),
    Guide.ylabel("Φ(⋅, y)"),
    Scale.color_continuous_gradient(colormap = Scale.lab_gradient("#440154", "#1FA187", "#FDE725")))
end



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
function plot_phi_chain(outputFile::String, burnin::Int64, thin::Int64)
    outputNames = split(readline(outputFile), ',')
    phiColumns = map(outputNames) do str
        occursin(r"phi_", str)
    end
    # phiColumns = occursin.(r"phi_", outputNames)
    output = readdlm(outputFile, ',', header = false, skipstart = burnin + 1)

    phiValues = DataFrame(output[1:thin:end, phiColumns])
    names!(phiValues, Symbol.(outputNames[phiColumns]))
    phiValues[:Iteration] = collect((1):thin:size(output, 1))

    plot(melt(phiValues), x = :Iteration, y = :value, color = :variable,
    Geom.line,
    Guide.xlabel("Iteration"),
    Guide.ylabel("Φ"),
    Coord.Cartesian(xmax = maximum(phiValues[:Iteration])))
end
