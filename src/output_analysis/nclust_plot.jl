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
function plot_nclust(outputFile::String, burnin::Int64, thin::Int64)
    outputNames = split(readline(outputFile), ',')
    phiColumns = ismatch.(r"phi_", outputNames)
    output = readcsv(outputFile, header = false, skipstart = burnin + 1)

    phiValues = DataFrame(output[1:thin:size(output, 1), phiColumns])
    names!(phiValues, Symbol.(outputNames[phiColumns]))
    phiValues[:Iteration] = collect((1):thin:size(output, 1))

    plot(melt(phiValues), x = :Iteration, y = :value, color = :variable,
    Geom.line,
    Guide.xlabel("Iteration"),
    Guide.ylabel("Φ"),
    Coord.Cartesian(xmax = maximum(phivalues[:Iteration])))
end
