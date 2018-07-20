using ColorBrewer
using DataFrames
using Gadfly


function plot_phimatrix(outputfile::String)
    output = readdlm(outputfile, ',', header = true)
    phicolumns = contains.( output[2], "phi_")[:]
    nphis = length(phicolumns[phicolumns])

    # Check if K == 1

    K = Int64(0.5 + sqrt(8 * nphis + 1) * 0.5)
    phimatrix = Matrix{Float64}(K, K)
    phimatrix[:] = NaN

    count = 1
    for k1 = 1:(K - 1)
        for k2 = (k1 + 1):K
            phimatrix[k1, k2] = phimatrix[k2, k1] = mean(output[1][:, K + count])
            count += 1
        end
    end
    spy(phimatrix,
    Guide.xlabel("K1"),
    Guide.ylabel("K2"))
end

function plot_phichain(outputfile::String, burnin::Int64, thin::Int64)
    # output = readdlm(outputfile, ',', header = true)
    names = DataFiles.readtable(outputfile, nrows = 1)
    output = DataFiles.readtable(outputfile, skipstart = burnin)
    phicolumns = contains.( output[2], "phi_")[:]
    phivalues = DataFrame(output[1][(1):thin:size(output[1], 1), phicolumns])
    names!(phivalues, Symbol.(output[2][:, phicolumns][:]))
    phivalues[:Iteration] = collect((burnin + 1):thin:size(output[1], 1))

    plot(melt(phivalues), x = :Iteration, y = :value, color = :variable,
    Geom.line,
    Guide.xlabel("Iteration"),
    Guide.ylabel("Φ"))
end
