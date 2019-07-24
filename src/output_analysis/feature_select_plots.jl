using Clustering
using LinearAlgebra
using Plots
using Plots.PlotMeasures

"""
`plot_pmdi_data(dataFile,
                psm::Posterior_similarity_matrix;
                k::Union{Int64, Nothing} = nothing,
                h::Union{Float64, Nothing} = nothing,
                orderby::Int64 = 0,
                featureSelectProbs = nothing,
                z_Score::Bool = false)`

Generates a plot of a single raw data file, with the inferred clusters (as derived from the psm). If featureSelectProbs is supplied, features will be ordered according to the frequency they are selected in the model and the density of feature selection probabilities will be displayed along the margins.
## Input
- `psm` a posterior similarity matrix struct as output from `generate_psm`
- `k` optional - the number of clusters to highlight in the data
- `h` optional - the distance at which to cut the dendrogram. One of `k` or `h` must be specified.
- `orderby` observations in all plots are ordered in a way to separate clusters as well as possible. `orderby` specifies which dataset should be used to inform this ordering. `orderby = 0` will let the overall consensus dictate the ordering.
- `featureSelect` optional - a Vector of feature selection probabilities as output from `get_feature_select_probs()`.
- `z_score` - a Bool indicating whether the data should be standardised and discretised. Can help elucidate patterns.

## Output
- A heatmap of the data with clusters indicated.
"""
function plot_pmdi_data(data,
                        psm::Posterior_similarity_matrix;
                        k::Union{Int64, Nothing} = nothing,
                        h::Union{Float64, Nothing} = nothing,
                        orderby::Int64 = 0,
                        featureSelectProbs = nothing,
                        z_score::Bool = false,
                        linkage = :ward)
  dataFile = deepcopy(data)
  if featureSelectProbs != nothing
    @assert length(featureSelectProbs) == size(dataFile, 2) "Feature selection vector is not the same length as the number of features"
  end
  if (z_score) & (typeof(dataFile) == Array{Float64, 2})
    sd = std(dataFile, dims = 1)
    dataFile .-= mean(dataFile, dims = 1)
    dataFile ./= sd
    dataFile .= floor.(Int64, dataFile)
    dataFile[dataFile .> 2] .= 2
    dataFile[dataFile .< - 2] .= - 2
  end


  # Extract clusters
  if orderby == 0
      orderby = size(psm.psm, 1)
  end
  hc = hclust(1 .- Symmetric(psm.psm[orderby], :L), linkage = linkage)
  if h == nothing
    cuts = cutree(hc, k = k)[hc.order]
  elseif k == nothing
    cuts = cutree(hc, h = h)[hc.order]
  end

  nclust = length(unique(cuts))
  ticks = indexin(1:nclust, cuts) .- 0.5
  sort!(ticks)
  ticks = ticks[2:end]
  order_rows = hc.order



  if featureSelectProbs != nothing
    order_cols = sortperm(featureSelectProbs, rev = true)
    fs_plot = Plots.bar(featureSelectProbs[order_cols] .* size(data, 1),
                        legend = false,
                        fill = "#000000",
                        linealpha = 0,
                        yflip = true,
                        tickfontsize = 1,
                        xticks = false,
                        grid = false,
                        widen = false,
                        top_margin = - 7.5px,
                        left_margin = 0px,
                        right_margin = - 7.5px,
                        xlims = (0.5, size(dataFile, 2) - 0.5),
                        xlabel = "Features",
                        ylabel = "P(select)")
    l = @layout [a{0.97w, 0.8h} b{0.03w, 0.8h}; c{0.97w, 0.2h} d{0.03w, 0.2h}]
  else
    l = @layout [a{0.97w, 0.97h} b{0.03w, 0.97h};]
    order_cols = 1:size(dataFile, 2)
  end

  # clusters_matrix = Matrix{Int64}(undef, size(dataFile, 1), 1)
  # clusters_matrix .= cuts

  clust_mat = Matrix{Int64}(undef, nclust, 2)
  clust_mat[:, 1] =  unique(cuts)
  tmp = cumsum(map(x -> count(cuts .== x), clust_mat[:, 1]))
  clust_mat[:, 2] = tmp[end:-1:1]
  labs_y = vcat(tmp[1] / 2,
              tmp[1:end - 1] + (tmp[2:end] - tmp[1:end-1]) ./ 2)
  C(n) = [RGB(cgrad(:viridis)[floor(Int, z)]) for z in collect(LinRange(1, 30, n))]
  clusters_plot = plot(palette = C(nclust))
  bar!(clust_mat[:, 2]',
       seriescolor = clust_mat[:, 1]',
          legend = false,
          yflip = false,
          widen = false,
          yaxis = false,
          xticks = false,
          tickfontsize = 1,
          bottom_margin = -7.5px,
          left_margin = - 7.5px,
          annotation = (1, labs_y, clust_mat[end:-1:1, 1]))


  #clusters_plot = Plots.heatmap(clusters_matrix,
  #tickfontsize = 1,
  #legend = false,
  #yaxis = false,
  #xticks = false,
  #palette = :viridis,
  #bottom_margin = -7.5px,
  #left_margin = - 7.5px)
  #hline!(ticks, palette = "#000000")


  clusters_plot2 = Plots.bar(clust_mat[:, 2]',
  tickfontsize = 1,
  legend = false,
  widen = false,
  α = 0,
  top_margin = - 7.5px,
  left_margin = - 7.5px
  )

  data_plot = Plots.heatmap(dataFile[order_rows, order_cols],
  tickfontsize = 1,
  xticks = false,
  yflip = false,
  legend = false,
  widen = false,
  left_margin = 0px,
  bottom_margin = -7.5px,
  right_margin = -7.5px,
  ylabel = "Observations",
  c = :viridis,
  xlims = (0.5, size(dataFile, 2) - 0.5)
  )
  hline!(ticks, linestyle = :dash, c = "#FFFFFF")

  if featureSelectProbs != nothing
    out = [data_plot, clusters_plot, fs_plot, clusters_plot2]
  else
    xlabel!("Features")
    out = [data_plot, clusters_plot]
  end


    Plots.plot(out...,
    layout = l,
    link = :both,
    framestyle = :none,
    right_margin = -5px,
    tickfontsize = 1)
end


"""
`get_feature_select_probs(featureSelect::String, dataName::String)`

Generates a vector of mean feature selection probabilities from the feature selection output of `pmdi()`.
## Input
- `featureSelect::String` - a file specifying a .csv file containing feature selection flags as returned from `pmdi()`
- `burnin::Int` - an integer specifying the number of iterations to discard from the beginning of the output
- `thin::Int` - an integer specifying the rate of thinning of the output, e.g. `thin = 2` will keep only every second iteration.
## Output
- A Vector of length `K`, where `K` is the number of datasets analysed, each element containing a feature selection probability for each feature.
"""
function get_feature_select_probs(featureSelect::String, burnin::Int = 0, thin::Int = 1)
  outputNames = split(readline(featureSelect), ',')
  dataNames = unique([replace(outputName, r"([A-Za-z0-9])(_d.+)" => s"\1") for outputName in outputNames])

  out = Vector{Any}(undef, length(dataNames))
  for i in 1:length(out)
      dataInd = map(outputNames) do str
          occursin(dataNames[i], str)
      end
      out[i] = [mean(readdlm(featureSelect, ',')[(2 + burnin):thin:end, dataInd], dims = 1)[:]][1]
  end
  return out
end
