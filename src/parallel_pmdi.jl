addprocs(Sys.CPU_CORES)
@everywhere include("particle_clust.jl")




arr = collect(1:100)
@time pmap(x -> x ^2 , arr )
