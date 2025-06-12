module SpikeTrainUtilities

#
# External dependencies
#
using Colors
using LinearAlgebra
using RollingFunctions
using Statistics, StatsBase
using Random

#
# Internal code
#
include("types.jl")        # defines types
include("processing.jl")   # processes spike trains
include("metrics.jl")      # statistics and metrics
include("generating.jl")   # generates randm spike trains for testing and examples
include("plotting.jl")     # processes data for plotting

#
# Exports
#
export
    SpikeTrains,
    binned,
    isi_hist,
    psth,
    rasterplot,
    spike_distance


end # module