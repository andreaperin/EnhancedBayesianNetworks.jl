using Test

using EnhancedBayesianNetworks
using Graphs

include("nodes/root.jl")
include("nodes/standard.jl")
include("nodes/functional.jl")
include("ebn/enhancedbn.jl")
include("ebn/reducedbn.jl")
include("ebn/discretization.jl")
include("ebn/bayesiannetwork.jl")
include("Inference/inferencestate.jl")
include("Inference/factors.jl")
include("Inference/factor_methods.jl")