using Test

using EnhancedBayesianNetworks
using Graphs
using Suppressor

include("nodes/nodes.jl")
include("nodes/discretenodes.jl")
include("nodes/continuousnode.jl")
include("nodes/functional.jl")
include("ebn/enhancedbn.jl")
include("ebn/transmission/transmission.jl")
include("ebn/discretization/discretize.jl")
include("ebn/evaluate/evaluate_node.jl")