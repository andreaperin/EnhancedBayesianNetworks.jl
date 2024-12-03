using Test

using EnhancedBayesianNetworks
using Graphs
using Suppressor

include("nodes/nodes.jl")
include("nodes/discretenodes.jl")
include("nodes/continuousnode.jl")
include("nodes/functional.jl")
include("networks/networks.jl")
include("networks/ebn/enhancedbn.jl")
include("networks/ebn/transmission/transmission.jl")
include("networks/ebn/discretization/discretize.jl")
include("networks/ebn/evaluate/evaluate_node.jl")
include("networks/ebn/evaluate/evaluate_net.jl")
include("networks/ebn/reduction/reduction.jl")
include("networks/bn/bayesnet.jl")
include("networks/cn/credalnet.jl")