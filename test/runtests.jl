using Test

using EnhancedBayesianNetworks
using Graphs
using Suppressor

include("nodes/nodes.jl")
include("nodes/root.jl")
include("nodes/child.jl")
include("nodes/functional.jl")
include("nodes/common.jl")

include("ebn/enhancedbn.jl")
include("ebn/reduction.jl")
include("ebn/discretization/discretize_node.jl")
include("ebn/discretization/discretize_net.jl")
include("ebn/trasmission/trasmission_node.jl")
include("ebn/trasmission/trasmission_net.jl")
include("ebn/evaluate/evaluate_node.jl")
include("ebn/evaluate/evaluate_net.jl")
include("ebn/bayesnet.jl")
include("ebn/credalnet.jl")
include("ebn/cpd/conditionalprobabilitydistribution.jl")


include("inference/inferencestate.jl")
include("inference/factors.jl")
include("inference/factor_methods.jl")
include("inference/factor_algebra.jl")
include("inference/variableselimination.jl")