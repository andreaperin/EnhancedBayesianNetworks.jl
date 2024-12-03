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
include("inference/inferencestate.jl")
# include("inference/factors.jl")
# include("inference/factor_methods.jl")
# include("inference/factor_algebra.jl")
# include("inference/variableselimination.jl")
include("utils/evidence_verification.jl")
include("utils/wrap.jl")

