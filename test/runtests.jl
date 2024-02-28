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

# include("ebn/reduction/dag_reduction.jl")
# include("ebn/reduction/ebn_reduction.jl")

# include("ebn/srp/evaluate_srp.jl")
# include("ebn/srp/evaluate_net.jl")

# include("ebn/cpd/conditionalprobabilitydistribution.jl")
# include("ebn/bayesiannet.jl")

# include("Inference/inferencestate.jl")
# include("Inference/factors.jl")
# include("Inference/factor_methods.jl")
# include("Inference/factor_algebra.jl")
# include("Inference/variableselimination.jl")