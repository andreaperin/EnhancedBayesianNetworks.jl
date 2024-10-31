using Test

using EnhancedBayesianNetworks
using Graphs
using Suppressor

include("utils/probabilities_verification.jl")
include("nodes/new_root.jl")
include("nodes/new_child.jl")
include("ebn/new_enhancedbn.jl")
include("ebn/transmission/transmission.jl")
include("ebn/discretization/discretize.jl")
include("ebn/reduction/reduction.jl")
include("ebn/evaluate/evaluate_node.jl")


# include("nodes/nodes.jl")
# include("nodes/root.jl")
# include("nodes/child.jl")
# include("nodes/functional.jl")
# include("nodes/common.jl")

# include("ebn/enhancedbn.jl")
# include("ebn/reduction.jl")
# include("ebn/discretization/discretize_node.jl")
# include("ebn/discretization/discretize_net.jl")
# include("ebn/transmission/transmission_node.jl")
# include("ebn/transmission/transmission_net.jl")
# include("ebn/evaluate/evaluate_node.jl")
# include("ebn/evaluate/evaluate_net.jl")
# include("ebn/bayesnet.jl")
# include("ebn/credalnet.jl")
# include("ebn/cpd/conditionalprobabilitydistribution.jl")

# include("inference/inferencestate.jl")
# include("inference/factors.jl")
# include("inference/factor_methods.jl")
# include("inference/factor_algebra.jl")
# include("inference/variableselimination.jl")

# include("utils/evidence_verification.jl")
# include("utils/node_verification.jl")
# include("utils/plots.jl")
# include("utils/wrap.jl")