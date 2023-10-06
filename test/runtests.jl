using Test

using EnhancedBayesianNetworks
using Graphs

include("nodes/root.jl")
include("nodes/child.jl")
include("nodes/functional.jl")
include("nodes/structuralreliabilityproblemnode.jl")
include("nodes/node.jl")

include("ebn/enhancedbn.jl")

include("ebn/discretization/discretization.jl")
include("ebn/discretization/discretize_root.jl")
include("ebn/discretization/discretize_child.jl")
include("ebn/discretization/discretize_net.jl")

include("ebn/reduction/reduction_alghorithm.jl")

include("ebn/srp/build_srp.jl")
include("ebn/srp/evaluate_srp.jl")

# include("ebn/bayesiannet.jl")
# include("Inference/inferencestate.jl")
# include("Inference/factors.jl")
# include("Inference/factor_methods.jl")
# include("Inference/factor_algebra.jl")
# include("Inference/variableselimination.jl")