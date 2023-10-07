abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("discretization/discretize_root.jl")
include("discretization/discretize_child.jl")
include("discretization/discretize_net.jl")
include("reduction/reduction_algorithm.jl")
include("srp/srp.jl")
include("cpd/conditionalprobabilitydistribution.jl")
include("bayesnet.jl")
