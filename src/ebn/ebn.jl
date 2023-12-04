abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("discretization/discretize_node.jl")
include("discretization/discretize_net.jl")
include("reduction/reduction_algorithm.jl")
include("reduction/new_routine.jl")
include("srp/srp.jl")
include("cpd/conditionalprobabilitydistribution.jl")
include("bayesnet.jl")
