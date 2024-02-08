abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("discretization/discretize_node.jl")
include("discretization/discretize_net.jl")
include("reduction/dag_reduction.jl")
include("reduction/ebn_reduction.jl")
include("srp/srp.jl")
include("cpd/conditionalprobabilitydistribution.jl")
include("bayesnet.jl")