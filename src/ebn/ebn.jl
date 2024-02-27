abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("discretization/discretize_node.jl")
include("discretization/discretize_net.jl")
include("reduction.jl")
include("trasmission/trasmission_node.jl")
include("trasmission/trasmission_net.jl")
include("evaluation/evaluate_node.jl")
include("evaluation/evaluate_net.jl")

include("cpd/conditionalprobabilitydistribution.jl")
include("bayesnet.jl")