abstract type ProbabilisticGraphicalModel end

include("cpd/conditionalprobabilitydistribution.jl")
include("enhancedbn.jl")
include("bayesnet.jl")
include("credalnet.jl")

include("discretization/discretize_node.jl")
include("discretization/discretize_net.jl")

include("reduction.jl")

include("trasmission/trasmission_node.jl")
include("trasmission/trasmission_net.jl")

include("evaluation/evaluate_node.jl")
include("evaluation/evaluate_net.jl")


include("../util/plots.jl")