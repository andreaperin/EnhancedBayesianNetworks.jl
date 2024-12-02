abstract type AbstractNetwork end

# include("../util/node_verification.jl")
include("../util/verification_add_child.jl")
include("enhancedbn.jl")
include("transmission/transmission.jl")
include("discretization/discretize.jl")
include("evaluation/evaluate_node.jl")
include("evaluation/evaluate_net.jl")
# include("reduction/reduction.jl")
# include("evaluation/evaluate_net.jl")

# include("cpd/conditionalprobabilitydistribution.jl")
# include("bayesnet.jl")
# include("credalnet.jl")
