abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("discretization/discretize_node.jl")
include("discretization/discretize_net.jl")
include("reduction/reduction_algorithm.jl")
include("reduction/new_routine.jl")
include("srp/srp.jl")
include("cpd/conditionalprobabilitydistribution.jl")
include("bayesnet.jl")

function evaluate!(ebn::EnhancedBayesianNetwork)
    ## transfer all possible continuous functional node's model to their discrete functional children
    new_ebn = deepcopy(ebn)
    new_ebn = transfer_continuous(new_ebn)
    ## determine functional nodes to evaluate
    functional_nodes = [i.name for i in filter(x -> isa(x, FunctionalNode), new_ebn.nodes)]
    e_ebn = deepcopy(new_ebn)
    ## evluate all functional nodes
    while !isempty(functional_nodes)
        e_ebn, evaluated_nodes = _evaluate_single_layer(e_ebn)
        functional_nodes = setdiff(functional_nodes, evaluated_nodes)
    end
    ## reduce obtained network 
    return reduce!(e_ebn)
end
