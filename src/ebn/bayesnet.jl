struct BayesianNetwork <: ProbabilisticGraphicalModel
    dag::SimpleDiGraph
    nodes::Vector{<:DiscreteNode}
    name_to_index::Dict{Symbol,Int}
end

function BayesianNetwork(rbns::Vector{ReducedBayesianNetwork})
    for rbn in rbns
        functional_nodes = filter(x -> isa(x, DiscreteFunctionalNode), rbn.nodes)
        any(isempty.([i.pf for i in functional_nodes])) && error("rbn needs to evaluated!")


    end
end


function DiscreteStandardNode(node::DiscreteFunctionalNode)
    states = OrderedDict{Vector{Symbol},Dict{Symbol,Float64}}()
    for (k, v) in node.pf
        states[k] = Dict(:f => v, :s => 1 - v)
    end
    DiscreteStandardNode(node.name, node.parents, states)
end

