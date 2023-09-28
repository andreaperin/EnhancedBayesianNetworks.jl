struct ReducedBayesianNetwork <: ProbabilisticGraphicalModel
    dag::SimpleDiGraph
    nodes::Vector{<:AbstractNode}
    name_to_index::Dict{Symbol,Int}
end

function ReducedBayesianNetwork(nodes_::Vector{<:AbstractNode})
    nodes = deepcopy(nodes_)
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    rbn = ReducedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
    return rbn
end

function get_children(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in outneighbors(ebn.dag, i)]
end

function get_parents(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in inneighbors(ebn.dag, i)]
end

function get_neighbors(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in unique(append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i)))]
end