
function reduce!(net::EnhancedBayesianNetwork, check::Bool=true, collect_samples::Bool=true)
    if !isempty(filter(x -> isa(x, FunctionalNode), net.nodes))
        evaluate!(net, check, collect_samples)
    end
    cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
    map(x -> _eliminate_continuous_node!(net, x), cont_nodes)
    return nothing
end

function _eliminate_continuous_node!(net::EnhancedBayesianNetwork, node::ContinuousNode)
    parents_indices = parents(net, node)[1]
    children_indices = children(net, node)[1]
    map((i, j) -> net.adj_matrix[i, j] = 1, parents_indices, children_indices)
    _remove_node!(net, node)
end