
function reduce!(net::EnhancedBayesianNetwork)
    if isempty(filter(x -> isa(x, FunctionalNode), net.nodes))
        cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
        map(x -> _eliminate_continuous_node!(net, x), cont_nodes)
    else
        evaluate!(net)
        cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
        map(x -> _eliminate_continuous_node!(net, x), cont_nodes)
    end
    return nothing
end


function dispatch_network(net::EnhancedBayesianNetwork)
    if isempty(filter(x -> isa(x, ContinuousNode), net.nodes))
        if all(.!_is_imprecise.(net.nodes))
            return BayesianNetwork(net.nodes, net.topology_dict, net.adj_matrix)
        else
            return CredalNetwork(net.nodes, net.topology_dict, net.adj_matrix)
        end
    else
        return net
    end
end

function _eliminate_continuous_node!(net::EnhancedBayesianNetwork, node::ContinuousNode)
    parents_indices = parents(net, node)[1]
    children_indices = children(net, node)[1]
    map((i, j) -> net.adj_matrix[i, j] = 1, parents_indices, children_indices)
    _remove_node!(net, node)
end