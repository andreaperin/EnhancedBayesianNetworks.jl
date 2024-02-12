function reduce(ebn::EnhancedBayesianNetwork)
    nodes = ebn.nodes
    continuous_nodes = filter(x -> isa(x, ContinuousNode), ebn.nodes)
    to_reduce = filter(x -> !isa(x, FunctionalNode), continuous_nodes)
    to_reduce_indices = map(x -> ebn.name_to_index[x.name], to_reduce)
    if _is_reducible(ebn.dag, to_reduce_indices)
        for node in to_reduce
            nodes = _reduce_node(ebn, node)
        end
        return EnhancedBayesianNetwork(nodes)
    else
        error("eBN cannot be reduced")
    end
end

function _reduce_node(ebn::EnhancedBayesianNetwork, node_to_reduce::AbstractNode)
    nodes = filter(x -> x != node_to_reduce, ebn.nodes)
    parents = node_to_reduce.parents
    for node in nodes
        if isa(node, RootNode)
            continue
        end
        if node_to_reduce in node.parents
            for n in [node_to_reduce, parents...]
                index = findfirst(x -> isequal(x, n), node.parents)
                deleteat!(node.parents, index)
            end
            append!(node.parents, parents)
        end
    end
    return nodes
end