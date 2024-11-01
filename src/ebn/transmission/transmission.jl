function _transfer_continuous!(net::EnhancedBayesianNetwork)
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), net.nodes)
    for c in continuous_functional
        _transfer_single_continuous_functional!(net, c)
    end
    return nothing
end

function _transfer_single_continuous_functional!(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode)
    node_children = get_children(net, node)[3]
    if isempty(node.discretization.intervals) && !isempty(node_children)
        node_parents = get_parents(net, node)[3]
        map(ch -> prepend!(ch.models, node.models), node_children)
        sim_incoherence = map(ch -> typeof(ch.simulation) != typeof(node.simulation), node_children)
        if any(sim_incoherence)
            inchoerent_children_names = [i.name for i in node_children[sim_incoherence]]
            inchoerent_children_simulations = typeof.([i.simulation for i in node_children[sim_incoherence]])
            error("node $(node.name) cannot be transferred into his children => $inchoerent_children_names, because its simulation type: $(typeof(node.simulation)) is non coherent with children simulation types: $inchoerent_children_simulations")
        end
        _remove_node!(net, node)

        function _add_all_children_single_parent(par::AbstractNode, chs::AbstractVector{<:AbstractNode})
            map(ch -> add_child!(net, par, ch), chs)
        end

        map(par -> _add_all_children_single_parent(par, node_children), node_parents)
        return order_net!(net)
    end
end