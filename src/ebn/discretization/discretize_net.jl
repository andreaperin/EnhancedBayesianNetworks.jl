function discretize!(net::EnhancedBayesianNetwork)
    ebn = deepcopy(net)
    nodes = ebn.nodes
    continuous_nodes = filter(j -> !isa(j, FunctionalNode), (filter(x -> isa(x, ContinuousNode), nodes)))
    a = isempty.([i.discretization.intervals for i in continuous_nodes])
    evidence_node = continuous_nodes[.!a]
    while !isempty(evidence_node)
        if isa(evidence_node[1], RootNode)
            nodes = _discretize_node(ebn, evidence_node[1])
            ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
            ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
        elseif isa(evidence_node[1], ChildNode)
            nodes = _discretize_node(ebn, evidence_node[1])
            ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
            ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
        end
        popfirst!(evidence_node)
    end
    return ebn
end