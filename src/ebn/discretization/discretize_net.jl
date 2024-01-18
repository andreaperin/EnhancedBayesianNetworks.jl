function discretize(ebn::EnhancedBayesianNetwork)
    nodes = ebn.nodes
    continuous_nodes = filter(n -> !isa(n, FunctionalNode), filter(n -> isa(n, ContinuousNode), nodes))
    evidence_nodes = filter(n -> !isempty(n.discretization.intervals), continuous_nodes)
    for n in evidence_nodes
        continuous_node, discretized_node = _discretize(n)

        # remove original continuous nodes
        nodes = filter(node -> node ∉ evidence_nodes, nodes)

        # update child nodes
        for node in nodes
            if isa(node, RootNode)
                continue
            end
            if n in node.parents
                node.parents[:] = [filter(x -> x !== n, node.parents)..., continuous_node]
            end
        end

        append!(nodes, [continuous_node, discretized_node])
    end

    return EnhancedBayesianNetwork(nodes)
end