function dispatch(net::EnhancedBayesianNetwork)
    ## If any ContinuousNode is present -> evaluate -> reduce
    if !isempty(filter(x -> isa(x, ContinuousNode), net.nodes))
        reduce!(net)
    end
    if isempty(filter(x -> isa(x, ContinuousNode), net.nodes))
        if all(isprecise.(net.nodes))
            return BayesianNetwork(net.nodes, net.topology_dict, net.adj_matrix)
        else
            return CredalNetwork(net.nodes, net.topology_dict, net.adj_matrix)
        end
    else
        return net
    end
end

dispatch(net::BayesianNetwork) = net
dispatch(net::CredalNetwork) = net