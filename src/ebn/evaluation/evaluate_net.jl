function evaluate!(net::EnhancedBayesianNetwork)
    _discretize!(net)
    _transfer_continuous!(net)
    functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    while !isempty(functional_nodes)
        first_node = first(functional_nodes)
        continuous_nodes_2_eliminate = filter(x -> isa(x, ContinuousNode), get_parents(net, first_node)[3])
        eliminable_node = map(x -> EnhancedBayesianNetworks._is_eliminable(net, x),
            continuous_nodes_2_eliminate)
        if any(.!eliminable_node)
            names = [i.name for i in continuous_nodes_2_eliminate[.!eliminable_node]]
            error("nodes elimnation algorithm will lead to a cyclic network when elimnating node/s $names")
        end
        try
            global evaluated_node = _evaluate_node(net, first_node)
        catch e
            if isa(e, AssertionError)
                error("node $(getproperty(first_node, :name)) has as imprecise parents only one or more child nodes with a discretization srtucture defined. They are approximated with Uniform and Exponential assumption and they are no more imprecise. A prices simulation technique must be selected")
            else
                parents = get_parents(net, first_node)[3]
                imprecise_parents = parents[_is_imprecise.(parents)]
                names = [i.name for i in imprecise_parents]
                error("node $(getproperty(first_node, :name)) has $(getproperty(first_node, :simulation)) as simulation technique, but have $names as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead")
            end
        end
        index = findfirst(==(first_node), net.nodes)
        net.nodes[index] = evaluated_node
        _discretize!(net)
        _transfer_continuous!(net)
        functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    end
end

function evaluate_with_envelopes(net::EnhancedBayesianNetwork)
    envelopes = markov_envelope(net)
    envelopes = map(x -> _add_root2envelope(net, x), envelopes)
    ebns = map(x -> _build_envelope_edges(net, x), envelopes)
    map(ebn -> evaluate!(ebn), ebns)
    return ebns
end

function reduce!(net::EnhancedBayesianNetwork)
    if isempty(filter(x -> isa(x, FunctionalNode), net.nodes))
        cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
        map(x -> _remove_node!(net, x), cont_nodes)
    else
        evaluate!(net)
        cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
        map(x -> _remove_node!(net, x), cont_nodes)
    end
    ##! todo add mapping to correct BN or CN struct
    return nothing
end

function _add_root2envelope(net::EnhancedBayesianNetwork, envelope::AbstractVector{<:AbstractNode})
    parents_vector = map(x -> get_parents(net, x)[3], envelope)
    is_not_in = map(x -> [i ∉ envelope for i in x], parents_vector)
    while any(collect(Iterators.Flatten(is_not_in)))
        missing_nodes = map((x, y) -> x[y], parents_vector, is_not_in)
        missing_nodes = unique(collect(Iterators.Flatten(missing_nodes)))
        envelope = append!(envelope, missing_nodes)
        parents_vector = map(x -> get_parents(net, x)[3], envelope)
        is_not_in = map(x -> [i ∉ envelope for i in x], parents_vector)
    end
    return envelope
end

function _build_envelope_edges(net::EnhancedBayesianNetwork, envelope::AbstractVector{<:AbstractNode})
    ebn = EnhancedBayesianNetwork(envelope)
    for node in envelope
        par = get_parents(net, node)[3]
        for p in par
            if p ∈ envelope
                add_child!(ebn, p, node)
            end
        end
    end
    order_net!(ebn)
    return ebn
end