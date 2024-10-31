# function evaluate_with_envelope(ebn::EnhancedBayesianNetwork)
#     ebns = markov_envelope(ebn)
#     if length(ebns) > 1
#         ebns = _add_missing_nodes_to_envelope.(ebns)
#         eebns = map(ebn -> evaluate(EnhancedBayesianNetwork(ebn)), ebns)
#         final_nodes = unique(collect(Iterators.Flatten([i.nodes for i in eebns])))
#         return get_specific_network(final_nodes)
#     else
#         return evaluate(ebn)
#     end
# end

# function evaluate(ebn::EnhancedBayesianNetwork)
#     while !isempty(filter(x -> isa(x, FunctionalNode), ebn.nodes))
#         ebn = _evaluate_routine(ebn)
#     end
#     return get_specific_network(ebn)
# end

function _evaluate_routine(ebn::EnhancedBayesianNetwork)
    ## Discretize ebn
    disc_ebn = _discretize(ebn)
    ## transfer_continuous
    ebn2eval = _transfer_continuous(disc_ebn)
    nodes = ebn2eval.nodes
    ## Reducibility check
    nodes2reduce = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    indices2reduce = map(x -> ebn2eval.name_to_index[x.name], nodes2reduce)
    dag = deepcopy(ebn2eval.dag)
    if all(map(x -> _is_reducible(dag, x), indices2reduce))
        i = first(filter(x -> isa(x, FunctionalNode), nodes))
        try
            global evaluated_i = _evaluate(i)
        catch e
            if isa(e, AssertionError)
                error("node $(getproperty(i, :name)) has as imprecise parents only one or more child nodes with a discretization srtucture defined. They are approximated with Uniform and Exponential assumption and they are no more imprecise. A prices simulation technique must be selected")
            else
                imprecise_parents = i.parents[_is_imprecise.(i.parents)]
                names = [i.name for i in imprecise_parents]
                error("node $(getproperty(i, :name)) has $(getproperty(i, :simulation)) as simulation technique, but have $names as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead.")
            end
        end
        nodes = _replace_node!(nodes, i, evaluated_i)
    else
        error("irreducible network")
    end
    ## Removing barren nodes
    _clean_up!(nodes)
    ebn = EnhancedBayesianNetwork(nodes)
end

function get_specific_network(nodes::Vector{<:AbstractNode})
    if isempty(filter(x -> isa(x, ContinuousNode), nodes))
        if all(.!_is_imprecise.(nodes))
            return BayesianNetwork(nodes)
        else
            return CredalNetwork(nodes)
        end
    else
        return EnhancedBayesianNetwork(nodes)
    end
end

get_specific_network(ebn::EnhancedBayesianNetwork) = get_specific_network(ebn.nodes)

function _replace_node!(nodes::AbstractVector{<:AbstractNode}, old::FunctionalNode, new::Union{ChildNode,RootNode})
    index = findfirst(x -> isequal(x, old), nodes)
    deleteat!(nodes, index)
    for node in nodes
        if isa(node, RootNode)
            continue
        else
            if old ∈ node.parents
                node.parents[:] = [filter(x -> x != old, node.parents)..., new]
            end
        end
    end
    insert!(nodes, index, new)
end

function _clean_up!(nodes::AbstractVector{<:AbstractNode})
    nodes2clean = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    is_withoutchild = map(x -> _count_children(x, nodes) == 0, nodes2clean)
    nodes2clean = nodes2clean[is_withoutchild]
    if !isempty(filter(x -> isa(x, DiscreteNode), nodes))
        for i in nodes2clean
            index = findfirst(x -> x == i, nodes)
            deleteat!(nodes, index)
        end
    else
        for i in nodes2clean
            index = findfirst(x -> x == i, nodes)
            if isempty(i.additional_info)
                deleteat!(nodes, index)
            end
        end
    end
end

function _count_children(n, nodes)
    counter = 0
    for x in nodes
        if isa(x, RootNode)
            continue
        else
            if n in x.parents
                counter += 1
            end
        end
    end
    return counter
end

function _add_missing_nodes_to_envelope(nodes::AbstractVector{<:AbstractNode})
    parents_vectors = map(x -> x.parents, filter(x -> !isa(x, RootNode), nodes))
    is_not_in = map(x -> [i ∉ nodes for i in x], parents_vectors)
    while any(collect(Iterators.Flatten(is_not_in)))
        missing_nodes = map((x, y) -> x[y], parents_vectors, is_not_in)
        missing_nodes = unique(collect(Iterators.Flatten(missing_nodes)))
        nodes = append!(nodes, missing_nodes)
        parents_vectors = map(x -> x.parents, filter(x -> !isa(x, RootNode), nodes))
        is_not_in = map(x -> [i ∉ nodes for i in x], parents_vectors)
    end
    return nodes
end