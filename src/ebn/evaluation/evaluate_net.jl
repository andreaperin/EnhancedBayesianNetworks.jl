function evaluate(ebn::EnhancedBayesianNetwork)
    while !isempty(filter(x -> isa(x, FunctionalNode), ebn.nodes))
        ebn = _evaluate_routine(ebn)
    end
    if isempty(filter(x -> isa(x, ContinuousNode), ebn.nodes))
        if all(.!_is_imprecise.(ebn.nodes))
            return BayesianNetwork(ebn.nodes)
        else
            return CredalNetwork(ebn.nodes)
        end
    else
        return ebn
    end
end

function _evaluate_routine(ebn::EnhancedBayesianNetwork)
    ## Discretize ebn
    disc_ebn = discretize(ebn)
    ## transfer_continuous
    ebn2eval = transfer_continuous(disc_ebn)
    nodes = ebn2eval.nodes
    ## Reducibility check
    nodes2reduce = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    indices2reduce = map(x -> ebn2eval.name_to_index[x.name], nodes2reduce)
    dag = deepcopy(ebn2eval.dag)
    if all(map(x -> _is_reducible(dag, x), indices2reduce))
        i = first(filter(x -> isa(x, FunctionalNode), nodes))
        evaluated_i = _evaluate(i)
        nodes = _replace_node!(nodes, i, evaluated_i)
    else
        error("irreducible network")
    end
    ## Removing barren nodes
    _clean_up!(nodes)
    ebn = EnhancedBayesianNetwork(nodes)

end

function _replace_node!(nodes::AbstractVector{AbstractNode}, old::FunctionalNode, new::Union{ChildNode,RootNode})
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

function _clean_up!(nodes::AbstractVector{AbstractNode})
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