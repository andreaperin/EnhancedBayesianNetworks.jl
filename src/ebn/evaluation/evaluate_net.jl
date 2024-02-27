function evaluate(ebn::EnhancedBayesianNetwork)
    while !isempty(filter(x -> isa(x, FunctionalNode), ebn.nodes))
        ebn = _evaluate_routine(ebn)
    end
    return ebn
end

function _evaluate_routine(ebn::EnhancedBayesianNetwork)
    ## Discretize ebn
    disc_ebn = discretize(ebn)
    ## transfer_continuous
    ebn2eval = transfer_continuous(disc_ebn)
    nodes = deepcopy(ebn2eval.nodes)
    ## Reducibility check
    nodes2reduce = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    indices2reduce = map(x -> ebn2eval.name_to_index[x.name], nodes2reduce)
    if _is_reducible(ebn2eval.dag, indices2reduce)
        i = first(filter(x -> isa(x, FunctionalNode), nodes))
        evaluated_i = evaluate(i)
        nodes = _replace_node!(deepcopy(nodes), i, evaluated_i)
    end
    ## Removing barren nodes
    _clean_up!(nodes)
    ebn = EnhancedBayesianNetwork(nodes)
end

function _clean_up!(nodes::AbstractVector{AbstractNode})
    nodes2clean = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    is_withoutchild = map(x -> _find_children(x, nodes) == 0, nodes2clean)
    nodes2clean = nodes2clean[is_withoutchild]
    for i in nodes2clean
        index = findfirst(x -> x == i, nodes)
        deleteat!(nodes, index)
    end
end

function _replace_node!(nodes::AbstractVector{AbstractNode}, old::FunctionalNode, new::ChildNode)
    index = findfirst(x -> isequal(x, old), nodes)
    deleteat!(nodes, index)
    for node in nodes
        if isa(node, RootNode)
            continue
        else
            if old ∈ node.parents
                @show(node.name)
                node.parents[:] = [filter(x -> x != old, node.parents)..., new]
            end
        end
    end
    insert!(nodes, index, new)
end

function _find_children(n, nodes)
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