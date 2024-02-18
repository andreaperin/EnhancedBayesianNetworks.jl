function evaluate(ebn::EnhancedBayesianNetwork)
    while !isempty(filter(x -> isa(x, FunctionalNode), ebn.nodes))
        ebn = evaluate_first(ebn)
    end
    return ebn
end

function evaluate_first(ebn::EnhancedBayesianNetwork)
    ## Discretize ebn
    disc_ebn = discretize(ebn)
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
    _clean_up!(nodes)
    ebn = EnhancedBayesianNetwork(nodes)
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

function _clean_up!(nodes::AbstractVector{AbstractNode})
    nodes2clean = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
    is_withoutchild = map(x -> _find_children(x, nodes) == 0, nodes2clean)
    nodes2clean = nodes2clean[is_withoutchild]
    for i in nodes2clean
        index = findfirst(x -> x == i, nodes)
        deleteat!(nodes, index)
    end
end

function _evaluate!(nodes::AbstractVector{AbstractNode})
    functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
    for i in functional_nodes
        evaluated_node = evaluate(i)
        nodes = _replace_node(nodes, i, evaluated_node)
        if isa(evaluated_node, ContinuousChildNode)
            if !isempty(evaluated_node.discretization.intervals)
                nodes = _discretize(nodes)
            end
        end
    end
end

function transfer_continuous(ebn::EnhancedBayesianNetwork)
    return EnhancedBayesianNetwork(_transfer_continuous!(deepcopy(ebn.nodes)))
end

function _transfer_continuous!(nodes::AbstractVector{AbstractNode})
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), nodes)
    for c in continuous_functional
        nodes = _transfer_single_continuous_functional!(nodes, c)
    end
    return nodes
end

function _transfer_single_continuous_functional!(nodes::AbstractVector{AbstractNode}, node::ContinuousFunctionalNode)
    node_children = filter(x -> node ∈ x.parents, filter(x -> !isa(x, RootNode), nodes))
    if isempty(node.discretization.intervals) && !isempty(node_children)
        nodes = setdiff(nodes, [node, node_children...])
        children = AbstractNode[]
        for child in node_children
            for n in [node, node.parents...]
                index = findfirst(x -> isequal(x, n), child.parents)
                isnothing(index) ? continue : deleteat!(child.parents, index)
            end
            append!(child.parents, node.parents)
            prepend!(child.models, node.models)
            push!(children, child)
        end
        append!(nodes, children)
    end
    return nodes
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