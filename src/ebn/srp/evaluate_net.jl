function evaluate(ebn::EnhancedBayesianNetwork)
    if _is_reducible(ebn)
        ## Discretize ebn
        disc_ebn = discretize(ebn)
        ## transfer all possible continuous functional node's model to their discrete functional children
        trans_ebn = _transfer_continuous(disc_ebn)
        nodes = trans_ebn.nodes
        functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
        while !isempty(functional_nodes)
            evaluated_node = evaluate(first(functional_nodes))
            nodes = _replace_node(nodes, first(functional_nodes), evaluated_node)
            if isa(evaluated_node, ContinuousChildNode)
                if !isempty(evaluated_node.discretization.intervals)
                    ebn = discretize(EnhancedBayesianNetwork(nodes))
                    nodes = ebn.nodes
                end
            end
            functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
        end
        return EnhancedBayesianNetwork(nodes)
    else
        error("Irreducible network")
    end
end

function _transfer_continuous(ebn::EnhancedBayesianNetwork)
    new_ebn = deepcopy(ebn)
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), new_ebn.nodes)
    continuous_functional_to_transfer = filter(x -> isempty(x.discretization.intervals), continuous_functional)
    while !isempty(continuous_functional_to_transfer)
        level = ContinuousFunctionalNode[]
        for i in continuous_functional
            if !any(isa.(i.parents, ContinuousFunctionalNode))
                push!(level, i)
            end
        end
        new_ebn = _transfer_continuous_functional(new_ebn, level)
        continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), new_ebn.nodes)
        continuous_functional_to_transfer = filter(x -> isempty(x.discretization.intervals), continuous_functional)
    end
    return new_ebn
end

function _transfer_continuous_functional(ebn::EnhancedBayesianNetwork, first_level::Vector{ContinuousFunctionalNode})
    while !isempty(first_level)
        ebn = _transfer_single_continuous_functional(ebn, first_level[1])
        popfirst!(first_level)
    end
    return ebn
end

function _transfer_single_continuous_functional_single_child(parent::ContinuousFunctionalNode, child::FunctionalNode)
    child.parents = append!(parent.parents, filter(x -> x.name != parent.name, child.parents))
    child.models = append!(parent.models, child.models)
    return child
end

function _transfer_single_continuous_functional(ebn::EnhancedBayesianNetwork, parent::ContinuousFunctionalNode)
    if isempty(parent.discretization.intervals)
        new_children = map(x -> _transfer_single_continuous_functional_single_child(parent, x), get_children(ebn, parent))
        name = push!([i.name for i in new_children], parent.name)
        new_nodes = filter(x -> x.name ∉ name, ebn.nodes)
        append!(new_nodes, new_children)
        new_ebn = EnhancedBayesianNetwork(new_nodes)
    else
        new_ebn = ebn
    end
    return new_ebn
end

function _replace_node(nodes::AbstractVector{AbstractNode}, old::FunctionalNode, new::ChildNode)
    if isa(old, DiscreteNode) && isa(new, ContinuousNode)
        error("cannot replace ContinuousNodes with DiscreteNodes or viceversa")
    end
    # remove original continuous nodes
    nodes = filter(x -> x != old, nodes)
    for node in nodes
        if isa(node, RootNode)
            continue
        end
        if old in node.parents
            node.parents[:] = [filter(x -> x !== old, node.parents)..., new]
        end
    end
    push!(nodes, new)
    return nodes
end