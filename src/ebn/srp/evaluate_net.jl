
function _evaluate_single_layer(ebn::EnhancedBayesianNetwork)
    ## Discretize ebn
    disc_ebn = discretize(ebn)
    ## transfer all possible continuous functional node's model to their discrete functional children
    trans_ebn = transfer_continuous(disc_ebn)
    nodes = trans_ebn.nodes
    ## get 1st layer nodes
    functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
    functional_nodes_to_eval = filter(x -> all(!isa(y, FunctionalNode) for y in x.parents), functional_nodes)
    ## evaluate
    res_nodes = _evaluate(functional_nodes_to_eval)
    ## Ask Jasper a for loop braker
    for i in range(1, length(res_nodes))
        nodes = _replace_node(nodes, functional_nodes_to_eval[i], res_nodes[i])
    end
    return nodes
end

function evaluate(ebn::EnhancedBayesianNetwork)
    ## test reducibility
    nodes = ebn.nodes
    if _is_reducible(ebn)
        while !isempty(filter(x -> isa(x, FunctionalNode), nodes))
            nodes = _evaluate_single_layer(ebn)
            ebn = EnhancedBayesianNetwork(nodes)
        end
        return reduce!(ebn)
    else
        error("Irreducible network")
    end
end
function transfer_continuous(ebn::EnhancedBayesianNetwork)
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
\
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