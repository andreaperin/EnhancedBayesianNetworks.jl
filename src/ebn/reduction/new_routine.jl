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