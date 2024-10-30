function _verify_node(node::ChildNode, parents::AbstractVector{<:AbstractNode})
    ## Check scenarios coherence for non functional nodes
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
    discrete_parents_combination = vec(map(t -> [t...], discrete_parents_combination))
    discrete_parents_combination_set = map(x -> Set(x), discrete_parents_combination)
    scenarios_set = map(x -> Set(x), _get_scenarios(node))
    if !issetequal(discrete_parents_combination, _get_scenarios(node))
        is_present = map(x -> x ∈ scenarios_set, discrete_parents_combination_set)
        missing_parents_combinations = discrete_parents_combination_set[.!is_present]
        if !isempty(missing_parents_combinations)
            error("parents combinations $missing_parents_combinations, are missing in node $(node.name) defined scenarios $scenarios_set")
        end
    end
    return nothing
end

function _verify_node(node::FunctionalNode, parents::AbstractVector{<:AbstractNode})
    continuous_parents = filter(x -> isa(x, ContinuousNode), parents)
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    if isempty(continuous_parents)
        error("functional nodes $(node.name) must have at least one continuous parent")
    end
    have_no_parameter = map(x -> isempty(x.parameters), discrete_parents)
    no_parameters_nodes = discrete_parents[have_no_parameter]
    ## Functional Node must have discrete parents with a defined parameters argument
    if !isempty(no_parameters_nodes)
        no_parameters_nodes_name = [i.name for i in no_parameters_nodes]
        error("node/s $no_parameters_nodes_name are discrete and parents of the functional node $(node.name), therefore a parameter argument must be defined")
    end
    return nothing
end