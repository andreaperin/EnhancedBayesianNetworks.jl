function _byrow(evidence::Evidence)
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end

function _verify_parameters(node::DiscreteNode)
    if !isempty(parameters)
        states_id = collect(keys(states))
        parameters_id = collect(keys(parameters))
        if keys(states) != keys(parameters)
            error("parameters keys $parameters_id must be coherent with states $states_id")
        end
    end
end

function _check_root_states!(states::Dict{Symbol,<:AbstractDiscreteProbability})
    _verify_probabilities(states)
    _normalize_states!(states)
end

## Child Discrete
function _check_child_states!(states)
    ## check states coherency over scenarios
    defined_states = map(s -> (collect(keys(s)), collect(values(s))), values(states))
    if !allequal([s[1] for s in defined_states])
        error("non coherent definition of states over scenarios: $defined_states")
    end
    ## check states values coherency over scenarios
    if !allequal(typeof.([s[2] for s in defined_states]))
        error("mixed interval and single value states probabilities are not allowed")
    end
    ## Normalize and Verigy single states
    states = Dict(map((scenario, state) -> (scenario, _check_root_states!(state)), keys(states), values(states)))
    return states
end

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
        @warn("functional nodes $(node.name) have no continuous parents")
    end
    non_functional_discrete_parents = filter(x -> !isa(x, FunctionalNode), discrete_parents)
    have_no_parameter = map(x -> isempty(x.parameters), non_functional_discrete_parents)
    no_parameters_nodes = non_functional_discrete_parents[have_no_parameter]
    ## Functional Node must have discrete parents with a defined parameters argument
    if !isempty(no_parameters_nodes)
        no_parameters_nodes_name = [i.name for i in no_parameters_nodes]
        error("node/s $no_parameters_nodes_name are discrete and parents of the functional node $(node.name), therefore a parameter argument must be defined")
    end
    return nothing
end