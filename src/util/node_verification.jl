function _verify_node(node::ChildNode, parents::AbstractVector)
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

function _verify_node(node::FunctionalNode, parents::AbstractVector)
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

# function _verify_probabilities(states::Dict{Symbol,<:Real})
#     any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
#     any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
#     total_probability = sum(values(states))
#     if total_probability != 1
#         if isapprox(total_probability, 1; atol=0.05)
#             @warn "total probaility should be one, but the evaluated value is $total_probability , and will be normalized"
#         else
#             probs = collect(values(states))
#             error("defined states probabilities $probs are wrong")
#         end
#     end
# end

# function _verify_probabilities(states::Dict{Symbol,AbstractVector{Real}})
#     probability_values = vcat(collect(values(states))...)
#     if any(probability_values .< 0)
#         error("Probabilites must be nonnegative")
#     elseif any(probability_values .> 1)
#         error("Probabilites must be less or equal to 1.0")
#     elseif sum(first.(values(states))) > 1
#         error("sum of intervals lower bounds is bigger than 1 in $states")
#     elseif sum(last.(values(states))) < 1
#         error("sum of intervals upper bounds is smaller than 1 in $states")
#     end
# end

# function __verify_parameters(states::Dict, parameters::Dict{Symbol,Vector{Parameter}})
#     if !isempty(parameters)
#         if keys(states) != keys(parameters)
#             error("parameters must be coherent with states")
#         end
#     end
# end

# function _verify_single_state(state::Union{Dict{Symbol,Real},Dict{Symbol,AbstractVector{Real}}}, parameters::Dict{Symbol,Vector{Parameter}})
#     _verify_probabilities(state)
#     _verify_parameters(state, parameters)
# end

# function _normalize_state!(states::Dict{Symbol,Real})
#     normalized_prob = normalize(collect(values(states)), 1)
#     normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
#     return convert(Dict{Symbol,Real}, normalized_states)
# end

# function _verify_child_node_state!(states, parameters)
#     try
#         states = convert(Dict{Symbol,Real}, states)
#     catch
#         try
#             states = convert(Dict{Symbol,AbstractVector{Real}}, states)
#         catch
#             error("node $name must have real valued states probailities or real valued interval states probabilities")
#         end
#     end
#     _verify_single_state(states, parameters)
#     if isa(states, Dict{Symbol,Real})
#         states = _normalize_state!(states)
#     end
#     return states
# end

# function _verify_child_parents(states, parents::AbstractVector{<:AbstractNode})
#     functional_parents = filter(x -> isa(x, FunctionalNode), parents)
#     if !isempty(functional_parents)
#         functional_names = [i.name for i in functional_parents]
#         error("Children of functional node/s $functional_names, must be defined through a FunctionalNode struct")
#     end
#     continuous_parents = filter(x -> isa(x, ContinuousNode), parents)
#     if !isempty(continuous_parents)
#         continuous_names = [i.name for i in continuous_parents]
#         error("Children of continuous node/s $continuous_names, must be defined through a FunctionalNode struct")
#     end
# end

# function _verify_child_node_states_scenario(states, parents::AbstractVector{<:AbstractNode})
#     discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
#     discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
#     discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
#     if !issetequal(discrete_parents_combination, collect(keys(states)))
#         combination_list = collect(keys(states))
#         error("Defined combinations, $combination_list ,are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
#     end
# end

# function _verify_functionalnode_parents(parents::Vector{<:AbstractNode})
#     discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
#     discrete_parents = filter(x -> !isa(x, FunctionalNode), discrete_parents)
#     if !isempty(discrete_parents)
#         discrete_parents_states = vcat([collect(keys(x.states)) for x in discrete_parents]...)
#         discrete_parents_states = mapreduce(x -> _get_states(x), vcat, discrete_parents)

#         any(isempty.([x.parameters for x in discrete_parents])) && error("all discrete parents of a functional node must have a non-empty parameters dictionary")

#         unique(discrete_parents_states) != discrete_parents_states && error("all discrete parents of a functional node must have different named states")
#     end
# end

