function verify_probabilities(states::Dict{Symbol,<:Real})
    any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
    any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
    total_proability = sum(values(states))
    if total_proability != 1
        if isapprox(total_proability, 1; rtol=0.01)
            @warn "total probaility should be one, but the evaluated value is $total_proability , and will be normalized"
        else
            error("defined states probabilities $states are wrong")
        end
    end
end

function verify_probabilities(states::Dict{Symbol,AbstractVector{Real}})
    probability_values = vcat(collect(values(states))...)
    if any(probability_values .< 0)
        error("Probabilites must be nonnegative")
    elseif any(probability_values .> 1)
        error("Probabilites must be less or equal to 1.0")
    elseif sum(first.(values(states))) > 1
        error("sum of intervals lower bounds is bigger than 1 in $states")
    elseif sum(last.(values(states))) < 1
        error("sum of intervals upper bounds is smaller than 1 in $states")
    end
end

function verify_parameters(states::Dict, parameters::Dict{Symbol,Vector{Parameter}})
    if !isempty(parameters)
        if keys(states) != keys(parameters)
            error("parameters must be coherent with states")
        end
    end
end

function _verify_single_state(state::Union{Dict{Symbol,Real},Dict{Symbol,AbstractVector{Real}}}, parameters::Dict{Symbol,Vector{Parameter}})
    verify_probabilities(state)
    verify_parameters(state, parameters)
end

function _verify_discrete_root_node_state!(states, parameters)
    try
        states = convert(Dict{Symbol,Real}, states)
    catch
        try
            states = convert(Dict{Symbol,AbstractVector{Real}}, states)
        catch
            error("node $name must have real valued states probailities or real valued interval states probabilities")
        end
    end
    _verify_single_state(states, parameters)
    if isa(states, Dict{Symbol,Real})
        states = _normalize_state!(states)
    end
    return states
end

function verify_functionalnode_parents(parents::Vector{<:AbstractNode})
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents = filter(x -> !isa(x, FunctionalNode), discrete_parents)
    if !isempty(discrete_parents)
        discrete_parents_states = vcat([collect(keys(x.states)) for x in discrete_parents]...)
        discrete_parents_states = mapreduce(x -> _get_states(x), vcat, discrete_parents)

        any(isempty.([x.parameters for x in discrete_parents])) && error("all discrete parents of a functional node must have a non-empty parameters dictionary")

        unique(discrete_parents_states) != discrete_parents_states && error("all discrete parents of a functional node must have different named states")
    end
end

