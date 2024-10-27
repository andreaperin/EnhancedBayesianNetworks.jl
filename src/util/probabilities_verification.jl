## Root Discrete
function _verify_probabilities(states::Dict{Symbol,<:Real})
    vals = collect(values(states))
    total_probability = sum(vals)
    if any(vals .< 0)
        error("probabilities must be non-negative")
    elseif any(vals .> 1)
        error("probabilities must be lower or equal than 1")
    elseif sum(vals) != 1
        if isapprox(total_probability, 1; atol=0.05)
            @warn "total probaility should be one, but the evaluated value is $total_probability , and will be normalized"
        else
            sts = collect(keys(states))
            probs = collect(values(states))
            error("states $sts are exhaustives and mutually exclusive. Their probabilities $probs does not sum up to 1")
        end
    end
end

function _verify_probabilities(states::Dict{Symbol,<:AbstractVector{<:Real}})
    if any(length.(values(states)) .!= 2)
        error("interval probabilities must be defined with a 2-values vector")
    end
    probability_values = vcat(collect(values(states))...)
    if any(probability_values .< 0)
        error("probabilities must be non-negative: $probability_values")
    elseif any(probability_values .> 1)
        error("probabilities must be lower or equal than 1: $probability_values")
    elseif sum(first.(values(states))) > 1
        error("sum of intervals lower bounds is bigger than 1: $probability_values")
    elseif sum(last.(values(states))) < 1
        error("sum of intervals upper bounds is smaller than 1: $probability_values")
    end
end

function _normalize_states!(states::Dict{Symbol,<:Real})
    normalized_prob = normalize(collect(values(states)), 1)
    normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
    return convert(Dict{Symbol,Real}, normalized_states)
end

function _normalize_states!(states::Dict{Symbol,<:AbstractVector{<:Real}})
    return states
end

function _verify_parameters(states::Dict{Symbol,<:AbstractDiscreteProbability}, parameters::Dict{Symbol,Vector{Parameter}})
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
    states = Dict(map((scenario, state) -> (scenario, EnhancedBayesianNetworks._check_root_states!(state)), keys(states), values(states)))
    return states
end
