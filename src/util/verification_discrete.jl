function _verify_cpt_coherence(cpt::DataFrame)
    _verify_prob_column(cpt)
    if !all(isa.(cpt[!, :Π], Vector{<:Real})) && !all(isa.(cpt[!, :Π], Real))
        error("Mixed precise and imprecise probabilities values $cpt")
    end
end

function _verify_precise_probabilities_values(cpt::DataFrame)
    probabilities = vcat(cpt[!, :Π]...)
    if any(probabilities .< 0)
        error("probabilities must be non-negative: $cpt")
    elseif any(probabilities .> 1)
        error("probabilities must be lower or equal than 1: $cpt")
    end
end

function _verify_imprecise_probabilities_values(cpt::DataFrame)
    if all(isa.(cpt[!, :Π], Vector{<:Real}))
        if any(length.(cpt[!, :Π]) .!= 2)
            error("interval probabilities must be defined with a 2-values vector. $cpt")
        end
        if any(first.(cpt[!, :Π]) .- last.(cpt[!, :Π]) .> 0)
            error("interval probabilities must lower bound smaller than upper bound. $cpt")
        end
    end
end

## for sub_cpts
function _verify_precise_exhaustiveness_and_normalize!(sub_cpt::DataFrame)
    if all(isa.(sub_cpt[!, :Π], Real))
        total_probability = sum(sub_cpt[!, :Π])
        if total_probability != 1
            if isapprox(total_probability, 1; atol=0.05)
                @warn "total probability should be one, but the evaluated value is $total_probability, and will be normalized"
                _normalize!(sub_cpt)
            else
                error("states are not exhaustive and mutually exclusives for the following cpt: $sub_cpt")
            end
        end
    end
    return sub_cpt
end

function _verify_imprecise_exhaustiveness(sub_cpt::DataFrame)
    if all(isa.(sub_cpt[!, :Prob], Vector{<:Real}))
        if sum(first.(sub_cpt[!, :Prob])) > 1
            error("sum of intervals lower bounds is bigger than 1: $sub_cpt")
        elseif sum(last.(sub_cpt[!, :Prob])) < 1
            error("sum of intervals upper bounds is smaller than 1: $sub_cpt")
        end
    end
end

function _normalize!(sub_cpt::DataFrame)
    sub_cpt[!, :Π] = normalize(sub_cpt[!, :Π], 1)
end