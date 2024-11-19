# function _verify_probability_values(cpt::DataFrame)
#     probabilities = vcat(cpt[!, :Prob]...)
#     if any(probabilities .< 0)
#         error("probabilities must be non-negative")
#     elseif any(probabilities .> 1)
#         error("probabilities must be lower or equal than 1")
#     end
#     if !all(isa.(cpt[!, :Prob], Vector{<:Real})) && !all(isa.(cpt[!, :Prob], Real))
#         error("Mixed precise and imprecise probabilities values $cpt")
#     end
#     if all(isa.(cpt[!, :Prob], Vector{<:Real})) && any(length.(cpt[!, :Prob]) .!= 2)
#         error("interval probabilities must be defined with a 2-values vector")
#     end
# end

# function _verify_mutual_exclusivity_and_exhaustiveness(sub_cpt::DataFrame)
#     if all(isa.(sub_cpt[!, :Prob], Real))
#         if sum(sub_cpt[!, :Prob]) != 1
#             error("States are not exhaustive and mutually exclusives for the following cpt: $sub_cpt")
#         end
#     elseif all(isa.(sub_cpt[!, :Prob], Vector{<:Real}))
#         if sum(first.(sub_cpt[!, :Prob])) >= 1
#             error("sum of intervals lower bounds is greater than 1: $cpt")
#         elseif sum(last.(sub_cpt[!, :Prob])) <= 1
#             error("sum of intervals upper bounds is smaller than 1: $cpt")
#         end
#     end
# end

# function _normalize_states!(sub_cpt::DataFrame)
#     if all(isa.(sub_cpt[!, :Prob], Real))
#         sub_cpt[!, :Prob] = normalize(sub_cpt[!, :Prob], 1)
#     end
# end

# function _byrow(evidence::Evidence)
#     k = collect(keys(evidence))
#     v = collect(values(evidence))
#     return map((n, s) -> n => ByRow(x -> x == s), k, v)
# end