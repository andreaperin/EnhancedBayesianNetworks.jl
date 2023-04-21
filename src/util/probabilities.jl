_not_negative(states::Dict{Symbol,<:Real}) = any(values(states) .< 0.0)
_less_than_one(states::Dict{Symbol,<:Real}) = any(values(states) .> 1.0)
_sum_up_to_one(states::Dict{Symbol,<:Real}) = sum(values(states)) != 1