@auto_hash_equals struct ConditionalProbabilityDistribution
    target::Symbol
    parents::Vector{Symbol}
    parents_states_mapping_dict::Dict{Symbol,Dict{Symbol,Int64}}
    parental_ncategories::Vector{Int64}
    states::Vector{Symbol}
    probabilities::Dict{Vector{Symbol},Dict{Symbol,Float64}}
end