abstract type AbstractNetwork end

@auto_hash_equals struct ConditionalProbabilityDistribution
    target::Symbol
    parents::Vector{Symbol}
    parental_ncategories::Vector{Int64}
    states::Vector{Symbol}
    probabilities::DataFrame
end

include("../util/verification_add_child.jl")
include("networks_common.jl")
include("ebn/ebn.jl")
include("bn/bayesnet.jl")
include("cn/credalnet.jl")