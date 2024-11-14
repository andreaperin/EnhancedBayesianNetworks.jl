@auto_hash_equals struct ContinuousNode{T<:AbstractContinuousInput} <: AbstractNode
    name::Symbol
    cpt::DataFrame
    discretization::AbstractDiscretization
    additional_info::Dict{Vector{Symbol},Dict}
end

function ContinuousNode{T}(name::Symbol, cpt::DataFrame) where {T<:AbstractContinuousInput}
    ncol(cpt) > 2 ? d = ApproximatedDiscretization() : d = ExactDiscretization()
    ContinuousNode{T}(name, cpt, d, Dict{Vector{Symbol},Dict}())
end

function ContinuousNode{T}(name::Symbol, cpt::DataFrame, discretization::AbstractDiscretization) where {T<:AbstractContinuousInput}
    ContinuousNode{T}(name, cpt, discretization, Dict{Vector{Symbol},Dict}())
end

function _distributions(node::ContinuousNode)
    return unique(node.cpt[!, :Prob])
end

function _scenarios(node::ContinuousNode)
    return copy.(eachrow(node.cpt[!, Not(:Prob)]))
end

function _continuous_input(node::ContinuousNode{UnivariateDistribution}, evidence::Evidence)
    df_row = subset(node.cpt, _byrow(evidence))
    if nrow(df_row) == 1
        return RandomVariable(node.cpt[!, :Prob][1], node.name)
    elseif nrow(df_row) > 1
        return map(dist -> RandomVariable(dist, node.name), node.cpt[!, :Prob])
    end
end

function _continuous_input(node::ContinuousNode{Tuple{Real,Real}}, evidence::Evidence)
    df_row = subset(node.cpt, _byrow(evidence))
    return map(tup -> Interval(tup..., node.name), df_row[!, :Prob])
end

function _continuous_input(node::ContinuousNode{UnamedProbabilityBox}, evidence::Evidence)
    df_row = subset(node.cpt, _byrow(evidence))
    dists = map(r -> first(typeof(r).parameters), df_row[!, :Prob])
    return map((upb, dist) -> ProbabilityBox{dist}(upb.parameters, node.name, upb.lb, upb.ub), df_row[!, :Prob], dists)
end

_continuous_input(node::ContinuousNode) = _continuous_input(node, Evidence())

function _byrow(evidence::Evidence)
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end

function _truncate(dist::UnivariateDistribution, i::AbstractVector)
    return truncated(dist, i[1], i[2])
end

function _truncate(dist::UnamedProbabilityBox, i::AbstractVector)
    return UnamedProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, i[1], i[2])
end

function _truncate(_::Tuple{T,T}, i::AbstractVector) where {T<:Real}
    return (i[1], i[2])
end

function _is_precise(node::ContinuousNode)
    all(isa.(node.cpt[!, :Prob], UnivariateDistribution))
end