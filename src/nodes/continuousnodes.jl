@auto_hash_equals struct ContinuousNode{T<:AbstractContinuousInput} <: AbstractContinuousNode
    name::Symbol
    cpt::DataFrame
    discretization::AbstractDiscretization
    additional_info::Dict{Vector{Symbol},Dict}

    function ContinuousNode{T}(
        name::Symbol,
        cpt::DataFrame,
        discretization::AbstractDiscretization,
        additional_info::Dict{Vector{Symbol},Dict}
    ) where {T<:AbstractContinuousInput}
        ## Check appropriate Discretization struct
        _verify_discretization(cpt, discretization)
        ## setting :Prob as last column and sorting
        select!(cpt, Not(:Prob), :Prob)
        sort!(cpt)
        new{T}(name, cpt, discretization, additional_info)
    end
end

function ContinuousNode{T}(name::Symbol, cpt::DataFrame) where {T<:AbstractContinuousInput}
    _is_continuous_root(cpt) ? d = ExactDiscretization() : d = ApproximatedDiscretization()
    ContinuousNode{T}(name, cpt, d, Dict{Vector{Symbol},Dict}())
end

function ContinuousNode{T}(name::Symbol, cpt::DataFrame, discretization::AbstractDiscretization) where {T<:AbstractContinuousInput}
    ContinuousNode{T}(name, cpt, discretization, Dict{Vector{Symbol},Dict}())
end

function _distributions(cpt::DataFrame)
    unique(cpt[!, :Prob])
end

_distributions(node::ContinuousNode) = _distributions(node.cpt)

function _scenarios(cpt::DataFrame)
    scenarios = copy.(eachrow(cpt[!, Not(:Prob)]))
    return unique(map(s -> Dict(pairs(s)), scenarios))
end

_scenarios(node::ContinuousNode) = _scenarios(node.cpt)

function _continuous_input(node::ContinuousNode{UnivariateDistribution}, evidence::Evidence)
    new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
    df_row = subset(node.cpt, _by_row(new_evidence))
    return map(dist -> RandomVariable(dist, node.name), df_row[!, :Prob])
end

function _continuous_input(node::ContinuousNode{Tuple{Real,Real}}, evidence::Evidence)
    new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
    df_row = subset(node.cpt, _by_row(new_evidence))
    return map(tup -> Interval(tup..., node.name), df_row[!, :Prob])
end

function _continuous_input(node::ContinuousNode{UnamedProbabilityBox}, evidence::Evidence)
    new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
    df_row = subset(node.cpt, _by_row(new_evidence))
    dists = map(r -> first(typeof(r).parameters), df_row[!, :Prob])
    return map((upb, dist) -> ProbabilityBox{dist}(upb.parameters, node.name, upb.lb, upb.ub), df_row[!, :Prob], dists)
end

_continuous_input(node::ContinuousNode) = _continuous_input(node, Evidence())

function _distribution_bounds(dist::UnivariateDistribution)
    return [support(dist).lb, support(dist).ub]
end

function _distribution_bounds(dist::Tuple{T,T}) where {T<:Real}
    return [dist[1], dist[2]]
end

function _distribution_bounds(dist::UnamedProbabilityBox)
    return [minimum(vcat(map(x -> x.lb, dist.parameters), dist.lb)), maximum(vcat(map(x -> x.ub, dist.parameters), dist.ub))]
end

function _distribution_bounds(node::ContinuousNode)
    bounds = mapreduce(dist -> _distribution_bounds(dist), hcat, node.cpt[!, :Prob])
    return [minimum(bounds[1, :]), maximum(bounds[2, :])]
end

function _truncate(dist::UnivariateDistribution, i::AbstractVector)
    return truncated(dist, i[1], i[2])
end

function _truncate(_::Tuple{T,T}, i::AbstractVector) where {T<:Real}
    return (i[1], i[2])
end

function _truncate(dist::UnamedProbabilityBox, i::AbstractVector)
    return UnamedProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, i[1], i[2])
end

function _is_precise(node::ContinuousNode)
    all(isa.(node.cpt[!, :Prob], UnivariateDistribution))
end

function _is_continuous_root(cpt::DataFrame)
    ncol(cpt) == 1
end

_is_root(node::ContinuousNode) = _is_continuous_root(node.cpt)