@auto_hash_equals struct ContinuousNode <: AbstractContinuousNode
    name::Symbol
    cpt::ContinuousConditionalProbabilityTable
    discretization::AbstractDiscretization
    additional_info::Dict{Vector{Symbol},Dict}

    function ContinuousNode(
        name::Symbol,
        cpt::ContinuousConditionalProbabilityTable,
        discretization::AbstractDiscretization,
        additional_info::Dict{Vector{Symbol},Dict}
    )
        ## Check appropriate Discretization struct
        _verify_discretization(cpt, discretization)
        ## setting :Π  as last column and sorting
        select!(cpt.data, Not(:Π), :Π)
        sort!(cpt.data)
        new(name, cpt, discretization, additional_info)
    end
end

function ContinuousNode(name::Symbol, cpt::ContinuousConditionalProbabilityTable)
    isroot(cpt) ? d = ExactDiscretization() : d = ApproximatedDiscretization()
    ContinuousNode(name, cpt, d, Dict{Vector{Symbol},Dict}())
end

function ContinuousNode(name::Symbol, cpt::ContinuousConditionalProbabilityTable, discretization::AbstractDiscretization)
    ContinuousNode(name, cpt, discretization, Dict{Vector{Symbol},Dict}())
end

distributions(node::ContinuousNode) = distributions(node.cpt)

scenarios(node::ContinuousNode) = scenarios(node.cpt)

isprecise(node::ContinuousNode) = isprecise(node.cpt)

isroot(node::ContinuousNode) = isroot(node.cpt)

function _uq_inputs(node::ContinuousNode, evidence::Evidence)
    new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt.data)), evidence)
    df_row = subset(node.cpt.data, _by_row(new_evidence))
    if typeof(node.cpt).parameters[1] == PreciseContinuousInput
        return map(dist -> RandomVariable(dist, node.name), df_row[!, :Π])
    elseif typeof(node.cpt).parameters[1] == Tuple{<:Real,<:Real} || isa(node.cpt.data.Π[1], Tuple{<:Real,<:Real})
        return map(tup -> Interval(tup..., node.name), df_row[!, :Π])
    elseif typeof(node.cpt).parameters[1] == UnamedProbabilityBox || isa(node.cpt.data.Π[1], UnamedProbabilityBox)
        dists = map(r -> first(typeof(r).parameters), df_row[!, :Π])
        return map((upb, dist) -> ProbabilityBox{dist}(upb.parameters, node.name, upb.lb, upb.ub), df_row[!, :Π], dists)
    end
end

_uq_inputs(node::ContinuousNode) = _uq_inputs(node, Evidence())


# function _continuous_node_input_type(x::ContinuousInput)
#     if isa(x, UnivariateDistribution)
#         return UnivariateDistribution
#     elseif isa(x, Tuple{Real,Real})
#         return Tuple{Real,Real}
#     elseif isa(x, UnamedProbabilityBox)
#         return UnamedProbabilityBox
#     end
# end


# function _continuous_input(node::ContinuousNode{UnivariateDistribution}, evidence::Evidence)
#     new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
#     df_row = subset(node.cpt, _by_row(new_evidence))
#     return map(dist -> RandomVariable(dist, node.name), df_row[!, :Π])
# end

# function _continuous_input(node::ContinuousNode{Tuple{Real,Real}}, evidence::Evidence)
#     new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
#     df_row = subset(node.cpt, _by_row(new_evidence))
#     return map(tup -> Interval(tup..., node.name), df_row[!, :Π])
# end

# function _continuous_input(node::ContinuousNode{UnamedProbabilityBox}, evidence::Evidence)
#     new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt)), evidence)
#     df_row = subset(node.cpt, _by_row(new_evidence))
#     dists = map(r -> first(typeof(r).parameters), df_row[!, :Π])
#     return map((upb, dist) -> ProbabilityBox{dist}(upb.parameters, node.name, upb.lb, upb.ub), df_row[!, :Π], dists)
# end

# function _distribution_bounds(dist::UnivariateDistribution)
#     return [support(dist).lb, support(dist).ub]
# end

# function _distribution_bounds(dist::Tuple{T,T}) where {T<:Real}
#     return [dist[1], dist[2]]
# end

# function _distribution_bounds(dist::UnamedProbabilityBox)
#     return [minimum(vcat(map(x -> x.lb, dist.parameters), dist.lb)), maximum(vcat(map(x -> x.ub, dist.parameters), dist.ub))]
# end

# function _distribution_bounds(node::ContinuousNode)
#     bounds = mapreduce(dist -> _distribution_bounds(dist), hcat, node.cpt[!, :Π])
#     return [minimum(bounds[1, :]), maximum(bounds[2, :])]
# end

# function _truncate(dist::UnivariateDistribution, i::AbstractVector)
#     return truncated(dist, i[1], i[2])
# end

# function _truncate(_::Tuple{T,T}, i::AbstractVector) where {T<:Real}
#     return (i[1], i[2])
# end

# function _truncate(dist::UnamedProbabilityBox, i::AbstractVector)
#     return UnamedProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, i[1], i[2])
# end