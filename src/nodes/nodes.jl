abstract type AbstractNode end
abstract type DiscreteNode <: AbstractNode end
abstract type ContinuousNode <: AbstractNode end

@auto_hash_equals struct UnamedProbabilityBox{T<:UnivariateDistribution}
    parameters::Vector{Interval}
    lb::Real
    ub::Real
end

function UnamedProbabilityBox{T}(p::AbstractVector{<:UQInput}) where {T<:UnivariateDistribution}
    domain = support(T())
    return UnamedProbabilityBox{T}(p, domain.lb, domain.ub)
end

const AbstractContinuousInput = Union{UnivariateDistribution,Tuple{Real,Real},UnamedProbabilityBox}

const AbstractDiscreteProbability = Union{Real,AbstractVector{Real}}

include("../util/wrap.jl")

abstract type AbstractDiscretization end

""" ExactDiscretization

    Used for ContinuousRootNode whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support

"""
@auto_hash_equals struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}

    function ExactDiscretization(intervals::Vector{<:Real})
        if !issorted(intervals)
            error("interval values $intervals are not sorted")
        end
        new(intervals)
    end
end

ExactDiscretization() = ExactDiscretization(Vector{Real}())


""" ApproximatedDiscretization

    Used for continuous Non-Root nodes whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support
        sigma: variance of the normal distribution used for appriximate initial continuous distribution

"""
@auto_hash_equals struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    sigma::Real

    function ApproximatedDiscretization(intervals::Vector{<:Real}, sigma::Real)
        if !issorted(intervals)
            error("interval values $intervals are not sorted")
        elseif sigma < 0
            error("variance must be positive")
        elseif sigma > 2
            @warn "Selected variance values $sigma can be too big, and the approximation not realistic"
        end
        new(intervals, sigma)
    end
end

ApproximatedDiscretization() = ApproximatedDiscretization(Vector{Real}(), 0)

function _get_position(nodes::AbstractVector{<:AbstractNode})
    adj_matrix = get_adj_matrix(nodes)
    pos = spring(adj_matrix; iterations=1000)
    return pos
end

function get_adj_matrix(nodes::AbstractVector{<:AbstractNode})
    ordered_list = _order_node(nodes)
    n = length(ordered_list)
    adj_matrix = zeros(n, n)
    for i in range(1, n)
        for j in range(1, n)
            if !isa(ordered_list[j], RootNode) && ordered_list[i] ∈ ordered_list[j].parents
                adj_matrix[i, j] = 1
            end
        end
    end
    return sparse(adj_matrix)
end

function _get_edges(adj_matrix::SparseMatrixCSC)
    n = size(adj_matrix)
    edge_list = Vector{Tuple{Int64,Int64}}()
    for i in range(1, n[1])
        for j in range(1, n[2])
            if adj_matrix[i, j] != 0
                push!(edge_list, (i, j))
            end
        end
    end
    return edge_list
end

function _order_node(nodes::AbstractVector{<:AbstractNode})
    root = filter(n -> isa(n, RootNode), nodes)
    list = setdiff(nodes, root)
    while !isempty(list)
        new_root = filter(x -> all(x.parents .∈ [root]), list)
        root = append!(root, new_root)
        list = setdiff(list, new_root)
    end
    return root
end

include("../util/node_verification.jl")
include("root.jl")
include("child.jl")
include("functional.jl")
include("common.jl")


include("new_child.jl")
include("new_functional.jl")