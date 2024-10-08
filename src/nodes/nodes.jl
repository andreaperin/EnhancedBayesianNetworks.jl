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

include("../util/node_verification.jl")
include("root.jl")
include("child.jl")
include("functional.jl")
include("common.jl")
