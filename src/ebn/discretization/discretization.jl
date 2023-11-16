abstract type AbstractDiscretization end


``` ExactDiscretization

    Used for ContinuousRootNode whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support

```
struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}

    function ExactDiscretization(intervals::Vector{<:Real})
        if !issorted(intervals)
            error("interval values $intervals are not sorted")
        end
        new(intervals)
    end
end

function Base.isequal(discretization1::ExactDiscretization, discretization2::ExactDiscretization)
    discretization1.intervals == discretization2.intervals
end

function Base.hash(discretization::ExactDiscretization, h::UInt)
    h = hash(discretization.intervals, h)
    return h
end

ExactDiscretization() = ExactDiscretization(Vector{Real}())

``` ApproximatedDiscretization

    Used for continuous Non-Root nodes whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support
        sigma: variance of the normal distribution used for appriximate initial continuous distribution

```
struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    sigma::Union{Real,Nothing}

    function ApproximatedDiscretization(intervals::Vector{<:Real}, sigma::Union{Real,Nothing})
        if sort(intervals) != intervals
            error("interval values $intervals are not sorted")
        elseif !isa(sigma, Nothing) && sigma ≤ 0
            error("variance must be positive")
        elseif !isa(sigma, Nothing) && sigma > 2
            @warn "Selected variance values $sigma can be too big, and the approximation not realistic"
        end
        new(intervals, sigma)
    end
end

function Base.isequal(discretization1::ApproximatedDiscretization, discretization2::ApproximatedDiscretization)
    discretization1.intervals == discretization2.intervals && discretization1.sigma == discretization2.sigma
end

function Base.hash(discretization::ApproximatedDiscretization, h::UInt)
    h = hash(discretization.intervals, h)
    h = hash(discretization.sigma, h)
    return h
end

ApproximatedDiscretization() = ApproximatedDiscretization(Vector{Real}(), nothing)