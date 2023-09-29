abstract type AbstractDiscretization end


``` ExactDiscretization

    Used for ContinuousRootNode whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support

```
struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Number}

    function ExactDiscretization(intervals::Vector{<:Number})
        if sort(intervals) != intervals
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

ExactDiscretization() = ExactDiscretization(Vector{Number}())

``` ApproximatedDiscretization

    Used for continuous Non-Root nodes whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support
        sigma: variance of the normal distribution used for appriximate initial continuous distribution

```
struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Number}
    sigma::Union{Number,Nothing}

    function ApproximatedDiscretization(intervals::Vector{<:Number}, sigma::Union{Number,Nothing})
        if sort(intervals) != intervals
            error("interval values $intervals are not sorted")
        end
        new(intervals, sigma)
    end
end

function Base.isequal(discretization1::ApproximatedDiscretization, discretization2::ApproximatedDiscretization)
    discretization1.intervals == discretization2.intervals
end

function Base.hash(discretization::ApproximatedDiscretization, h::UInt)
    h = hash(discretization.intervals, h)
    return h
end

ApproximatedDiscretization() = ApproximatedDiscretization(Vector{Number}(), nothing)