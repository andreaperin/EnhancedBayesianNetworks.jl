abstract type AbstractNode end
abstract type DiscreteNode <: AbstractNode end
abstract type ContinuousNode <: AbstractNode end

include("root.jl")
include("standard.jl")
# include("functional.jl")
