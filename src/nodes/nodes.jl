abstract type AbstractNode end
abstract type DiscreteNode <: AbstractNode end
abstract type ContinuousNode <: AbstractNode end

const AbstractSimulation = Union{AbstractMonteCarlo,LineSampling,SubSetSimulation}

# include("evidencedistribution.jl")
include("functional.jl")
include("root.jl")
include("child.jl")
include("structuralreliabilityproblemnode.jl")

