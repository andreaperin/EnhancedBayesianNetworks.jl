@auto_hash_equals struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    models::Vector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization
end

function ContinuousFunctionalNode(
    name::Symbol,
    models::Vector{<:UQModel},
    simulation::AbstractMonteCarlo
)
    discretization = ApproximatedDiscretization()
    ContinuousFunctionalNode(name, models, simulation, discretization)
end

@auto_hash_equals struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    models::Vector{<:UQModel}
    performance::Function
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
    parameters::Dict{Symbol,Vector{Parameter}}
end

function DiscreteFunctionalNode(
    name::Symbol,
    models::Vector{<:UQModel},
    performance::Function,
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, models, performance, simulation, parameters)
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}