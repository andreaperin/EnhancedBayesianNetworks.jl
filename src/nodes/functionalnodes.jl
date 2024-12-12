@auto_hash_equals struct ContinuousFunctionalNode <: AbstractContinuousNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::AbstractMonteCarlo,
        discretization::ApproximatedDiscretization
    )
        new(name, wrap(models), simulation, discretization)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
    simulation::AbstractMonteCarlo
)
    ContinuousFunctionalNode(name, models, simulation, ApproximatedDiscretization())
end

_is_root(node::ContinuousFunctionalNode) = false

@auto_hash_equals struct DiscreteFunctionalNode <: AbstractDiscreteNode
    name::Symbol
    models::Vector{<:UQModel}
    performance::Function
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        new(name, wrap(models), performance, simulation, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
    performance::Function,
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
)
    DiscreteFunctionalNode(name, models, performance, simulation, Dict{Symbol,Vector{Parameter}}())
end

_is_root(node::DiscreteFunctionalNode) = false

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}