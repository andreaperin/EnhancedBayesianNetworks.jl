@auto_hash_equals struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    models::Vector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::AbstractMonteCarlo,
        discretization::ApproximatedDiscretization
    )
        models = wrap(models)
        new(name, models, simulation, discretization)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
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

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        models = wrap(models)
        new(name, models, performance, simulation, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
    performance::Function,
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, models, performance, simulation, parameters)
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}