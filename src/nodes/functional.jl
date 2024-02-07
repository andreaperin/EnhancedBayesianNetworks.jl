@auto_hash_equals struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        simulation::AbstractMonteCarlo,
        discretization::ApproximatedDiscretization
    )
        verify_functionalnode_parents(parents)

        new(name, parents, models, simulation, discretization)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    simulation::AbstractMonteCarlo
)

    discretization = ApproximatedDiscretization()
    ContinuousFunctionalNode(name, parents, models, simulation, discretization)
end

@auto_hash_equals struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    performance::Function
    simulation::AbstractSimulation
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        performance::Function,
        simulation::AbstractSimulation,
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        if isempty(filter(x -> isa(x, FunctionalNode), parents))
            verify_functionalnode_parents(parents)
        end
        new(name, parents, models, performance, simulation, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    performance::Function,
    simulation::AbstractSimulation
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, parents, models, performance, simulation, parameters)
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}