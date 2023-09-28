mutable struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    simulations::AbstractMonteCarlo
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        simulations::AbstractMonteCarlo,
        discretization::ApproximatedDiscretization
    )
        verify_functionalnode_parents(parents)

        new(name, parents, models, simulations, discretization)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    simulations::AbstractMonteCarlo
)

    discretization = ApproximatedDiscretization()
    ContinuousFunctionalNode(name, parents, models, simulations, discretization)
end

function get_randomvariable(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    return mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents)
end

function Base.isequal(node1::ContinuousFunctionalNode, node2::ContinuousFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models && node1.simulations == node2.simulations && node1.discretization.intervals == node2.discretization.intervals && node1.discretization.sigma == node2.discretization.sigma
end

function Base.hash(node::ContinuousFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)
    h = hash(node.simulations, h)
    h = hash(node.discretization.intervals, h)
    h = hash(node.discretization.sigma, h)
    return h
end
mutable struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    performance::Function
    simulations::AbstractSimulation
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        performance::Function,
        simulations::AbstractSimulation,
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        if isempty(filter(x -> isa(x, FunctionalNode), parents))
            verify_functionalnode_parents(parents)
        end
        new(name, parents, models, performance, simulations, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    performance::Function,
    simulations::AbstractSimulation
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, parents, models, performance, simulations, parameters)
end

function Base.isequal(node1::DiscreteFunctionalNode, node2::DiscreteFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models && node1.performance == node2.performance && node1.simulations == node2.simulations && node1.parameters == node2.parameters
end

function Base.hash(node::DiscreteFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)
    h = hash(node.performance, h)
    h = hash(node.simulations, h)
    h = hash(node.parameters, h)
    return h
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}