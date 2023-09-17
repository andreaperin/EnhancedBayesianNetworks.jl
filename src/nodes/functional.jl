mutable struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    simulations::AbstractMonteCarlo
    samples::Dict{Vector{Symbol},Any}
    distributions::Dict{Vector{Symbol},EmpiricalDistribution}

    function ContinuousFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        simulations::AbstractMonteCarlo,
        samples::Dict{Vector{Symbol},Any},
        distributions::Dict{Vector{Symbol},EmpiricalDistribution}
    )
        verify_functionalnode_parents(parents)

        new(name, parents, models, simulations, samples, distributions)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    simulations::AbstractMonteCarlo
)
    samples = Dict{Vector{Symbol},Any}()
    distributions = Dict{Vector{Symbol},Distribution}()
    ContinuousFunctionalNode(name, parents, models, simulations, samples, distributions)
end

## Get all the parents random variable if the evidence gives uniques random variables 
function get_randomvariable(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    return mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents)
end

function Base.isequal(node1::ContinuousFunctionalNode, node2::ContinuousFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models && node1.simulations == node.simulations
end

function Base.hash(node::ContinuousFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)

    return h
end

mutable struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Vector{<:UQModel}
    performances::Function
    simulations::AbstractSimulation
    states::Dict{Vector{Symbol},Dict{Symbol,Real}}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Vector{<:UQModel},
        performances::Function,
        simulations::AbstractSimulation,
        states::Dict{Vector{Symbol},Dict{Symbol,Real}},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        if isempty(filter(x -> isa(x, FunctionalNode), parents))
            verify_functionalnode_parents(parents)
        end
        new(name, parents, models, performances, simulations, states, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    performances::Function,
    simulations::AbstractSimulation,
    parameters::Dict{Symbol,Vector{Parameter}}
)
    states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    DiscreteFunctionalNode(name, parents, models, performances, simulations, states, parameters)
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    performances::Function,
    simulations::AbstractSimulation
)
    states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, parents, models, performances, simulations, states, parameters)
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Vector{<:UQModel},
    performances::Function,
    simulations::AbstractSimulation,
    states::Dict{Vector{Symbol},Dict{Symbol,Real}}
)

    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteFunctionalNode(name, parents, models, performances, simulations, states, parameters)
end

function Base.isequal(node1::DiscreteFunctionalNode, node2::DiscreteFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models && node1.performances == node2.performances && node1.simulations == node2.simulations
end

function Base.hash(node::DiscreteFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)
    h = hash(node.performances, h)
    h = hash(node.simulations, h)
    return h
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}