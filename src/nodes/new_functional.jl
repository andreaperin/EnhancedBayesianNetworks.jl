@auto_hash_equals struct NewContinuousFunctionalNode
    name::Symbol
    models::Vector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization
end

function NewContinuousFunctionalNode(
    name::Symbol,
    models::Vector{<:UQModel},
    simulation::AbstractMonteCarlo
)
    discretization = ApproximatedDiscretization()
    NewContinuousFunctionalNode(name, models, simulation, discretization)
end

@auto_hash_equals struct NewDiscreteFunctionalNode
    name::Symbol
    models::Vector{<:UQModel}
    performance::Function
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
    parameters::Dict{Symbol,Vector{Parameter}}
end

function NewDiscreteFunctionalNode(
    name::Symbol,
    models::Vector{<:UQModel},
    performance::Function,
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    NewDiscreteFunctionalNode(name, models, performance, simulation, parameters)
end