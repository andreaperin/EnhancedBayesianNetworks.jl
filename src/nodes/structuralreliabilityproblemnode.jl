struct StructuralReliabilityProblem
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

struct StructuralReliabilityProblemNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},StructuralReliabilityProblem}
    parameters::Dict{Symbol,Vector{Parameter}}
end

