struct StructuralReliabilityProblem
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

struct StructuralReliabilityPDF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    simulation::MonteCarlo
end
struct StructuralReliabilityProblemNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},Union{StructuralReliabilityProblem,StructuralReliabilityPDF}}
    parameters::Dict{Symbol,Vector{Parameter}}
end

