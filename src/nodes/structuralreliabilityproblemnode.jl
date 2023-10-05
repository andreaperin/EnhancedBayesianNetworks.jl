struct StructuralReliabilityProblemPDF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    simulation::MonteCarlo
end

function Base.isequal(srp1::StructuralReliabilityProblemPDF, srp2::StructuralReliabilityProblemPDF)
    srp1.models == srp2.models && srp1.inputs == srp2.inputs && srp1.simulation == srp2.simulation
end

function Base.hash(srp::StructuralReliabilityProblemPDF, h::UInt)
    h = hash(srp.model, h)
    h = hash(srp.inputs, h)
    h = hash(srp.simulation, h)
    return h
end

struct StructuralReliabilityProblemPMF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

function Base.isequal(srp1::StructuralReliabilityProblemPMF, srp2::StructuralReliabilityProblemPMF)
    srp1.models == srp2.models && srp1.inputs == srp2.inputs && srp1.performance == srp2.performance && srp1.simulation == srp2.simulation
end

function Base.hash(srp::StructuralReliabilityProblemPMF, h::UInt)
    h = hash(srp.model, h)
    h = hash(srp.inputs, h)
    h = hash(srp.performance, h)
    h = hash(srp.simulation, h)
    return h
end


struct ContinuousStructuralReliabilityProblemNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},StructuralReliabilityProblemPDF}
    discretization::AbstractDiscretization
end

function Base.isequal(node1::ContinuousStructuralReliabilityProblemNode, node2::ContinuousStructuralReliabilityProblemNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && isequal(node1.srps, node2.srps) && isequal(node1.discretization, node2.discretization)
end

function Base.hash(node::ContinuousStructuralReliabilityProblemNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.srps, h)
    h = hash(node.discretization, h)
    return h
end

struct DiscreteStructuralReliabilityProblemNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},StructuralReliabilityProblemPMF}
    parameters::Dict{Symbol,Vector{Parameter}}
end

function Base.isequal(node1::DiscreteStructuralReliabilityProblemNode, node2::DiscreteStructuralReliabilityProblemNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && isequal(node1.srps, node2.srps) && node1.parameters == node2.parameters
end

function Base.hash(node::DiscreteStructuralReliabilityProblemNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.srps, h)
    h = hash(node.parameters, h)
    return h
end


const global StructuralReliabilityProblemNode = Union{DiscreteStructuralReliabilityProblemNode,ContinuousStructuralReliabilityProblemNode}