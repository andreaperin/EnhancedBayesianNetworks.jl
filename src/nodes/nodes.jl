abstract type AbstractNode end

include("root.jl")
include("standard.jl")
include("functional.jl")

const global Assignment = Dict{NodeName,Int}

mutable struct EvidenceTable
    evidence::Assignment
    distribution::Union{<:AbstractDistribution,ModelWithName}
end

mutable struct ModelParameters
    node::NodeName
    model::Symbol
    parameters::Vector{Parameter}
end

ModelParameters() = ModelParameters(Symbol(), Symbol(), Parameter[])

mutable struct ModelParametersTable
    evidence::Assignment
    parameters::Vector{ModelParameters}
end

function _build_evidencetable_from_cpd(cpd::RootCPD)
    [EvidenceTable(Assignment(), cpd.distributions[1])]
end

function _build_evidencetable_from_cpd(cpd::C, parents::Vector{<:AbstractNode}) where {C<:CPD}
    f_e = (tup, pare) -> Dict([(name(pare[i]) => tup[i]) for i in range(1, length(tup))])
    f_t = (evid, dist) -> EvidenceTable(evid, dist)
    discrete_parents = filter(x -> x.type == "discrete", parents)
    combinations = _get_nodes_combinations(discrete_parents)
    evidences = f_e.(combinations, repeat([discrete_parents], length(combinations)))
    evidence_table = f_t.(evidences, cpd.distributions)
    return evidence_table
end

function _build_modelparametertable(cpd::Union{RootCPD,StdCPD}, parameters_vector::Vector{Vector{ModelParameters}})
    map(x -> ModelParametersTable(Dict([(cpd.target => findall(y -> y == x, parameters_vector)[1])]), x), parameters_vector)
end

function name(node::T) where {T<:AbstractNode}
    return node.cpd.target
end

function _get_number_of_discretestates(node::T) where {T<:AbstractNode}
    node.type == "continuous" ? throw(DomainError(node, "This is a continuous node")) : number = length(node.cpd.distributions[1].items)
    return number
end

function _get_states_or_distributions(node::T) where {T<:AbstractNode}
    if node.type == "continuous"
        return node.cpd.distributions
    elseif node.type == "discrete"
        return collect(values(node.cpd.distributions[1].map.d2n))
    end
end

function _get_model_parameters_given_evidence(assignment::Assignment, node::T) where {T<:AbstractNode}
    filter(
        el -> begin
            key = collect(keys(el.evidence))[1]
            return haskey(assignment, key) && el.evidence[key] == assignment[key]
        end,
        node.model_parameters
    )
end

function _get_distribution_table_given_evidence(assignment::Assignment, node::T) where {T<:AbstractNode}
    if length(node.evidence_table) == 1
        return node.evidence_table
    else
        return filter(
            el -> begin
                key = collect(keys(el.evidence))[1]
                return haskey(assignment, key) && el.evidence[key] == assignment[key]
            end,
            node.evidence_table
        )
    end
end

function _get_nodes_combinations(nodes::Vector{T}) where {T<:AbstractNode}
    a = vec(collect(values.(_get_states_or_distributions.(nodes))))
    sort!(vec(collect(Iterators.product(_map_states_to_integer.(a, nodes)...))))
end

function get_states_mapping_dict(node::T) where {T<:AbstractNode}
    mapping = Dict{NodeName,Dict{}}()
    if node.type == "discrete"
        mapping[name(node)] = node.cpd.distributions[1].map.n2d
    elseif node.type == "continuous"
        mapping[name(node)] = Dict(node.cpd.distributions => node.cpd.distributions)
    end
    return mapping
end

function _map_states_to_integer(states::Vector, node::T) where {T<:AbstractNode}
    new_states = []
    mapping = get_states_mapping_dict(node)
    for i in range(1, length(states))
        if isa(states[i], Symbol)
            push!(new_states, mapping[name(node)][states[i]])
        else
            push!(new_states, states[i])
        end
    end
    return new_states
end
