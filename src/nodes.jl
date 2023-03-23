using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end
const global Assignment = Union{Vector{Dict{Union{NodeName,<:AbstractNode},Union{Int,Symbol}}},Nothing}

mutable struct EvidenceTable
    evidence::Assignment
    distribution::Union{<:Distribution,FunctionalModelCPD}
end

function _build_evidencetable_from_cpd(cpd::RootCPD)
    [EvidenceTable(nothing, cpd.distributions[1])]
end

"""
    Definition of Node Struct for RootCPD and CategoricalCPD (No continuous parents)
"""

struct RootNode <: AbstractNode
    cpd::RootCPD
    parents::Vector{<:AbstractNode}
    type::String
    evidence_table::Vector{EvidenceTable}

    function RootNode(cpd::RootCPD, parents::Vector{<:AbstractNode}, type::String, evidence_table::Vector{EvidenceTable})
        isempty(parents) ? new(cpd, parents, type, evidence_table) : throw(DomainError(cpd.target, "RootNode have no parents"))
    end
end

function RootNode(cpd::RootCPD)
    parents = Vector{AbstractNode}()
    type = _get_type_of_cpd(cpd)
    evidence_table = _build_evidencetable_from_cpd(cpd)
    RootNode(cpd, parents, type, evidence_table)
end


function _build_evidencetable_from_cpd(cpd::CategoricalCPD, parents::Vector{<:AbstractNode})
    f_e = (tup, pare) -> [Dict(name(pare[i]) => tup[i]) for i in range(1, length(tup))]
    f_t = (evid, dist) -> EvidenceTable(evid, dist)
    combinations = _get_nodes_combinations(parents)
    evidences = f_e.(combinations, repeat([parents], length(combinations)))
    evidence_table = f_t.(evidences, cpd.distributions)
    return evidence_table
end


struct StdNode <: AbstractNode
    cpd::CategoricalCPD
    parents::Vector{<:AbstractNode}
    type::String
    evidence_table::Vector{EvidenceTable}

    function StdNode(cpd::CategoricalCPD, parents::Vector{<:AbstractNode}, type::String, evidence_table::Vector{EvidenceTable})
        ~isempty(filter(x -> x.type == "continuous", parents)) ? throw(DomainError(cpd.target, "CategoricalCPD is for discrete parents only")) : nothing
        _get_number_of_discretestates.(parents) == cpd.parental_ncategories ? new(cpd, parents, type, evidence_table) : throw(DomainError(cpd.target, "parental_ncategories - parents discrete states missmatch"))
    end
end

function StdNode(cpd::CategoricalCPD, parents::Vector{<:AbstractNode})
    type = _get_type_of_cpd(cpd)
    evidence_table = _build_evidencetable_from_cpd(cpd, parents)
    StdNode(cpd, parents, type, evidence_table)
end


function _build_evidencetable_from_cpd(cpd::FunctionalCPD, parents::Vector{<:AbstractNode})
    f_e = (tup, pare) -> [Dict(name(pare[i]) => tup[i]) for i in range(1, length(tup))]
    f_t = (evid, dist) -> EvidenceTable(evid, dist)
    discrete_parents = filter(x -> x.type == "discrete", parents)
    combinations = _get_nodes_combinations(discrete_parents)
    evidences = f_e.(combinations, repeat([discrete_parents], length(combinations)))
    evidence_table = f_t.(evidences, cpd.distributions)
    return evidence_table
end

mutable struct FunctionalNode <: AbstractNode
    cpd::FunctionalCPD
    parents::Vector{<:AbstractNode}
    type::String
    evidence_table::Vector{EvidenceTable}
end


function FunctionalNode(cpd::FunctionalCPD, parents::Vector{<:AbstractNode}, type::String)
    evidence_table = _build_evidencetable_from_cpd(cpd, parents)
    FunctionalNode(cpd, parents, type, evidence_table)
end


"""
    Functions Section
"""

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



## TODO old function Section


# """ 
#     1) discrete_states returns:
#             states for discrete nodes 
#                 Vector{Dict{state_description_Symbol, state_number}}
#     2) continuous_distributions returns:
#             possible distributions for continuous nodes 
#                 Vector{Dict{distribution_description, distribution_number}}
# """

# function discrete_states(node::AbstractNode)
#     if node.type == "continuous"
#         throw(DomainError(node, "This is a continuous node"))
#     else
#         if node.cpd isa RootCPD
#             return node.cpd.distributions.map.n2d
#         else
#             return node.cpd.distributions[1].map.n2d
#         end
#     end
# end



# function continuous_distributions(node::AbstractNode)
#     if node.type == "discrete"
#         throw(DomainError(node, "This is a continuous node"))
#     else
#         return node.cpd.distributions
#     end
# end

# function get_discrete_parents(node::T) where {T<:AbstractNode}
#     discrete_parents = copy(node.parents)
#     return filter(x -> x.type == "discrete", discrete_parents)
# end

# function get_continuous_parents(node::T) where {T<:AbstractNode}
#     continuous_parents = copy(node.parents)
#     return filter(x -> x.type == "continuous", continuous_parents)
# end

# function nodes_split(nodes::Vector{T}) where {T<:AbstractNode}
#     discrete_parents = filter(x -> x.type == "discrete", nodes)
#     continuous_parents = filter(x -> x.type == "continuous", nodes)
#     continuous_nonroot_parents = filter(x -> ~isa(x.cpd, RootCPD), continuous_parents)
#     continuous_root_parents = filter(x -> isa(x.cpd, RootCPD), continuous_parents)
#     return discrete_parents, continuous_nonroot_parents, continuous_root_parents
# end

# function get_discrete_ancestors(nodes::Vector{T}) where {T<:AbstractNode}

# end

# function get_statesordistributions(node::T) where {T<:AbstractNode}
#     if node.type == "continuous"
#         return Dict(node => node.cpd.distributions)
#     elseif node.type == "discrete"
#         ~isa(node.cpd, RootCPD) ? result = Dict(node => collect(values(node.cpd.distributions[1].map.d2n))) : result = Dict(node => collect(values(node.cpd.distributions.map.d2n)))
#         return result
#     end
# end

# function get_combinations(nodes::Vector{T}) where {T<:AbstractNode}
#     states_dictionary = get_statesordistributions.(nodes)
#     to_combine = []
#     reference_vector = Vector{T}()
#     for node in states_dictionary
#         push!(to_combine, collect(values(node))[1])
#         push!(reference_vector, collect(keys(node))[1])
#     end
#     return vec(collect(Iterators.product(to_combine...))), reference_vector
# end

# function get_states_mapping_dict(node::T) where {T<:AbstractNode}
#     mapping = Dict{NodeName,Dict{}}()
#     if node.type == "discrete"
#         mapping[name(node)] = node.cpd.distributions[1].map.n2d
#     elseif node.type == "continuous"
#         mapping[name(node)] = Dict(node.cpd.distributions => node.cpd.distributions)
#     end
#     return mapping
# end

# function _map_state_to_integer(states::Tuple, nodes::Vector{T}) where {T<:AbstractNode}
#     new_states = []
#     mapping = get_states_mapping_dict(nodes)
#     for i in range(1, length(states))
#         if isa(states[i], Symbol)
#             push!(new_states, mapping[name(nodes[i])][states[i]])
#         else
#             push!(new_states, states[i])
#         end
#     end
#     return Tuple(new_states)
# end



# function get_ancestors(nodes::Vector{T}) where {T<:AbstractNode}
#     discrete, cont_nonroot, cont_root = nodes_split(nodes)
#     append!(discrete, cont_root)
#     while ~isempty(cont_nonroot)
#         new_nodes = Vector{AbstractNode}()
#         for single_cont_nonroot in cont_nonroot
#             append!(new_nodes, single_cont_nonroot.parents)
#         end
#         discrete_new, cont_nonroot_new, cont_root_new = nodes_split(new_nodes)
#         append!(discrete, discrete_new)
#         append!(discrete, cont_root_new)
#         cont_nonroot = cont_nonroot_new
#     end
#     return unique(discrete)
# end



# function convert_prob_dict_2_node_prob_dict(prob_dict::Union{ProbabilityDictionary,CPDProbabilityDictionaryFunctional}, discrete_ancestors::Vector{T}, all_parents::Vector{T}) where {T<:AbstractNode}
#     f1 = (nodename, p) -> filter(p -> name(p) == nodename, p)
#     evid = Dict()
#     for (key, value) in prob_dict.evidence
#         node = f1(key, discrete_ancestors)[1]
#         evid[node] = value
#     end
#     if isa(prob_dict, CPDProbabilityDictionary)
#         node_prob_dict = ProbabilityDictionary(tuple(evid, prob_dict.distribution))
#     else
#         new_node_correlationcopula = convert_correlation_2_node_correlation.(prob_dict.distribution.correlation, repeat([all_parents], length(prob_dict.distribution.correlation)))
#         new_node_srp = NodeSystemReliabilityProblem(prob_dict.distribution.model, prob_dict.distribution.parameters, prob_dict.distribution.performance, new_node_correlationcopula, prob_dict.distribution.simulation,)
#         node_prob_dict = ProbabilityDictionaryFunctional(tuple(evid, new_node_srp))
#     end
#     return node_prob_dict
# end

# function convert_correlation_2_node_correlation(correlation::CPDCorrelationCopula, parents::Vector{T}) where {T<:AbstractNode}
#     f1 = (nodename, p) -> filter(p -> name(p) == nodename, p)
#     new_nodes = Vector()
#     for n in correlation.nodes
#         push!(new_nodes, f1(n, parents)[1])
#     end
#     return NodeCorrelationCopula(new_nodes, correlation.copula, correlation.name)
# end


# """
# The cpd of a StdNode(Continuous or Discrete) given an assignment => Discrete Parents nodes only!!
# """

# function to_numerical_values(evidence::ProbabilityDictionaryEvidence)
#     convertedevidence = Dict()
#     if isa(evidence, Nothing)
#         convertedevidence = evidence
#     else
#         for (key, val) in evidence
#             if ~isa(val, Number)
#                 convertedevidence[key] = get_states_mapping_dict(key)[name(key)][val]
#             else
#                 val ∈ collect(values(get_states_mapping_dict(key)[name(key)])) ? convertedevidence[key] = val : throw(DomainError(evidence, "assigned evidence number is not available in the node"))
#             end
#         end
#     end
#     return convertedevidence
# end

# function evaluate_nodecpd_with_evidence_standard(node_to_eval::StdNode, evidence::ProbabilityDictionaryEvidence)
#     check = x -> x.type == "discrete"
#     if all([check(i) for i in keys(evidence)])
#         # convert symbolic evicences to numerical evidences
#         evidence = to_numerical_values(evidence)
#         f = x -> CPDProbabilityDictionary(to_numerical_values(x.evidence) => x.distribution)
#         converted_node_to_eval_prob_dict = f.(node_to_eval.node_prob_dict)
#         node_parents = node_to_eval.parents
#         assigned_nodes = collect(keys(evidence))
#         undefined_assignmets = node_parents[node_parents.∉[assigned_nodes]]
#         useless_assignmets = assigned_nodes[assigned_nodes.∉[node_parents]]
#         evidence_to_compare = evidence
#         if ~isempty(useless_assignmets)
#             useless = name.(useless_assignmets)
#             println("Evidences on $useless should be treated with join pdf")
#             for i in useless_assignmets
#                 pop!(evidence_to_compare, i)
#             end
#         end
#         if ~isempty(undefined_assignmets)
#             undefined = name.(undefined_assignmets)
#             println("Following cpds are defined for each value of $undefined")
#         end
#         distribution_under_given_evidence = []
#         for element in converted_node_to_eval_prob_dict
#             if all([element.evidence[key] == val for (key, val) in evidence_to_compare])
#                 push!(distribution_under_given_evidence, element.distribution)
#             end
#         end
#         return distribution_under_given_evidence
#     else
#         throw(DomainError("evaluate node cpd with evidence do not allow nodes with continuous parents"))
#     end
# end


# # ##TODO starts from here for building UQinputs when continuous parent of the functional node is not a RootNode!

# function build_UQInputs_singlecase(node::FunctionalNode, prob_dict::ProbabilityDictionaryFunctional)
#     continuous_parents = get_continuous_parents(node)
#     non_correlated_continuous_parents = Vector{UQInput}()
#     joint_rvs = Vector{UQInput}()
#     for copula in prob_dict.distribution.correlation
#         continuous_parents = setdiff(continuous_parents, copula.nodes)
#         if ~isempty(copula.nodes)
#             rvs = [RandomVariable(evaluate_nodecpd_with_evidence_standard(x, prob_dict.evidence)[1][:all_states], x.cpd.target) for x in copula.nodes]
#             if all(length.([evaluate_nodecpd_with_evidence_standard(x, prob_dict.evidence) for x in copula.nodes]) .== 1)
#                 push!(joint_rvs, JointDistribution(rvs, copula.copula))
#             else
#                 throw(DomainError(prob_dict.evidence, "Not sufficient evidence condition"))
#             end
#         end
#     end
#     rvs = [RandomVariable(evaluate_nodecpd_with_evidence_standard(x, prob_dict.evidence)[1][:all_states], x.cpd.target) for x in continuous_parents]
#     if all(length.([evaluate_nodecpd_with_evidence_standard(x, prob_dict.evidence) for x in continuous_parents]) .== 1)
#         append!(non_correlated_continuous_parents, rvs)
#     else
#         throw(DomainError(prob_dict.evidence, "Not sufficient evidence condition"))
#     end
#     return vcat(joint_rvs, non_correlated_continuous_parents)
# end
