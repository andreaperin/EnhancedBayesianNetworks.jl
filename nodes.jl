using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end


"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const global Assignment = Dict{AbstractNode,Any}
"""
    Definition of the Node_SystemReliabilityPorblem
"""
struct NodeCorrelationCopula
    nodes::Vector{AbstractNode}
    copula::Union{GaussianCopula,Nothing}
    name::NodeName
end

function NodeCorrelationCopula()
    nodes = Vector{AbastractNodes}()
    copula = nothing
    name = NodeName()
    new(nodes, copula, name)
end

struct NodeSystemReliabilityProblem <: SystemReliabilityProblem
    model::Union{Array{<:UQModel},UQModel}
    parameters::Vector{Parameter}
    performance::Function
    correlation::Vector{NodeCorrelationCopula}
    simulation::Any
    function NodeSystemReliabilityProblem(
        model::Union{Array{<:UQModel},UQModel},
        parameters::Vector{Parameter},
        performance::Function,
        correlation::Vector{NodeCorrelationCopula},
        simulation::Any
    )
        new(model, parameters, performance, correlation, simulation)
    end

    function NodeSystemReliabilityProblem(
        model::Union{Array{<:UQModel},UQModel},
        parameters::Vector{Parameter},
        performance::Function,
        simulation::Any
    )
        correlation = [NodeCorrelationCopula()]
        new(model, parameters, performance, correlation, simulation)
    end
end


const global NodeProbabilityDictionary = NamedTuple{(:evidence, :distribution),Tuple{ProbabilityDictionaryEvidence,CPDProbabilityDictionaryDistribution}}
const global NodeProbabilityDictionaryFunctional = NamedTuple{(:evidence, :distribution),Tuple{ProbabilityDictionaryEvidence,NodeSystemReliabilityProblem}}

"""
    Definition of the StdNode Struct
"""
struct StdNode <: AbstractNode
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    ## node_prob_dict structure:
    #   NamedTuple(
    #       :evidence = Dict(node => number_of_specific_state)
    #       :distribution =  Distribution
    #   )
    node_prob_dict::Vector{NodeProbabilityDictionary}
    function StdNode(cpd::CPD, parents::Vector{T}, type::String, node_prob_dict::Vector{NodeProbabilityDictionary}) where {T<:AbstractNode}
        if ~isa(cpd, RootCPD)
            ## Checks:
            #    - No continuous parents
            #    - name(parents as nodes) - CPD.parents as nodenames
            #    - number of parents (as node)- length of CPD's parental_ncategories
            #    - number of parents (as nodes) states - values of CPD's parental_ncategories
            isempty(filter(x -> x.type == "continuous", parents)) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "CategoricalCPD is for node with discrete parents only!"))
            cpd.parents == name.(parents) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Assigned parents are not equals to the one of CPD"))
            length(parents) == length(cpd.parental_ncategories) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "parents mismatch in CPD for discrete parents and parental_ncategories"))
            get_numberofstates.(parents) == cpd.parental_ncategories ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Missmatch in parents categories and (manually defined) parental_ncategories"))
        else
            ## No check needed for RootNode
            new(cpd, parents, type, node_prob_dict)
        end
    end

    function StdNode(cpd::CPD)
        ```Function for Root Node only```
        parents = Vector{AbstractNode}()
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
            # Building node_prob_dict for RootNode
            node_prob_dict = cpd.prob_dict
        else
            node_name = cpd.target
            throw(DomainError(node_name, "Missing parents argument as vector of AbstractNodes for $node_name"))
        end
        StdNode(cpd, parents, type, node_prob_dict)
    end

    function StdNode(cpd::CPD, parents::Vector{T}) where {T<:AbstractNode}
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
            node_prob_dict = cpd.prob_dict
            isempty(parents) ? StdNode(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "a RootNode cannot have parents"))
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
            # Building node_prob_dict for Non-RootNode
            node_prob_dict = convert_prob_dict_2_node_prob_dict.(cpd.prob_dict, repeat([parents], length(cpd.prob_dict)), repeat([parents], length(cpd.prob_dict)))
            StdNode(cpd, parents, type, node_prob_dict)
        end
    end
end


mutable struct FunctionalNode <: AbstractNode
    cpd::FunctionalCPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    node_prob_dict::Union{Vector{NodeProbabilityDictionaryFunctional},Vector{CPDProbabilityDictionaryFunctional}}
    function FunctionalNode(
        cpd::FunctionalCPD,
        parents::Vector{T},
        type::String,
        node_prob_dict::Union{Vector{NodeProbabilityDictionaryFunctional},Vector{CPDProbabilityDictionaryFunctional}}
    ) where {T<:AbstractNode}
        discrete_parents = filter(x -> x.type == "discrete", parents)
        ## Checks on
        #    - parents as nodenames and parents as nodes coherence between CPD and FunctionalNode
        #    - number of parents as (Discrete) nodes - number of parental_ncategories coherence 
        #    - number of parents as (Discrete) nodes - length of the evidence in CPD's prob_dict
        #    - number of parents as (Discrete) nodes states - values inside CPD's parental_ncategories   
        name.(parents) == cpd.parents ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Missmatch in parents assigned in CPD and assigned in Node Struct"))
        length(discrete_parents) == length(cpd.parental_ncategories) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Number of discrete parents in not equals to the length of CPD.parental_ncategories"))
        length(discrete_parents) == length(cpd.prob_dict[1].evidence) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Number of discrete parents in not equals to the length of CPD.prob_dict.evidence"))
        get_numberofstates.(discrete_parents) == cpd.parental_ncategories ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(cpd.target, "Missmatch in parents categories and (manually defined) parental_ncategories in $node_name"))
    end

    function FunctionalNode(cpd::FunctionalCPD, parents::Vector{T}, type::String) where {T<:AbstractNode}
        discrete_parents = filter(x -> x.type == "discrete", parents)
        # Building node_prob_dict for Non-FunctionalNode
        node_prob_dict = convert_prob_dict_2_node_prob_dict.(cpd.prob_dict, repeat([discrete_parents], length(cpd.prob_dict)), repeat([parents], length(cpd.prob_dict)))
        FunctionalNode(cpd, parents, type, node_prob_dict)
    end
end



"""
    Functions Section
"""

function name(node::T) where {T<:AbstractNode}
    return node.cpd.target
end

""" 
    1) discrete_states returns:
            states for discrete nodes 
                Vector{Dict{state_description_Symbol, state_number}}
    2) continuous_distributions returns:
            possible distributions for continuous nodes 
                Vector{Dict{distribution_description, distribution_number}}
"""

function discrete_states(node::AbstractNode)
    if node.type == "continuous"
        throw(DomainError(node, "This is a continuous node"))
    else
        if node.cpd isa RootCPD
            return node.cpd.distributions.map.n2d
        else
            return node.cpd.distributions[1].map.n2d
        end
    end
end

function get_numberofstates(node::AbstractNode)
    if node.type == "continuous"
        throw(DomainError(node, "This is a continuous node"))
    else
        if node.cpd isa RootCPD
            return length(node.cpd.distributions.items)
        else
            return length(node.cpd.distributions[1].items)
        end
    end
end

function continuous_distributions(node::AbstractNode)
    if node.type == "discrete"
        throw(DomainError(node, "This is a continuous node"))
    else
        return node.cpd.distributions
    end
end

function get_discrete_parents(node::T) where {T<:AbstractNode}
    discrete_parents = copy(node.parents)
    return filter(x -> x.type == "discrete", discrete_parents)
end

function get_continuous_parents(node::T) where {T<:AbstractNode}
    continuous_parents = copy(node.parents)
    return filter(x -> x.type == "continuous", continuous_parents)
end

function nodes_split(nodes::Vector{T}) where {T<:AbstractNode}
    discrete_parents = filter(x -> x.type == "discrete", nodes)
    continuous_parents = filter(x -> x.type == "continuous", nodes)
    continuous_nonroot_parents = filter(x -> ~isa(x.cpd, RootCPD), continuous_parents)
    continuous_root_parents = filter(x -> isa(x.cpd, RootCPD), continuous_parents)
    return discrete_parents, continuous_nonroot_parents, continuous_root_parents
end

function get_statesordistributions(node::T) where {T<:AbstractNode}
    if node.type == "continuous"
        return Dict(node => node.cpd.distributions)
    elseif node.type == "discrete"
        ~isa(node.cpd, RootCPD) ? result = Dict(node => collect(values(node.cpd.distributions[1].map.d2n))) : result = Dict(node => collect(values(node.cpd.distributions.map.d2n)))
        return result
    end
end

function get_combinations(nodes::Vector{T}) where {T<:AbstractNode}
    states_dictionary = get_statesordistributions.(nodes)
    to_combine = []
    reference_vector = Vector{T}()
    for node in states_dictionary
        push!(to_combine, collect(values(node))[1])
        push!(reference_vector, collect(keys(node))[1])
    end
    return vec(collect(Iterators.product(to_combine...))), reference_vector
end

function get_ancestors(node::T) where {T<:AbstractNode}
    parents = node.parents
    discrete, cont_nonroot, cont_root = nodes_split(parents)
    append!(discrete, cont_root)
    while ~isempty(cont_nonroot)
        new_parents = Vector{AbstractNode}()
        for single_cont_nonroot in cont_nonroot
            append!(new_parents, single_cont_nonroot.parents)
        end
        discrete_new, cont_nonroot_new, cont_root_new = nodes_split(new_parents)
        append!(discrete, discrete_new)
        append!(discrete, cont_root_new)
        cont_nonroot = cont_nonroot_new
    end
    ancestors = unique(discrete)
end

function get_states_mapping_dict(node::T) where {T<:AbstractNode}
    mapping = Dict{NodeName,Dict{}}()
    if node.type == "discrete"
        isa(node.cpd, RootCPD) ? mapping[name(node)] = node.cpd.distributions.map.n2d : mapping[name(node)] = node.cpd.distributions[1].map.n2d
    elseif node.type == "continuous"
        mapping[name(node)] = Dict(node.cpd.distributions => node.cpd.distributions)
    end
    return mapping
end

function map_state_to_integer(states::Tuple, nodes::Vector{T}) where {T<:AbstractNode}
    new_states = []
    mapping = get_states_mapping_dict(nodes)
    for i in range(1, length(states))
        if isa(states[i], Symbol)
            push!(new_states, mapping[name(nodes[i])][states[i]])
        else
            push!(new_states, states[i])
        end
    end
    return Tuple(new_states)
end

function convert_prob_dict_2_node_prob_dict(prob_dict::Union{NodeProbabilityDictionary,CPDProbabilityDictionaryFunctional}, discrete_parents::Vector{T}, all_parents::Vector{T}) where {T<:AbstractNode}
    f1 = (nodename, p) -> filter(p -> name(p) == nodename, p)
    evid = Dict()
    for (key, value) in prob_dict.evidence
        node = f1(key, discrete_parents)[1]
        evid[node] = value
    end
    if isa(prob_dict, CPDProbabilityDictionary)
        node_prob_dict = NodeProbabilityDictionary(tuple(evid, prob_dict.distribution))
    else
        new_node_correlationcopula = convert_correlation_2_node_correlation.(prob_dict.distribution.correlation, repeat([all_parents], length(prob_dict.distribution.correlation)))
        new_node_srp = NodeSystemReliabilityProblem(prob_dict.distribution.model, prob_dict.distribution.parameters, prob_dict.distribution.performance, new_node_correlationcopula, prob_dict.distribution.simulation,)
        node_prob_dict = NodeProbabilityDictionaryFunctional(tuple(evid, new_node_srp))
    end
    return node_prob_dict
end

function convert_correlation_2_node_correlation(correlation::CPDCorrelationCopula, parents::Vector{T}) where {T<:AbstractNode}
    f1 = (nodename, p) -> filter(p -> name(p) == nodename, p)
    new_nodes = Vector()
    for n in correlation.nodes
        push!(new_nodes, f1(n, parents)[1])
    end
    return NodeCorrelationCopula(new_nodes, correlation.copula, correlation.name)
end


function build_UQInputs_singlecase(node::FunctionalNode, prob_dict::NodeProbabilityDictionaryFunctional)
    continuous_parents = get_continuous_parents(node)
    non_correlated_continuous_parents = Vector{UQInput}()
    joint_rvs = Vector{UQInput}()
    for copula in prob_dict.distribution.correlation
        continuous_parents = setdiff(continuous_parents, copula.nodes)
        if all([isa(n.cpd, RootCPD) for n in copula.nodes])
            rvs = [RandomVariable(x.cpd.distributions, x.cpd.target) for x in copula.nodes]
            push!(joint_rvs, JointDistribution(rvs, copula.copula))
        else
            throw(DomainError(copula.nodes, "Implement when a nodes for joint distribution is not a root node"))
        end
    end
    for node in continuous_parents
        if isa(node.cpd, RootCPD)
            push!(non_correlated_continuous_parents, RandomVariable(node.cpd.distributions, name(node)))
        else
            throw(DomainError(name(node), "Implement when a nodes for joint distribution is not a root node"))
        end
    end
    return vcat(joint_rvs, non_correlated_continuous_parents)
end

