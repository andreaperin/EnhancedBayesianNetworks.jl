using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end
abstract type NodeToBe <: AbstractNode end
abstract type Node <: AbstractNode end
"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const global Assignment = Dict{Node,Any}

struct StdNode <: Node
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    ## node_prob_dict structure:
    #   NamedTuple(
    #       :evidence = Dict(node => number_of_specific_state)
    #       :distribution =  Distribution
    #   )
    node_prob_dict::Vector{ProbabilityDictionary}
    function StdNode(cpd::CPD, parents::Vector{T}, type::String, node_prob_dict::Vector{ProbabilityDictionary}) where {T<:AbstractNode}
        node_name = cpd.target
        if ~isa(cpd, RootCPD)
            ## Checks:
            #    - No continuous parents
            #    - name(parents) - CPD.parents
            #    - parents - parental_ncategories
            #    - parents_states - parental_ncategories
            isempty(filter(x -> x.type == "continuous", parents)) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(node_name, "CategoricalCPD is for node with discrete parents only!"))
            cpd.parents == name.(parents) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(node_name, "Assigned parents are not equals to the one of CPD"))
            length(parents) == length(cpd.parental_ncategories) ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(node_name, "parents mismatch in CPD for discrete parents and parental_ncategories in $node_name"))
            get_numberofstates.(parents) == cpd.parental_ncategories ? new(cpd, parents, type, node_prob_dict) : throw(DomainError(node_name, "Missmatch in parents categories and (manually defined) parental_ncategories in $node_name"))
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
            node_prob_dict = [ProbabilityDictionary(((parents => nothing), Dict("all states" => cpd.distributions)))]
        else
            node_name = cpd.target
            throw(DomainError(node_name, "Missing parents argument as vector of AbstractNodes for $node_name"))
        end
        StdNode(cpd, parents, type, node_prob_dict)
    end

    function StdNode(cpd::CPD, parents::Vector{T}) where {T<:AbstractNode}
        node_name = cpd.target
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
            node_prob_dict = [ProbabilityDictionary(((parents => nothing), Dict("all states" => cpd.distributions)))]
            isempty(parents) ? StdNode(cpd, parents, type, node_prob_dict) : throw(DomainError(node_name, "a RootNode cannot have parents"))
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
            # Building node_prob_dict for Non-RootNode
            node_prob_dict = build_node_prob_dict_fromCPD.(cpd.prob_dict, repeat([parents], length(cpd.prob_dict)))
            StdNode(cpd, parents, type, node_prob_dict)
        end
    end
end

mutable struct FunctionalNode <: Node
    cpd::FunctionalCPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    function FunctionalNode(cpd::FunctionalCPD, parents::Vector{T}, type::String) where {T<:AbstractNode}
        ## Check parents coherence between CPD and FunctionalNode
        if name.(parents) != cpd.parents
            throw(ArgumentError(cpd.target, "Missmatch in parents assigned in CPD and assigned in Node Struct"))
        end
    end
end

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

function build_node_prob_dict_fromCPD(prob_dict::ProbabilityDictionary, parents::Vector{T}) where {T<:AbstractNode}
    f1 = (nodename, p) -> filter(p -> name(p) == nodename, p)
    evid = Dict()
    for key in collect(keys(prob_dict.evidence))
        node = f1(key, parents)[1]
        evid[node] = prob_dict.evidence[key]
    end
    node_prob_dict = ProbabilityDictionary(tuple(evid, prob_dict.distribution))
    return node_prob_dict
end
