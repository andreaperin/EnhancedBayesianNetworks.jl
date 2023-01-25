using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end
abstract type Node <: AbstractNode end
abstract type ModelInput end
"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const global Assignment = Dict{Node,Any}

struct ModelNode <: Node
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    ## TODO check if this function works
    function ModelNode(target::NodeName, parents::Vector{T}, model::Vector{UQModel}, type::String) where {T<:AbstractNode}
        ancestors_states_combinations, parents_states_combination_reduced = get_evidences_vectors_for_modelnodes(parents)
        ##TODO go on from here (now we have the prob_dict entries and we need to create the SRP problem from here and then adjust them following the parents and states order)
    end
end


function get_evidences_vectors_for_modelnodes(parents::Vector{T}) where {T<:AbstractNode}
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
    ## Get Ancestors evidences vector
    ancestors_states_combinations, reference_vec = get_combinations(ancestors)
    evidence_over_ancestors = Vector{Assignment}()
    for state_combination in ancestors_states_combinations
        evidence = Dict()
        for i in range(1, length(reference_vec))
            evidence[reference_vec[i]] = state_combination[i]
        end
        push!(evidence_over_ancestors, evidence)
    end
    evidence_over_parents = Vector{Assignment}()
    parents_states_combination_reduced = []
    for single_evidence_over_ancestors in evidence_over_ancestors
        single_evidence_over_parents = Assignment()
        for node in parents
            if haskey(single_evidence_over_ancestors, node)
                single_evidence_over_parents[node] = single_evidence_over_ancestors[node]
            else
                single_evidence_over_parents[node] = collect(values(evaluate_nodecpd_with_evidence(node, single_evidence_over_ancestors)))
            end
        end
        push!(evidence_over_parents, single_evidence_over_parents)
        push!(parents_states_combination_reduced, Tuple(x for x in collect(values(single_evidence_over_parents))))
    end
    return ancestors_states_combinations, parents_states_combination_reduced
end


struct StdNode <: Node
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    ##TODO add to log
    function StdNode(cpd::CPD, parents::Vector{T}, type::String) where {T<:AbstractNode}
        node_name = cpd.target
        if ~isa(cpd, RootCPD)
            cpd.parents == name.(parents) ? new(cpd, parents, type) : throw(DomainError(node_name, "Assigned parents are not equals to the one of CPD"))
            length(filter(x -> x.type == "discrete", parents)) == length(cpd.parental_ncategories) ? new(cpd, parents, type) : throw(DomainError(node_name, "parents mismatch in CPD for discrete parents and parental_ncategories in $node_name"))
        else
            new(cpd, parents, type)
        end
    end

    function StdNode(cpd::CPD)
        ```Function for Root Node only```
        parents = Vector{AbstractNode}()
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
        end
        StdNode(cpd, parents, type)
    end
    function StdNode(cpd::CPD, parents::Vector{T}) where {T<:AbstractNode}
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
        end
        ##TODO add to log
        f = x -> findmax(collect(values(x)), dims=1)[1][1]
        discrete_parents = filter(x -> x.type == "discrete", parents)
        continuous_parents = filter(x -> x.type == "continuous", parents)
        discrete_parental_ncategories = f.(discrete_states.(discrete_parents))
        isempty(discrete_parental_ncategories) ? discrete_parental_ncategories = [1] : discrete_parental_ncategories = discrete_parental_ncategories
        continuous_parental_ncategories = length.(continuous_distributions.(continuous_parents))
        isempty(continuous_parental_ncategories) ? continuous_parental_ncategories = [1] : continuous_parental_ncategories = continuous_parental_ncategories
        theorical_parents_categories = prod(discrete_parental_ncategories) * prod(continuous_parental_ncategories)
        if theorical_parents_categories != prod(cpd.parental_ncategories)
            node_name = cpd.target
            throw(DomainError(node_name, "number of assigned cpds id not equal to parental categories $theorical_parents_categories"))
        else
            StdNode(cpd, parents, type)
        end
    end
end


struct ModelInputNode <: Node
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    map2uqinputs::Dict{Symbol,UQInput}
    ##TODO add to log
    function ModelInputNode(cpd::CPD, parents::Vector{T}, type::String, map2uqinputs::Dict{Symbol,UQInput}) where {T<:AbstractNode}
        node_name = cpd.target
        if ~isa(cpd, RootCPD)
            cpd.parents == name.(parents) ? new(cpd, parents, type, map2uqinputs) : throw(DomainError(node_name, "Assigned parents are not equals to the one of CPD"))
            length(filter(x -> x.type == "discrete", parents)) == length(cpd.parental_ncategories) ? new(cpd, parents, type, map2uqinputs) : throw(DomainError(node_name, "parents mismatch in CPD for discrete parents and parental_ncategories in $node_name"))
        else
            new(cpd, parents, type, map2uqinputs)
        end
    end

    function ModelInputNode(cpd::CPD, map2uqinputs::Dict{Symbol,UQInput})
        ```Function for Root Node only```
        parents = Vector{AbstractNode}()
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
        end
        ModelInputNode(cpd, parents, type, map2uqinputs)
    end
    function ModelInputNode(cpd::CPD, parents::Vector{T}, map2uqinputs::Dict{Symbol,UQInput}) where {T<:AbstractNode}
        if isa(cpd, RootCPD)
            type = isa(cpd.distributions, NamedCategorical) ? "discrete" : "continuous"
        elseif isa(cpd, CategoricalCPD)
            type = isa(cpd.distributions[1], NamedCategorical) ? "discrete" : "continuous"
        end
        ##TODO add to log
        f = x -> findmax(collect(values(x)), dims=1)[1][1]
        discrete_parents = filter(x -> x.type == "discrete", parents)
        continuous_parents = filter(x -> x.type == "continuous", parents)
        discrete_parental_ncategories = f.(discrete_states.(discrete_parents))
        isempty(discrete_parental_ncategories) ? discrete_parental_ncategories = [1] : discrete_parental_ncategories = discrete_parental_ncategories
        continuous_parental_ncategories = length.(continuous_distributions.(continuous_parents))
        isempty(continuous_parental_ncategories) ? continuous_parental_ncategories = [1] : continuous_parental_ncategories = continuous_parental_ncategories
        theorical_parents_categories = prod(discrete_parental_ncategories) * prod(continuous_parental_ncategories)
        if theorical_parents_categories != prod(cpd.parental_ncategories)
            node_name = cpd.target
            throw(DomainError(node_name, "number of assigned cpds id not equal to parental categories $theorical_parents_categories"))
        else
            ModelInputNode(cpd, parents, type, map2uqinputs)
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
    reference_vector = []
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

function get_states_mapping_dict(nodes::Vector{T}) where {T<:AbstractNode}
    mapping = Dict{NodeName,Dict{}}()
    for node in nodes
        if node.type == "discrete"
            isa(node.cpd, RootCPD) ? mapping[name(node)] = node.cpd.distributions.map.n2d : mapping[name(node)] = node.cpd.distributions[1].map.n2d
        elseif node.type == "continuous"
            mapping[name(node)] = Dict(node.cpd.distributions => node.cpd.distributions)
        end
    end
    return mapping
end

function map_state_to_integer(dict_to_be_mapped::Dict, nodes::Vector{T}) where {T<:AbstractNode}
    new_dict = Dict()
    mapping = get_states_mapping_dict(nodes)
    for (key, val) in dict_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            push!(new_key, mapping[nodes[collect(keys(nodes))[i]]][key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = val
    end
    return new_dict
end

function map_state_to_integer(vector_to_be_mapped::Vector, nodes::Vector{T}) where {T<:AbstractNode}
    new_dict = Dict()
    mapping = get_states_mapping_dict(nodes)
    for key in vector_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            push!(new_key, mapping[nodes[collect(keys(nodes))[i]]][key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = undef
    end
    return new_dict
end

function map_integer_to_state(dict_to_be_mapped::Dict, nodes::Vector{T}) where {T<:AbstractNode}
    new_dict = Dict()
    mapping = get_states_mapping_dict(nodes)
    for (key, val) in dict_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            rmapping = Dict(values(mapping[nodes[collect(keys(nodes))[i]]]) .=> keys(mapping[nodes[collect(keys(nodes))[i]]]))
            push!(new_key, rmapping[key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = val
    end
    return new_dict
end

function map_integer_to_state(vector_to_be_mapped::Vector, nodes::Vector{T}) where {T<:AbstractNode}
    new_dict = Dict()
    mapping = get_states_mapping_dict(nodes)
    for key in vector_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            rmapping = Dict(values(mapping[nodes[collect(keys(nodes))[i]]]) .=> keys(mapping[nodes[collect(keys(nodes))[i]]]))
            push!(new_key, rmapping[key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = undef
    end
    return new_dict
end

function get_discreteparents_states_combinations(node::T) where {T<:AbstractNode}
    discrete_parents = get_discrete_parents(node)
    return get_states_combination(discrete_parents)
end

function get_discreteparents_states_mapping_dict(node::T) where {T<:AbstractNode}
    discrete_parents = get_discrete_parents(node)
    return get_states_mapping_dict(discrete_parents)
end

## To be used when bn is not already defined
function evaluate_nodecpd_with_evidence(node::T, evidence::Assignment) where {T<:AbstractNode}
    convertedevidence = Assignment()
    for (key, val) in evidence
        if ~isa(val, Number)
            convertedevidence[key] = get_states_mapping_dict([key])[name(key)][val]
        else
            convertedevidence[key] = val
        end
    end

    cpd_dict = node.cpd.prob_dict
    parents_nodes = node.parents
    evidenced_nodes = collect(keys(convertedevidence))
    assignment_index = (parents_nodes) .∈ [evidenced_nodes]
    undefined_parents = (parents_nodes)[(parents_nodes).∉[evidenced_nodes]]
    useless_assignmets = evidenced_nodes[evidenced_nodes.∉[(parents_nodes)]]
    if ~isempty(useless_assignmets)
        println("Evidences on $useless_assignmets should be treated with join pdf")
    end
    if ~isempty(undefined_parents)
        println("Following cpds are defined for each value of $undefined_parents")
    end
    vec_keys = Vector()
    for i in range(1, length(assignment_index))
        if assignment_index[i]
            push!(vec_keys, [convertedevidence[(parents_nodes[i])]])
        else
            if isa(parents_nodes[i].cpd, RootCPD)
                push!(vec_keys, [1:length(parents_nodes[i].cpd.distributions);])
            else
                push!(vec_keys, [1:length(parents_nodes[i].cpd.distributions);])
            end
        end
    end
    new_states = collect(filter(key -> all([key[i] in vec_keys[i] for i in 1:length(assignment_index)]), keys(cpd_dict)))
    new_cond_dic = Dict{Tuple,Union{CPD,Distribution}}()
    for k in new_states
        new_cond_dic[k] = cpd_dict[k]
    end
    return new_cond_dic
end