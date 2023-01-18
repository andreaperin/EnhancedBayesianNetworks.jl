using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end
abstract type Node <: AbstractNode end
abstract type ModelInput end

# struct ModelNode <: Node
#     cpd::CPD
#     parents::Vector{T} where {T<:AbstractNode}
#     type::String
#     ##TODO add to log
#     function ModelNode(cpd::CPD, parents::Vector{T}, type::String) where {T<:AbstractNode}
#         node_name = cpd.target
#         cpd.parents == name.(parents) ? new(cpd, parents, type) : error("parents mismatch between CPD and node in $node_name")



#     end

#     function ModelNode(cpd::CPD, parents::Vector{T}) where {T<:AbstractNode}

#     end

# end

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


#####

function name(node::T) where {T<:AbstractNode}
    return node.cpd.target
end

# ``` 1) discrete_states returns:
#             states for discrete nodes 
#                 Vector{Dict{state_description_Symbol, state_number}}
#     2) continuous_distributions returns:
#             possible distributions for continuous nodes 
#                 Vector{Dict{distribution_description, distribution_number}}
# ```

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


# function states(node::AbstractNode)
#     if node.cpd isa RootCPD
#         return node.type == "discrete" ? node.cpd.distributions.map.n2d : "this is a continuous node"
#     else
#         return node.type == "discrete" ? node.cpd.distributions[1].map.n2d : "this is a continuous node"
#     end
# end

function get_discrete_parents(node::T) where {T<:AbstractNode}
    discrete_parents = copy(node.parents)
    return filter(x -> x.type == "discrete", discrete_parents)
end

function get_continuous_parents(node::T) where {T<:AbstractNode}
    continuous_parents = copy(node.parents)
    return filter(x -> x.type == "continuous", continuous_parents)
end

function get_states_combination(nodes::Vector{T}) where {T<:AbstractNode}
    continuous = name.(filter(x -> x.type == "continuous", nodes))
    if ~isempty(continuous)
        filter(x -> x.type == "continuous", nodes)
        println("$continuous are continuous nodes => Not taken into account")
    end
    discrete_nodes = filter(x -> x.type == "discrete", nodes)
    nodes_states = Vector{Dict{Symbol,Vector}}()
    to_combine = []
    nodes_combinations = Vector{Tuple{Symbol}}()
    for node in discrete_nodes
        if node.cpd isa RootCPD
            push!(nodes_states, Dict(name(node) => collect(values(node.cpd.distributions.map.d2n))))
            push!(to_combine, collect(values(node.cpd.distributions.map.d2n)))
        else
            push!(nodes_states, Dict(name(node) => collect(values(node.cpd.distributions[1].map.d2n))))
            push!(to_combine, collect(values(node.cpd.distributions[1].map.d2n)))
        end
    end
    nodes_combinations = collect(Iterators.product(to_combine...))
    return nodes_states, nodes_combinations
end

function get_states_mapping_dict(nodes::Vector{T}) where {T<:AbstractNode}
    mapping = Dict{AbstractNode,Dict{}}()
    for node in nodes
        if node.cpd isa RootCPD
            mapping[node] = node.cpd.distributions.map.n2d
        else
            mapping[node] = node.cpd.distributions[1].map.n2d
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


# function get_common_parents(nodes::Vector{T}) where {T<:AbstractNode}
#     all_parents_dict = Dict{Any,Vector{Any}}()
#     all_parents_vector = Vector{Any}()
#     for node in nodes
#         all_parents_dict[name(node)] = [name(i) for i in node.parents]
#         for grandparent in node.parents
#             push!(all_parents_vector, name(grandparent))
#         end
#     end
#     unique!(all_parents_vector)
#     final_dict = Dict{Any,Vector{Any}}()
#     for el in all_parents_vector
#         final_vect_i = Vector{Any}()
#         for (k, v) in all_parents_dict
#             if el in v
#                 push!(final_vect_i, k)
#             end
#         end
#         if length(final_vect_i) > 1
#             final_dict[el] = final_vect_i
#         end
#     end
#     return final_dict
# end

# function get_cpd_dict(node::Node)
#     if length(node.parents) != 0
#         cpds_dict = Dict{Tuple,Union{CPD,Distribution}}()
#         combinations = vec(get_discreteparents_states_combinations(node)[2])
#         ordered_dict = sort(map_state_to_integer(combinations, node))
#         for i in range(1, length(node.cpd.distributions))
#             cpds_dict[collect(keys(ordered_dict))[i]] = node.cpd.distributions[i]
#         end
#         return sort(cpds_dict)
#     else
#         return Dict(tuple(undef) => node.cpd.distributions)
#     end
# end