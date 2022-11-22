using UncertaintyQuantification
using JLD2
using Formatting
using Graphs
using ProgressMeter

include("CPDs.jl")

abstract type AbstractNode end
abstract type Node <: AbstractNode end
mutable struct StdNode <: Node
    cpd::CPD
    parents::Vector{T} where {T<:AbstractNode}
    type::String
    model_input::Dict{Symbol,Dict{String,Vector}}
    function StdNode(cpd::CPD)
        parents = Vector{AbstractNode}()
        type = isa(cpd, CPD{Distribution}) ? "continuous" : "discrete"
        model_input = Dict{Symbol,Dict{String,Vector}}()
        new(cpd, parents, type, model_input)
    end
    function StdNode(cpd::CPD, parents::Vector{T}) where {T<:AbstractNode}
        type = isa(cpd, CPD{Distribution}) ? "continuous" : "discrete"
        model_input = Dict{Symbol,Dict{String,Vector}}()
        new(cpd, parents, type, model_input)
    end
    function StdNode(cpd::CPD, parents::Vector{T}, model_input::Dict{Symbol,Dict{String,Vector}}) where {T<:AbstractNode}
        ## TODO Add check for dictionary coherence
        type = isa(cpd, CPD{Distribution}) ? "continuous" : "discrete"
        new(cpd, parents, type, model_input)
    end
end

struct NewModelNode <: AbstractNode
    parents::Vector{T} where {T<:AbstractNode}
    models::Dict{Any,<:UQModel}
    uqinputs::Dict{Any,Vector{<:UQInput}}
    performances::Dict{Symbol,Function}
    function NewModelNode(parents::Vector{<:AbstractNode}, models::Dict{Any,<:UQModel}, uqinputs::Dict{Any,Vector{<:UQInput}}, performances::Dict{Symbol,Function})
        new(parents, models, uqinputs, performances)
    end
    function NewModelNode(parents::Vector{<:AbstractNode}, models::Dict{Any,<:UQModel}, uqinputs::Dict{Any,Vector{<:UQInput}})
        performances = Dict{Symbol,Function}()
        new(parents, models, uqinputs, performances)
    end
end

# struct ModelNode <: AbstractNode
#     name::Symbol
#     parents::Vector{T} where {T<:AbstractNode}
#     default_inputs::Dict{String,Vector}
#     sourcedir::String
#     source_file::String
#     extras::Vector{String}
#     solvername::String
#     output_parameters::Dict
#     performances::Dict{Symbol,Function}
#     cleanup::Bool
#     inputs_states_mapping_dict::Dict{Any,Vector}
#     updated_inputs::Dict{Any,Vector{<:UQInput}}
#     sim::AbstractMonteCarlo
#     function ModelNode(name::Symbol,
#         parents::Vector{T} where {T<:AbstractNode},
#         default_inputs::Dict{String,Vector},
#         sourcedir::String,
#         source_file::String,
#         extras::Vector{String},
#         solvername::String,
#         output_parameters::Dict,
#         performances::Dict{Symbol,Function},
#         cleanup::Bool,
#         sim::AbstractMonteCarlo)

#         inputs_states_mapping_dict = Dict{Any,Vector}()
#         updated_inputs = Dict{Any,Vector{<:UQInput}}()
#         new(name, parents, default_inputs, sourcedir, source_file, extras, solvername, output_parameters, performances, cleanup, inputs_states_mapping_dict, updated_inputs, sim)
#     end
#     function ModelNode(name::Symbol,
#         parents::Vector{T} where {T<:AbstractNode},
#         default_inputs::Dict{String,Vector},
#         sourcedir::String,
#         source_file::String,
#         extras::Vector{String},
#         solvername::String,
#         output_parameters::Dict,
#         performances::Dict{Symbol,Function},
#         cleanup::Bool,
#         inputs_states_mapping_dict::Dict{Any,Vector},
#         updated_inputs::Dict{Any,Vector{<:UQInput}},
#         sim::AbstractMonteCarlo)

#         new(name, parents, default_inputs, sourcedir, source_file, extras, solvername, output_parameters, performances, cleanup, inputs_states_mapping_dict, updated_inputs, sim)
#     end
# end

# function evaluate_cpd_from_model(node::ModelNode, model_inputs_mapping_dict::Dict, performances::Dict{Symbol,Function}, uqinputs::Dict{Any,Vector{<:UQInput}})
#     states_comb = get_discreteparents_states_combinations(node)[2]
#     if length(collect(keys(model_inputs_mapping_dict))[collect(keys(model_inputs_mapping_dict)).∉Ref(vec(states_comb))]) != 0
#         @show("parents states mismatch")
#     else
#         cond_probs_dict = Dict()
#         cpd = Dict{Tuple,NamedCategorical}()
#         if length(get_continuous_parents(node)) == 0
#             sim = MonteCarlo(1)
#             @showprogress 1 "Evaluating Model..." for combination in collect(keys(model_inputs_mapping_dict))
#                 th_single_state_model = ExternalModel(th_node.inputs_states_mapping_dict[combination]...)
#                 probs, variances, covs, samples = probabilities_of_events(th_single_state_model, performances, uqinputs[combination], sim)
#                 cond_probs_dict[combination] = Dict(
#                     "prob" => probs,
#                     "cov" => covs
#                 )
#                 cpd[combination] = NamedCategorical(Vector{Symbol}(collect(keys(probs))), Vector{Float64}(collect(values(probs))))
#             end
#         end
#     end
#     cpd_ordered = sort(map_state_to_integer(cpd, node))
#     new_ordered_parents = get_new_ordered_parents(node)
#     CPD = CategoricalCPD(node.name, name.(new_ordered_parents), [length(states(i)) for i in new_ordered_parents], Vector{NamedCategorical}(collect(values(cpd_ordered))))
#     node = StdNode(CPD, new_ordered_parents)
#     return cond_probs_dict, cpd, CPD, node
# end

function name(node::T) where {T<:AbstractNode}
    return node.cpd.target
end

function states(node::AbstractNode)
    if node.cpd isa RootCPD
        return node.type == "discrete" ? node.cpd.distributions.map.n2d : "this is a continuous node"
    else
        return node.type == "discrete" ? node.cpd.distributions[1].map.n2d : "this is a continuous node"
    end
end

function get_discrete_parents(node::T) where {T<:AbstractNode}
    discrete_parents_dict = copy(node.parents)
    filter(x -> x.type == "discrete", discrete_parents_dict)
    return discrete_parents_dict
end

function get_continuous_parents(node::T) where {T<:AbstractNode}
    continuous_parents = copy(node.parent)
    filter(x -> x.type == "continuous", discrete_parents_dict)
    return continuous_parents
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