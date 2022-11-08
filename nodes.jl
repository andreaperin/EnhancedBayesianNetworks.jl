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

struct ModelNode <: AbstractNode
    name::Symbol
    parents::Vector{T} where {T<:AbstractNode}
    default_inputs::Dict{String,Vector}
    sourcedir::String
    source_file::String
    extras::Vector{String}
    solvername::String
    output_parameters::Dict
    performances::Dict{Symbol,Function}
    cleanup::Bool
    inputs_states_mapping_dict::Dict{Any,Vector}
    updated_inputs::Dict{Any,Vector{<:UQInput}}
    sim::AbstractMonteCarlo
    function ModelNode(name::Symbol,
        parents::Vector{T} where {T<:AbstractNode},
        default_inputs::Dict{String,Vector},
        sourcedir::String,
        source_file::String,
        extras::Vector{String},
        solvername::String,
        output_parameters::Dict,
        performances::Dict{Symbol,Function},
        cleanup::Bool,
        sim::AbstractMonteCarlo)
        inputs_states_mapping_dict = Dict{Any,Vector}()
        updated_inputs = Dict{Any,Vector{<:UQInput}}()
        new(name, parents, default_inputs, sourcedir, source_file, extras, solvername, output_parameters, performances, cleanup, inputs_states_mapping_dict, updated_inputs, sim)
    end
    function ModelNode(name::Symbol,
        parents::Vector{T} where {T<:AbstractNode},
        default_inputs::Dict{String,Vector},
        sourcedir::String,
        source_file::String,
        extras::Vector{String},
        solvername::String,
        output_parameters::Dict,
        performances::Dict{Symbol,Function},
        cleanup::Bool,
        inputs_states_mapping_dict::Dict{Any,Vector},
        updated_inputs::Dict{Any,Vector{<:UQInput}},
        sim::AbstractMonteCarlo)
        new(name, parents, default_inputs, sourcedir, source_file, extras, solvername, output_parameters, performances, cleanup, inputs_states_mapping_dict, updated_inputs, sim)
    end
end

function evaluate_cpd_from_model(node::ModelNode, model_inputs_mapping_dict::Dict, performances::Dict{Symbol,Function}, uqinputs::Dict{Any,Vector{<:UQInput}})
    states_comb = get_discreteparents_states_combinations(node)[2]
    if length(collect(keys(model_inputs_mapping_dict))[collect(keys(model_inputs_mapping_dict)).∉Ref(vec(states_comb))]) != 0
        @show("parents states mismatch")
    else
        cond_probs_dict = Dict()
        cpd = Dict{Tuple,NamedCategorical}()
        if length(get_continuous_parents(node)) == 0
            sim = MonteCarlo(1)
            @showprogress 1 "Evaluating Model..." for combination in collect(keys(model_inputs_mapping_dict))
                th_single_state_model = ExternalModel(th_node.inputs_states_mapping_dict[combination]...)
                probs, variances, covs, samples = probabilities_of_events(th_single_state_model, performances, uqinputs[combination], sim)
                cond_probs_dict[combination] = Dict(
                    "prob" => probs,
                    "cov" => covs
                )
                cpd[combination] = NamedCategorical(Vector{Symbol}(collect(keys(probs))), Vector{Float64}(collect(values(probs))))
            end
        end
    end
    cpd_ordered = sort(map_state_to_integer(cpd, node))
    new_ordered_parents = get_new_ordered_parents(node)
    CPD = CategoricalCPD(node.name, name.(new_ordered_parents), [length(states(i)) for i in new_ordered_parents], Vector{NamedCategorical}(collect(values(cpd_ordered))))
    node = StdNode(CPD, new_ordered_parents)
    return cond_probs_dict, cpd, CPD, node
end

function name(node::T) where {T<:AbstractNode}
    return node.cpd.target
end

function states(node::AbstractNode)
    if node.cpd isa StaticCPD
        return node.type == "discrete" ? node.cpd.d.map.n2d : "this is a continuous node"
    else
        return node.type == "discrete" ? node.cpd.distributions[1].map.n2d : "this is a continuous node"
    end
end

function get_discrete_parents(node::T) where {T<:AbstractNode}
    # discrete_parents = Vector{Node}()
    discrete_parents_dict = Dict{Symbol,Node}()
    for parent in node.parents
        if parent.type == "discrete"
            # push!(discrete_parents, parent)
            discrete_parents_dict[parent.cpd.target] = parent
        end
    end
    return discrete_parents_dict
end

function get_continuous_parents(node::T) where {T<:AbstractNode}
    continuous_parents = Vector{Node}()
    for parent in node.parents
        if parent.type == "continuous"
            push!(continuous_parents, parent)
        end
    end
    return continuous_parents
end

function get_discreteparents_states_combinations(node::T) where {T<:AbstractNode}
    discrete_parents = get_discrete_parents(node)
    all_discreteparents_states = Dict()
    combinations = Vector{Tuple{Symbol}}()
    for (parent_name, parent) in discrete_parents
        if parent.cpd isa StaticCPD
            all_discreteparents_states[parent_name] = collect(values(parent.cpd.d.map.d2n))
        else
            all_discreteparents_states[parent_name] = collect(values(parent.cpd.distributions[1].map.d2n))
        end
    end
    combinations = collect(Iterators.product(collect(values(all_discreteparents_states))...))
    return all_discreteparents_states, combinations
end

function get_new_ordered_parents(node::T) where {T<:AbstractNode}
    new_ordered_parents = [get_discrete_parents(node)[i] for i in collect(keys(get_discreteparents_states_combinations(node)[1]))]
    return new_ordered_parents
end

function get_discreteparents_states_mapping_dict(node::T) where {T<:AbstractNode}
    parents = get_discrete_parents(node)
    mapping = Dict{AbstractNode,Dict{}}()
    for parent_node in collect(values(parents))
        if parent_node.cpd isa StaticCPD
            mapping[parent_node] = parent_node.cpd.d.map.n2d
        else
            mapping[parent_node] = parent_node.cpd.distributions[1].map.n2d
        end
    end
    return mapping
end

function map_state_to_integer(dict_to_be_mapped::Dict, node::T) where {T<:AbstractNode}
    new_dict = Dict()
    discrete_parents = get_discrete_parents(node)
    mapping = get_discreteparents_states_mapping_dict(node)
    new_ordered_parents = get_discreteparents_states_combinations(th_node)[1]
    for (key, val) in dict_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            push!(new_key, mapping[discrete_parents[collect(keys(new_ordered_parents))[i]]][key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = val
    end
    return new_dict
end

function map_state_to_integer(vector_to_be_mapped::Vector, node::T) where {T<:AbstractNode}
    new_dict = Dict()
    discrete_parents = get_discrete_parents(node)
    mapping = get_discreteparents_states_mapping_dict(node)
    new_ordered_parents = get_discreteparents_states_combinations(node)[1]
    for key in vector_to_be_mapped
        new_key = []
        for i in range(1, length(key))
            push!(new_key, mapping[discrete_parents[collect(keys(new_ordered_parents))[i]]][key[i]])
        end
        new_key = tuple(new_key...)
        new_dict[new_key] = undef
    end
    return new_dict
end

function get_common_parents(nodes::Vector{T}) where {T<:AbstractNode}
    all_parents_dict = Dict{Any,Vector{Any}}()
    all_parents_vector = Vector{Any}()
    for node in nodes
        all_parents_dict[name(node)] = [name(i) for i in node.parents]
        for grandparent in node.parents
            push!(all_parents_vector, name(grandparent))
        end
    end
    unique!(all_parents_vector)
    final_dict = Dict{Any,Vector{Any}}()
    for el in all_parents_vector
        final_vect_i = Vector{Any}()
        for (k, v) in all_parents_dict
            if el in v
                push!(final_vect_i, k)
            end
        end
        if length(final_vect_i) > 1
            final_dict[el] = final_vect_i
        end
    end
    return final_dict
end

function get_cpd_dict(node::Node)
    if length(node.parents) != 0
        cpds_dict = Dict{Tuple,Union{CPD,Distribution}}()
        combinations = vec(get_discreteparents_states_combinations(node)[2])
        ordered_dict = sort(map_state_to_integer(combinations, node))
        for i in range(1, length(node.cpd.distributions))
            cpds_dict[collect(keys(ordered_dict))[i]] = node.cpd.distributions[i]
        end
        return sort(cpds_dict)
    else
        return Dict(tuple(undef) => node.cpd.d)
    end
end