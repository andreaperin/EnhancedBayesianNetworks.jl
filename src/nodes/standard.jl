struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},D} where {D<:Distribution}

    function ContinuousStandardNode(name::Symbol, parents::Vector{N}, distribution::OrderedDict{Vector{Symbol},D}) where {N<:AbstractNode,D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in distribution
            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations must be equal to the discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(distribution)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, distribution)
    end
end

function _get_states(node::ContinuousStandardNode)
    list = collect(values(node.distribution))
    return list
end

struct DiscreteStandardNode <: DiscreteNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    states::OrderedDict{Vector{Symbol},Dict{Symbol,T}} where {T<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteStandardNode(name::Symbol, parents::Vector{N}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}, parameters::Dict{Symbol,Vector{Parameter}}) where {N<:AbstractNode,T<:Real}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in states
            any(values(val) .< 0.0) && error("Probabilites must be nonnegative")
            any(values(val) .> 1.0) && error("Probabilites must be less or equal to 1.0")
            sum(values(val)) > 1.0 && error("Probabilites must sum up to 1.0")

            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("defined combinations must be equal to the discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(states)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, states, parameters)
    end
end

function DiscreteStandardNode(name::Symbol, parents::Vector{N}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}) where {N<:AbstractNode,T<:Real}
    DiscreteStandardNode(name, parents, states, Dict{Symbol,Vector{Parameter}}())
end

function _get_states(node::DiscreteStandardNode)
    list = []
    for (key, val) in node.states
        push!(list, collect(keys(val)))
    end
    unique!(list)
    length(list) != 1 && error("non coherent definition of nodes states in the ordered dict")
    return list[1]
end

const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}

# function _get_discrete_parents_combinations(node::AbstractNode)

# struct StdNode <: AbstractNode
#     cpd::StdCPD
#     parents::Vector{<:AbstractNode}
#     type::String
#     evidence_table::Vector{EvidenceTable}
#     model_parameters::Vector{ModelParametersTable}

#     function StdNode(
#         cpd::StdCPD,
#         parents::Vector{<:AbstractNode},
#         type::String,
#         evidence_table::Vector{EvidenceTable},
#         model_parameters::Vector{ModelParametersTable}
#     )
#         # if ~isempty(filter(x -> x.type == "continuous", parents))
#         #     throw(DomainError(cpd.target, "StdCPD is for discrete parents only"))
#         # end
#         discrete_parents = filter(x -> x.type == "discrete", parents)
#         if length(discrete_parents) != length(cpd.parental_ncategories)
#             throw(DomainError(cpd.target, "parents-parental_ncategories length missmatch"))
#         end
#         if _get_number_of_discretestates.(discrete_parents) != cpd.parental_ncategories
#             throw(DomainError(cpd.target, "parental_ncategories - parents discrete states missmatch"))
#         end
#         new(cpd, parents, type, evidence_table, model_parameters)
#     end
# end

# function StdNode(cpd::StdCPD, parents::Vector{<:AbstractNode})
#     type = _get_type_of_cpd(cpd)
#     evidence_table = _build_evidencetable_from_cpd(cpd, parents)
#     model_parameters = [ModelParametersTable(Assignment(), [ModelParameters()])]
#     StdNode(cpd, parents, type, evidence_table, model_parameters)
# end

# function StdNode(cpd::StdCPD, parents::Vector{<:AbstractNode}, parameters_vector::Vector{Vector{ModelParameters}})
#     type = _get_type_of_cpd(cpd)
#     evidence_table = _build_evidencetable_from_cpd(cpd, parents)
#     model_parameters = _build_modelparametertable(cpd, parameters_vector)
#     StdNode(cpd, parents, type, evidence_table, model_parameters)
# end