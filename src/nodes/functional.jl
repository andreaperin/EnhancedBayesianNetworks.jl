struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{M}} where {M<:UQModel}

    function DiscreteFunctionalNode(name::Symbol, parents::Vector{<:AbstractNode}, models::OrderedDict{Vector{Symbol},Vector{M}}) where {M<:UQModel}
        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, _) in models
            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(models) && error("defined combinations must be equal to the discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(models)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, models)
    end
end
