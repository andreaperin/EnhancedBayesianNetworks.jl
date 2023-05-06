struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{M}} where {M<:UQModel}

    function DiscreteFunctionalNode(name::Symbol, parents::Vector{<:AbstractNode}, models::OrderedDict{Vector{Symbol},Vector{M}}) where {M<:UQModel}
        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        verify_functionalnode_parents(parents)

        for (key, _) in models
            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")

            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(models) && error("defined combinations must be equal to the discrete parents combinations")

        return new(name, parents, models)
    end
end