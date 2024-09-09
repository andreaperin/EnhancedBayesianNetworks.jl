function discrete_ancestors(node::AbstractNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    if isempty(continuous_parents)
        return discrete_parents
    end
    return unique([discrete_parents..., mapreduce(discrete_ancestors, vcat, continuous_parents)...])
end

function discrete_ancestors(_::RootNode)
    return AbstractNode[]
end

function state_combinations(node::AbstractNode)
    par = discrete_ancestors(node)
    discrete_parents = filter(x -> isa(x, DiscreteNode), par)
    discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
    discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
    return vec(discrete_parents_combination)
end

function state_combinations(_::RootNode)
    return AbstractNode[]
end

function _extreme_points_states_probabilities(states::Dict{Symbol,Union{Real,AbstractVector{Real}}})
    n = length(states)
    A = zeros(2 * n, n)
    A[collect(1:2:2*n), :] = Matrix(-1.0I, n, n)
    A[collect(2:2:2*n), :] = Matrix(1.0I, n, n)
    A = vcat(A, [-ones(n)'; ones(n)'])

    b = collect(Iterators.flatten(collect(values(states))))
    b[collect(1:2:2*n)] = -b[collect(1:2:2*n)]
    b = vcat(b, [-1 1]')

    h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
    v = doubledescription(h)
    return map(val -> Dict(keys(states) .=> val), v.points.points)
end