using EnhancedBayesianNetworks
using Plots

v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
t = DiscreteChildNode(:T, [v], Dict(
    [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
    [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
)

l = DiscreteChildNode(:L, [s], Dict(
    [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
    [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
)

b = DiscreteChildNode(:B, [s], Dict(
    [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
    [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
)

e = DiscreteChildNode(:E, [t, l], Dict(
    [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
)

d = DiscreteChildNode(:D, [b, e], Dict(
    [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
    [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
    [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
    [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
)

x = DiscreteChildNode(:X, [e], Dict(
    [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
    [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
)

nodes = [v, s, t, l, b, e, d, x]
bn = BayesianNetwork(nodes)

function _find_order(nodes::AbstractVector{<:AbstractNode})
    ordered_list = []
    root_nodes = filter(x -> isa(x, RootNode), nodes)
    push!(ordered_list, root_nodes)
    ref_nodes = root_nodes
    new_nodes = setdiff(nodes, ref_nodes)
    while !isempty(new_nodes)
        ref_parents = new_nodes[map(x -> all(x.parents .∈ [ref_nodes]), new_nodes)]
        push!(ordered_list, ref_parents)
        ref_nodes = [ref_nodes..., ref_parents...]
        new_nodes = setdiff(new_nodes, ref_parents)
    end
    return ordered_list
end

function _adjacency_matrix(ordered_list::AbstractVector{<:AbstractNode})
    n = length(ordered_list)
    adj = falses(n, n)
    for j in axes(adj, 1)
        for i in axes(adj, 2)
            if i != j && !isa(a[j], RootNode)
                adj[i, j] = a[i] ∈ a[j].parents
            end
        end
    end
    return adj
end

EnhancedBayesianNetworks.plot(bn, :tree, 0.1, 13)

p = _find_order(nodes)

ordered_list = collect(Iterators.flatten(p))

adj = _adjacency_matrix(ordered_list)