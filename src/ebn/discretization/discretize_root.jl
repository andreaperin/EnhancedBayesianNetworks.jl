
function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousRootNode)

    intervals = deepcopy(node.discretization.intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]

    lower_bound = support(node.distribution).lb
    upper_bound = support(node.distribution).ub

    if minimum(min) != lower_bound
        @warn "selected minimum intervals value $min ≥ support lower buond $lower_bound. Support lower bound will be used as intervals starting value!"
        insert!(intervals, 1, lower_bound)
    end

    if maximum(max) != upper_bound
        @warn "selected maximum intervals value $max ≤ support's upper buond $upper_bound. Support's upper bound will be used as intervals final value!"
        push!(intervals, upper_bound)
    end

    nodes = deepcopy(ebn.nodes)
    intervals = [[intervals[i], intervals[i+1]] for i in range(1, length(intervals) - 1)]

    f_d = i -> cdf(node.distribution, i[2]) - cdf(node.distribution, i[1])

    states_symbols = [Symbol(i) for i in intervals]
    states = Dict(states_symbols .=> f_d.(intervals))

    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)

    ## Adding continuous node as parents of children of the discretized node
    f_c = i -> truncated(node.distribution, i[1], i[2])
    distributions_symbols = [[i] for i in states_symbols]
    distributions = Dict(distributions_symbols .=> f_c.(intervals))

    continuous_node = ContinuousChildNode(Symbol(string(node.name)), [discrete_node], distributions)

    children = get_children(ebn, node)
    children = filter(x -> x.name in [j.name for j in children], nodes)
    for child in children
        deleteat!(child.parents, findall(isequal.(repeat([node], length(child.parents)), child.parents)))
        push!(child.parents, continuous_node)
    end
    append!(nodes, [continuous_node, discrete_node])
    deleteat!(nodes, findall(isequal.(repeat([node], length(nodes)), nodes)))
    return nodes
end
