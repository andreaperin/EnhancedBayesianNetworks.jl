
function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousChildNode)
    intervals = _format_interval(node)
    variance = node.discretization.sigma

    f_d = (d, i) -> cdf(d, i[2]) - cdf(d, i[1])
    states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    for (key, dist) in node.distributions
        states[key] = Dict(Symbol.(intervals) .=> f_d.(dist, intervals))
    end
    discrete_node = DiscreteChildNode(Symbol(string(node.name) * "_d"), node.parents, states)

    ## Approximation function is a truncated normal (thicker tails)
    f_c = i -> begin
        a = isfinite.(i)
        all(a) ? Uniform(i...) : truncated(Normal(i[a][1], variance), i...)
    end
    distributions = Dict([Symbol(i)] => f_c(i) for i in intervals)
    continuous_node = ContinuousChildNode(Symbol(string(node.name)), [discrete_node], distributions)

    return _update_nodes_after_discretization(ebn, node, discrete_node, continuous_node)
end

function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousRootNode)
    intervals = _format_interval(node)

    f_d = i -> cdf(node.distribution, i[2]) - cdf(node.distribution, i[1])

    states_symbols = [Symbol(i) for i in intervals]
    states = Dict(states_symbols .=> f_d.(intervals))

    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)

    ## Adding continuous node as parents of children of the discretized node
    f_c = i -> truncated(node.distribution, i[1], i[2])
    distributions_symbols = [[i] for i in states_symbols]
    distributions = Dict(distributions_symbols .=> f_c.(intervals))

    continuous_node = ContinuousChildNode(Symbol(string(node.name)), [discrete_node], distributions)
    return _update_nodes_after_discretization(ebn, node, discrete_node, continuous_node)
end

function _get_node_distribution_bounds(node::ContinuousChildNode)
    lower_bound = minimum(support(i).lb for i in values(node.distributions))
    upper_bound = maximum(support(i).ub for i in values(node.distributions))
    return lower_bound, upper_bound
end

function _get_node_distribution_bounds(node::ContinuousRootNode)
    lower_bound = support(node.distribution).lb
    upper_bound = support(node.distribution).ub
    return lower_bound, upper_bound
end

function _format_interval(node::Union{ContinuousChildNode,ContinuousRootNode})
    intervals = deepcopy(node.discretization.intervals)

    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]

    lower_bound, upper_bound = _get_node_distribution_bounds(node)

    if minimum(min) != lower_bound
        @warn "selected minimum intervals value $min ≥ support lower buond $lower_bound. Support lower bound will be used as intervals starting value!"
        insert!(intervals, 1, lower_bound)
    end

    if maximum(max) != upper_bound
        @warn "selected maximum intervals value $max ≤ support's upper buond $upper_bound. Support's upper bound will be used as intervals final value!"
        push!(intervals, upper_bound)
    end

    return [[intervals[i], intervals[i+1]] for (i, _) in enumerate(intervals[1:end-1])]
end

function _update_nodes_after_discretization(ebn, discretizable_node, discretized_node, continuous_auxiliary_node)
    nodes = deepcopy(ebn.nodes)
    children = get_children(ebn, discretizable_node)
    children = filter(x -> x.name in [j.name for j in children], nodes)
    for child in children
        deleteat!(child.parents, findall(isequal.(repeat([discretizable_node], length(child.parents)), child.parents)))
        push!(child.parents, continuous_auxiliary_node)
    end
    append!(nodes, [continuous_auxiliary_node, discretized_node])
    deleteat!(nodes, findall(isequal.(repeat([discretizable_node], length(nodes)), nodes)))
    return nodes
end