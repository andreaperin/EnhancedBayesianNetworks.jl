
function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousChildNode)

    intervals = deepcopy(node.discretization.intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]

    lower_bound = minimum(support(i).lb for i in values(node.distributions))
    upper_bound = maximum(support(i).ub for i in values(node.distributions))


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

