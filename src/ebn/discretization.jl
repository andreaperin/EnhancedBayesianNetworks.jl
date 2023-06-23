
function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousRootNode, intervals::Vector{Vector{Float64}})
    ## Check intervals
    verify_intervals(intervals)
    lower_buond = support(node.distribution).lb
    upper_bound = support(node.distribution).ub
    minimum(minimum.(intervals)) != lower_buond && push!(intervals, [lower_buond, minimum(minimum.(intervals))])
    maximum(maximum.(intervals)) != upper_bound && push!(intervals, [maximum(maximum.(intervals)), upper_bound])

    nodes = deepcopy(ebn.nodes)
    f_d = i -> cdf(node.distribution, i[2]) - cdf(node.distribution, i[1])
    states_symbols = [Symbol(i) for i in intervals]
    states = Dict(states_symbols .=> f_d.(intervals))

    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)

    ## Adding continuous node as parents of children of the discretized node
    f_c = i -> truncated(node.distribution, i[1], i[2])
    distributions_symbols = [[i] for i in states_symbols]
    distributions = Dict(distributions_symbols .=> f_c.(intervals))

    continuous_node = ContinuousStandardNode(Symbol(string(node.name)), [discrete_node], distributions)

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

function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousStandardNode, intervals::Vector{Vector{Float64}}, variance::Real)
    ## Check intervals
    verify_intervals(intervals)
    lower_buond = minimum(support(i).lb for i in values(node.distributions))
    upper_buond = maximum(support(i).ub for i in values(node.distributions))
    minimum(minimum.(intervals)) != lower_buond && push!(intervals, [lower_buond, minimum(minimum.(intervals))])
    maximum(maximum.(intervals)) != upper_buond && push!(intervals, [maximum(maximum.(intervals)), upper_buond])

    nodes = deepcopy(ebn.nodes)
    f_d = (d, i) -> cdf(d, i[2]) - cdf(d, i[1])

    states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    for (key, dist) in node.distributions
        states[key] = Dict(Symbol.(intervals) .=> f_d.(dist, intervals))
    end

    discrete_node = DiscreteStandardNode(Symbol(string(node.name) * "_d"), node.parents, states)

    ## Approximation function is a truncated normal (thicker tails)
    f_c = i -> begin
        a = isfinite.(i)
        all(a) ? Uniform(i...) : truncated(Normal(i[a][1], variance), i...)
    end

    distributions = Dict([Symbol(i)] => f_c(i) for i in intervals)

    continuous_node = ContinuousStandardNode(Symbol(string(node.name)), [discrete_node], distributions)

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

