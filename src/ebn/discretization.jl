
function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousRootNode, intervals::Vector{Vector{Float64}})
    ## Check intervals
    ebn_new = copy(ebn)
    verify_intervals(intervals)
    lower_buond = support(node.distribution).lb
    upper_bound = support(node.distribution).ub
    minimum(minimum.(intervals)) != lower_buond && push!(intervals, [lower_buond, minimum(minimum.(intervals))])
    maximum(maximum.(intervals)) != upper_bound && push!(intervals, [maximum(maximum.(intervals)), upper_bound])

    f_d = i -> cdf(node.distribution, i[2]) - cdf(node.distribution, i[1])
    states_symbols = [Symbol(i) for i in intervals]
    states = Dict(states_symbols .=> f_d.(intervals))

    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)

    ## Adding continuous node as parents of children of the discretized node
    f_c = i -> Truncated(node.distribution, i[1], i[2])
    distributions_symbols = [[i] for i in states_symbols]
    distributions = OrderedDict(distributions_symbols .=> f_c.(intervals))

    continuous_node = ContinuousStandardNode(Symbol(string(node.name) * "_c"), [discrete_node], distributions)

    for child in get_children(ebn_new, node)
        deleteat!(child.parents, findall(x -> x == node, child.parents))
        push!(child.parents, continuous_node)
    end
    nodes = append!(ebn_new.nodes, [continuous_node, discrete_node])
    deleteat!(nodes, findall(x -> x == node, nodes))

    EnhancedBayesianNetwork(nodes)
end

function _discretize_node(ebn::EnhancedBayesianNetwork, node::ContinuousStandardNode, intervals::Vector{Vector{Float64}}, variance::Real)
    ## Check intervals
    ebn_new = copy(ebn)
    verify_intervals(intervals)
    lower_buond = minimum(support(i).lb for i in values(node.distribution))
    upper_buond = maximum(support(i).ub for i in values(node.distribution))
    minimum(minimum.(intervals)) != lower_buond && push!(intervals, [lower_buond, minimum(minimum.(intervals))])
    maximum(maximum.(intervals)) != upper_buond && push!(intervals, [maximum(maximum.(intervals)), upper_buond])

    f_d = (d, i) -> cdf(d, i[2]) - cdf(d, i[1])
    # states_symbols = [Symbol(i) for i in intervals]

    states = OrderedDict{Vector{Symbol},Dict{Symbol,Real}}()
    for (key, dist) in node.distribution
        states[key] = Dict(Symbol.(intervals) .=> f_d.(dist, intervals))
    end

    discrete_node = DiscreteStandardNode(Symbol(string(node.name) * "_d"), node.parents, states)

    f_c = i -> begin
        a = isfinite.(i)
        all(a) ? Uniform(i...) : Truncated(Normal(i[a][1], variance), i...)
    end

    distributions = OrderedDict([Symbol(i)] => f_c(i) for i in intervals)

    continuous_node = ContinuousStandardNode(Symbol(string(node.name) * "_c"), [discrete_node], distributions)

    ## Adding continuous node as parents of children of the discretized node
    for child in get_children(ebn_new, node)
        deleteat!(child.parents, findall(x -> x == node, child.parents))
        push!(child.parents, continuous_node)
    end
    nodes = append!(ebn_new.nodes, [continuous_node, discrete_node])
    deleteat!(nodes, findall(x -> x == node, nodes))

    EnhancedBayesianNetwork(nodes)
end

