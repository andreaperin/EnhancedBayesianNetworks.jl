function _discretize!(nodes::AbstractVector{AbstractNode})
    continuous_nodes = filter(n -> !isa(n, FunctionalNode), filter(n -> isa(n, ContinuousNode), nodes))
    evidence_nodes = filter(n -> !isempty(n.discretization.intervals), continuous_nodes)
    for n in evidence_nodes
        continuous_node, discretized_node = _discretize(n)
        # remove original continuous nodes
        index = findfirst(x -> isequal(x, n), nodes)
        deleteat!(nodes, index)
        # update child nodes
        for node in nodes
            if isa(node, RootNode)
                continue
            end
            if n in node.parents
                node.parents[:] = [filter(x -> !isequal(x, n), node.parents)..., continuous_node]
            end
        end
        append!(nodes, [continuous_node, discretized_node])
    end
    return nodes
end

## Single Node Continuous
function _discretize(node::ContinuousRootNode)
    intervals = _format_interval(node)
    states_symbols = Symbol.(intervals)
    states = Dict(states_symbols .=> _discretize(node.distribution, intervals))
    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)
    ## Adding continuous node as parents of children of the discretized node
    distribution_symbols = [[i] for i in states_symbols]
    distribution = Dict(distribution_symbols .=> _truncate.(node.distribution, intervals))
    continuous_node = ContinuousChildNode(node.name, [discrete_node], distribution)
    return continuous_node, discrete_node
end

## Single Node Discrete
function _discretize(node::ContinuousChildNode)
    intervals = _format_interval(node)
    k = keys(node.distribution)
    dists = values(node.distribution)
    states = [(key, Dict(Symbol.(intervals) .=> _discretize(dist, intervals))) for (key, dist) in zip(k, dists)]
    states = Dict(states)
    discrete_node = DiscreteChildNode(Symbol(string(node.name) * "_d"), node.parents, states)
    distribution_symbols = [[Symbol(i)] for i in intervals]
    distribution = Dict(distribution_symbols .=> _approximate(intervals, node.discretization.sigma))
    continuous_node = ContinuousChildNode(node.name, [discrete_node], distribution)
    return continuous_node, discrete_node
end

## Auxiliary function
function _discretize(dist::UnivariateDistribution, intervals::Vector)
    return cdf.(dist, getindex.(intervals, 2)) .- cdf.(dist, getindex.(intervals, 1))
end

function _approximate(intervals::Vector, σ::Real)
    dists = map(intervals) do i
        finite = isfinite.(i)
        if all(finite)
            return Uniform(i...)
        end
        return truncated(Normal(i[finite][1], σ), i...)
    end
    return dists
end

function _format_interval(node::Union{ContinuousChildNode,ContinuousRootNode})
    intervals = deepcopy(node.discretization.intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]
    lower_bound, upper_bound = _get_node_distribution_bounds(node)
    if minimum(min) != lower_bound
        @warn "Minimum intervals value $min >= support lower bound $lower_bound. Lower bound will be used as intervals start."
        insert!(intervals, 1, lower_bound)
    end
    if maximum(max) != upper_bound
        @warn "Maximum intervals value $max <= support upper bound $upper_bound. Upper bound will be used as intervals end."
        push!(intervals, upper_bound)
    end
    return [[intervals[i], intervals[i+1]] for i in 1:length(intervals)-1]
end