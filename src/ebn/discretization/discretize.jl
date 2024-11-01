function _discretize!(net::EnhancedBayesianNetwork)
    continuous_nodes = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), net.nodes)
    evidence_nodes = filter(n -> !isempty(n.discretization.intervals), continuous_nodes)
    discretizations_tuples = map(n -> (n, get_parents(net, n)[3], get_children(net, n)[3], _discretize(n)), evidence_nodes)
    for tup in discretizations_tuples
        node = tup[1]
        parents = tup[2]
        children = tup[3]
        disc_new = tup[4][1]
        cont_new = tup[4][2]
        _remove_node!(net, node)
        _add_node!(net, disc_new)
        _add_node!(net, cont_new)
        add_child!(net, disc_new, cont_new)
        for par in parents
            add_child!(net, par, disc_new)
        end
        for ch in children
            add_child!(net, cont_new, ch)
        end
        order_net!(net)
    end
    return nothing
end

## RootNode
function _discretize(node::ContinuousRootNode)
    intervals = _format_interval(node)
    states_symbols = Symbol.(intervals)
    states = Dict(states_symbols .=> _discretize(node.distribution, intervals))
    discrete_node = DiscreteRootNode(Symbol(string(node.name) * "_d"), states)
    ## Adding continuous node as parents of children of the discretized node
    distribution_symbols = [[i] for i in states_symbols]
    distribution = Dict(distribution_symbols .=> _truncate.(Ref(node.distribution), intervals))
    continuous_node = ContinuousChildNode(node.name, distribution)
    return [discrete_node, continuous_node]
end

# Child Node
function _discretize(node::ContinuousChildNode)
    intervals = _format_interval(node)
    k = keys(node.distribution)
    dists = values(node.distribution)
    states = [(key, Dict(Symbol.(intervals) .=> _discretize(dist, intervals))) for (key, dist) in zip(k, dists)]
    states = Dict(states)
    discrete_node = DiscreteChildNode(Symbol(string(node.name) * "_d"), states)
    distribution_symbols = [[Symbol(i)] for i in intervals]
    distribution = Dict(distribution_symbols .=> _approximate(intervals, node.discretization.sigma))
    continuous_node = ContinuousChildNode(node.name, distribution)
    return [discrete_node, continuous_node]
end

## Auxiliary function
function _discretize(dist::UnivariateDistribution, intervals::Vector)
    return cdf.(dist, getindex.(intervals, 2)) .- cdf.(dist, getindex.(intervals, 1))
end

function _discretize(dist::UnamedProbabilityBox, intervals::Vector)
    p_box = ProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, :temp)
    right_bounds = cdf.(Ref(p_box), getindex.(intervals, 2))
    left_bounds = cdf.(Ref(p_box), getindex.(intervals, 1))
    map((r, l) -> [minimum([r.lb - l.lb, r.ub - l.ub]), maximum([r.lb - l.lb, r.ub - l.ub])], right_bounds, left_bounds)
end

function _discretize(dist::Tuple{T,T}, intervals::Vector) where {T<:Real}
    repeat([[0, 1]], length(intervals))
end

function _approximate(intervals::Vector, λ::Real)
    dists = map(intervals) do i
        finite = isfinite.(i)
        if all(finite)
            return Uniform(i...)
        elseif finite[end] == true
            return -Exponential(λ) - finite[end]
        elseif finite[1] == true
            return Exponential(λ) + finite[1]
        end
    end
    return dists
end

function _format_interval(node::Union{ContinuousChildNode,ContinuousRootNode})
    intervals = node.discretization.intervals
    intervals = convert(Vector{Float64}, intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]
    lower_bound, upper_bound = _get_node_distribution_bounds(node)
    if minimum(min) > lower_bound
        @warn "node $(node.name) has minimum intervals value $min > support lower bound $lower_bound. Lower bound will be used as intervals start"
        insert!(intervals, 1, lower_bound)
    end
    if minimum(min) < lower_bound
        @warn "node $(node.name) has minimum intervals value $min < support lower bound $lower_bound. Lower bound will be used as intervals start"
        deleteat!(intervals, intervals .<= lower_bound)
        insert!(intervals, 1, lower_bound)
    end
    if maximum(max) < upper_bound
        @warn "node $(node.name) has maximum intervals value $max < support upper bound $upper_bound. Upper bound will be used as intervals end"
        push!(intervals, upper_bound)
    end
    if maximum(max) > upper_bound
        @warn "node $(node.name) has maximum intervals value $max > support upper bound $upper_bound. Upper bound will be used as intervals end"
        deleteat!(intervals, intervals .>= upper_bound)
        push!(intervals, upper_bound)
    end
    return [[intervals[i], intervals[i+1]] for i in 1:length(intervals)-1]
end