function _transfer_continuous!(nodes::AbstractVector{AbstractNode})
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), nodes)
    for c in continuous_functional
        nodes = _transfer_single_continuous_functional!(nodes, c)
    end
    return nodes
end

function _transfer_single_continuous_functional!(nodes::AbstractVector{AbstractNode}, node::ContinuousFunctionalNode)
    node_children = filter(x -> node ∈ x.parents, filter(x -> !isa(x, RootNode), nodes))
    if isempty(node.discretization.intervals) && !isempty(node_children)
        nodes = setdiff(nodes, [node, node_children...])
        children = AbstractNode[]
        for child in node_children
            for n in [node, node.parents...]
                index = findfirst(x -> x == n, child.parents)
                isnothing(index) ? continue : deleteat!(child.parents, index)
            end
            append!(child.parents, node.parents)
            prepend!(child.models, node.models)
            push!(children, child)
        end
        append!(nodes, children)
    end
    return nodes
end
