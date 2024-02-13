# function evaluate(ebn::EnhancedBayesianNetwork)
#     ## Discretize ebn
#     disc_ebn = discretize(ebn)
#     ## transfer all possible continuous functional node's model to their discrete functional children
#     trans_ebn = EnhancedBayesianNetwork(EnhancedBayesianNetworks._transfer_continuous(disc_ebn))

#     continuous_node_to_reduce = filter(j -> !isa(j, FunctionalNode), filter(x -> isa(x, ContinuousNode), trans_ebn.nodes))
#     indices = [findfirst(x -> x == c, trans_ebn.nodes) for c in continuous_node_to_reduce]
#     ## reducibility test
#     if EnhancedBayesianNetworks._is_reducible(trans_ebn.dag, indices)
#         nodes = trans_ebn.nodes
#         functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
#         for to_eval in functional_nodes
#             ## functional node evaluation
#             evaluated_node = evaluate(to_eval)
#             nodes = _replace_node(nodes, to_eval, evaluated_node)
#             if isa(evaluated_node, ContinuousChildNode)
#                 if !isempty(evaluated_node.discretization.intervals)
#                     nodes = discretize(EnhancedBayesianNetwork(nodes))
#                 end
#             end
#         end
#         return EnhancedBayesianNetwork(nodes)
#     else
#         error("Not A-Cyclic Network")
#     end
# end


function evaluate(ebn::EnhancedBayesianNetwork, index)
    ## Discretize ebn
    disc_ebn = discretize(ebn)
    ## transfer all possible continuous functional node's model to their discrete functional children
    trans_ebn = EnhancedBayesianNetwork(EnhancedBayesianNetworks._transfer_continuous(disc_ebn))
    nodes = trans_ebn.nodes
    functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)
    i = index
    @show(i, functional_nodes[i].name)
    evaluated_node = evaluate(functional_nodes[i])
    nodes = EnhancedBayesianNetworks._replace_node(nodes, functional_nodes[i], evaluated_node)
    if isa(evaluated_node, ContinuousChildNode)
        if !isempty(evaluated_node.discretization.intervals)
            ebn = EnhancedBayesianNetworks._discretize(nodes)
            nodes = ebn.nodes
        end
    end
    # functional_nodes = filter(x -> isa(x, FunctionalNode), nodes)

    return nodes
end


function _transfer_continuous(ebn::EnhancedBayesianNetwork)
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), ebn.nodes)
    nodes = ebn.nodes
    for c in continuous_functional
        nodes = _transfer_single_continuous_functional(nodes, c)
    end
    return nodes
end

function _transfer_single_continuous_functional(nodes::AbstractVector{AbstractNode}, node::ContinuousFunctionalNode)
    node_children = filter(x -> node ∈ x.parents, filter(x -> !isa(x, RootNode), nodes))
    if isempty(node.discretization.intervals) && !isempty(node_children)
        nodes = setdiff(nodes, [node, node_children...])
        children = AbstractNode[]
        for child in node_children
            for n in [node, node.parents...]
                index = findfirst(x -> isequal(x, n), child.parents)
                isnothing(index) ? continue : deleteat!(child.parents, index)
            end
            append!(child.parents, node.parents)
            prepend!(child.models, node.models)
            push!(children, child)
        end
        append!(nodes, children)
        return nodes
    else
        return nodes
    end
end


# function _replace_node(nodes::AbstractVector{AbstractNode}, old::FunctionalNode, new::ChildNode)
#     if isa(old, DiscreteNode) && isa(new, ContinuousNode)
#         error("cannot replace ContinuousNodes with DiscreteNodes or viceversa")
#     end
#     # remove original continuous nodes
#     nodes = filter(x -> !isequal(x, old), nodes)
#     for node in nodes
#         if isa(node, RootNode)
#             continue
#         end
#         if old in node.parents
#             node.parents[:] = [filter(x -> !isequal(x, old), node.parents)..., new]
#         end
#     end
#     push!(nodes, new)
#     return nodes
# end


function _replace_node(nodes::AbstractVector{AbstractNode}, old::FunctionalNode, new::ChildNode)
    index = findfirst(x -> isequal(x, old), nodes)
    deleteat!(nodes, index)
    for node in nodes
        if isa(node, RootNode)
            continue
        else
            if old ∈ node.parents
                @show(node.name)
                # node.parents[:] = [filter(x -> !isequal(x, old), node.parents)..., new]
            end
        end
    end
    insert!(nodes, index, new)
end