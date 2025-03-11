@auto_hash_equals struct DiscreteNode <: AbstractDiscreteNode
    name::Symbol
    cpt::DiscreteConditionalProbabilityTable
    parameters::Dict{Symbol,Vector{Parameter}}
    additional_info::Dict{AbstractVector{Symbol},Dict}

    function DiscreteNode(
        name::Symbol,
        cpt::DiscreteConditionalProbabilityTable,
        parameters::Dict{Symbol,Vector{Parameter}},
        additional_info::Dict{AbstractVector{Symbol},Dict}
    )
        if String(name) ∉ names(cpt.data)
            error("defined cpt does not contain a column refered to node name $name: $cpt")
        end
        _verify_probabilities!(cpt, name)
        _verify_parameters(cpt, parameters, name)
        ## setting node column as last column before :Π 
        select!(cpt.data, Not([name, :Π]), name, :Π)
        sort!(cpt.data)
        new(name, cpt, parameters, additional_info)
    end
end

function DiscreteNode(name::Symbol, cpt::DiscreteConditionalProbabilityTable)
    DiscreteNode(name, cpt, Dict{Symbol,Vector{Parameter}}(), Dict{AbstractVector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::DiscreteConditionalProbabilityTable, parameters::Dict{Symbol,Vector{Parameter}})
    DiscreteNode(name, cpt, parameters, Dict{AbstractVector{Symbol},Dict}())
end

states(node::DiscreteNode) = states(node.cpt, node.name)

scenarios(node::DiscreteNode) = scenarios(node.cpt, node.name)

isprecise(node::DiscreteNode) = isprecise(node.cpt)

isroot(node::DiscreteNode) = isroot(node.cpt)

_scenarios_cpt(node::DiscreteNode) = _scenarios_cpt(node.cpt, node.name)


# function _parameters_with_evidence(node::DiscreteNode, evidence::Evidence)
#     if node.name ∉ keys(evidence)
#         error("evidence $evidence does not contain the node $(node.name)")
#     else
#         return node.parameters[evidence[node.name]]
#     end
# end

function _extreme_points(node::DiscreteNode)
    if isprecise(node)
        return [node]
    else
        sub_cpts = EnhancedBayesianNetworks._scenarios_cpt(node.cpt, node.name)
        dfs = map(sc -> EnhancedBayesianNetworks._extreme_points_dfs(sc), sub_cpts)
        dfsa = vec(collect(Iterators.product(dfs...)))
        res = map(df -> vcat(df...), dfsa)
        cpts = map(r -> DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(r), res)
        map(cp -> DiscreteNode(node.name, cp, node.parameters, node.additional_info), cpts)
    end
end

function _extreme_points_dfs(sub_cpt::DataFrame)
    ext_points = _extreme_points_probabilities(sub_cpt)
    function _replace_prob(sub_cpt, ext_points)
        res = []
        for e in ext_points
            df = deepcopy(sub_cpt)
            df[!, :Π] = e
            push!(res, df)
        end
        return res
    end
    return _replace_prob(sub_cpt, ext_points)
end

function _extreme_points_probabilities(sub_cpt::DataFrame)
    if all(isa.(sub_cpt[!, :Π], Tuple))
        n = nrow(sub_cpt)
        A = zeros(2 * n, n)
        A[collect(1:2:2*n), :] = Matrix(-1.0I, n, n)
        A[collect(2:2:2*n), :] = Matrix(1.0I, n, n)
        A = vcat(A, [-ones(n)'; ones(n)'])

        b = collect(Iterators.flatten(sub_cpt[!, :Π]))
        b[collect(1:2:2*n)] = -b[collect(1:2:2*n)]
        b = vcat(b, [-1 1]')

        h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
        v = doubledescription(h)
    else
        error("Precise conditional probability table does not have extreme points: $sub_cpt")
    end
    return v.points.points
end