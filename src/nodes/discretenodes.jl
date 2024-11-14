@auto_hash_equals struct DiscreteNode <: AbstractNode
    name::Symbol
    cpt::DataFrame
    parameters::Dict{Symbol,Vector{Parameter}}
    additional_info::Dict{Vector{Symbol},Dict}

    function DiscreteNode(name::Symbol, cpt, parameters::Dict{Symbol,Vector{Parameter}}, additional_info::Dict{Vector{Symbol},Dict})
        new(name, _cpt(cpt), parameters, additional_info)
    end
end

function DiscreteNode(name::Symbol, cpt::DataFrame)
    DiscreteNode(name, cpt, Dict{Symbol,Vector{Parameter}}(), Dict{Vector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::DataFrame, parameters::Dict{Symbol,Vector{Parameter}})
    DiscreteNode(name, cpt, parameters, Dict{Vector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::Dict{Vector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol})
    cpt = _cpt(cpt, indices)
    DiscreteNode(name, cpt, Dict{Symbol,Vector{Parameter}}(), Dict{Vector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::Dict{Vector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol}, parameters::Dict{Symbol,Vector{Parameter}})
    cpt = _cpt(cpt, indices)
    DiscreteNode(name, cpt, parameters, Dict{Vector{Symbol},Dict}())
end

function _cpt(x::Dict{Vector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol})
    cpt = DataFrame()
    for (i, name) in enumerate(indices)
        cpt[!, name] = map(x -> x[i], collect(keys(x)))
    end
    cpt[!, :Prob] = collect(values(x))
    return cpt
end

_cpt(x::DataFrame) = x

function _states(node::DiscreteNode)
    return unique(node.cpt[!, node.name])
end

function _scenarios(node::DiscreteNode)
    return copy.(eachrow(node.cpt[!, Not(node.name, :Prob)]))
end

function _parameters_with_evidence(node::DiscreteNode; evidence::Evidence)
    if node.name ∉ keys(evidence)
        error("evidence $evidence does not contain the node $(node.name)")
    else
        return node.parameters[evidence[node.name]]
    end
end

function _is_precise(node::DiscreteNode)
    all(isa.(node.cpt[!, :Prob], Real))
end

function _extreme_points(node::DiscreteNode)
    if _is_precise(node)
        return [node]
    else
        ext_points = _extreme_points_probabilities(node.cpt)
        function f(ext_point)
            df = node.cpt[!, Not(:Prob)]
            df[!, :Prob] = ext_point
            return df
        end
        dfs = f.(ext_points)
        return map(df -> DiscreteNode(node.name, df, node.parameters, node.additional_info), dfs)
    end
end

function _extreme_points_probabilities(cpt::DataFrame)
    n = nrow(cpt)
    A = zeros(2 * n, n)
    A[collect(1:2:2*n), :] = Matrix(-1.0I, n, n)
    A[collect(2:2:2*n), :] = Matrix(1.0I, n, n)
    A = vcat(A, [-ones(n)'; ones(n)'])

    b = collect(Iterators.flatten(cpt[!, :Prob]))
    b[collect(1:2:2*n)] = -b[collect(1:2:2*n)]
    b = vcat(b, [-1 1]')

    h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
    v = doubledescription(h)
    return v.points.points
end