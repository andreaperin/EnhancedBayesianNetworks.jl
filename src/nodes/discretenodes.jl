@auto_hash_equals struct DiscreteNode <: AbstractNode
    name::Symbol
    cpt::DataFrame
    parameters::Dict{Symbol,Vector{Parameter}}
    additional_info::Dict{AbstractVector{Symbol},Dict}

    function DiscreteNode(name::Symbol, cpt, parameters::Dict{Symbol,Vector{Parameter}}, additional_info::Dict{AbstractVector{Symbol},Dict})
        cpt = _verify_cpt_and_normalize!(cpt, name)
        _verify_parameters(cpt, parameters, name)
        new(name, _cpt(cpt), parameters, additional_info)
    end
end

function DiscreteNode(name::Symbol, cpt::DataFrame)
    DiscreteNode(name, cpt, Dict{Symbol,Vector{Parameter}}(), Dict{AbstractVector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::DataFrame, parameters::Dict{Symbol,Vector{Parameter}})
    DiscreteNode(name, cpt, parameters, Dict{AbstractVector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::Dict{AbstractVector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol})
    cpt = _cpt(cpt, indices)
    DiscreteNode(name, cpt, Dict{Symbol,Vector{Parameter}}(), Dict{AbstractVector{Symbol},Dict}())
end

function DiscreteNode(name::Symbol, cpt::Dict{AbstractVector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol}, parameters::Dict{Symbol,Vector{Parameter}})
    cpt = _cpt(cpt, indices)
    DiscreteNode(name, cpt, parameters, Dict{AbstractVector{Symbol},Dict}())
end

function _verify_cpt_and_normalize!(cpt::DataFrame, name::Symbol)
    _verify_cpt_coherence(cpt)
    if ncol(cpt) == 2     ## Root Nodes
        sub_cpts = [cpt]
    else    ## Child Nodes
        scenarios = unique!(map(s -> _by_row(s), _scenarios(cpt, name)))
        sub_cpts = map(e -> subset(cpt, e), scenarios)
    end
    _verify_precise_probabilities_values(cpt)
    _verify_imprecise_probabilities_values(cpt)
    map(sc -> _verify_imprecise_exhaustiveness(sc), sub_cpts)
    return mapreduce(sc -> _verify_precise_exhaustiveness_and_normalize!(sc), vcat, sub_cpts)
end

function _verify_parameters(cpt::DataFrame, parameters::Dict{Symbol,Vector{Parameter}}, name::Symbol)
    if !isempty(parameters)
        if !issetequal(_states(cpt, name), keys(parameters))
            error("parameters keys $(keys(parameters)) must be coherent with states $(_states(cpt, name))")
        end
    end
end

_verify_parameters(node::DiscreteNode) = _verify_parameters(node.cpt, node.parameters, node.name)

function _states(cpt::DataFrame, name::Symbol)
    unique(cpt[!, name])
end

_states(node::DiscreteNode) = _states(node.cpt, node.name)

function _scenarios(cpt::DataFrame, name::Symbol)
    scenarios = copy.(eachrow(cpt[!, Not(name, :Prob)]))
    return map(s -> Dict(pairs(s)), scenarios)
end

_scenarios(node::DiscreteNode) = _scenarios(node.cpt, node.name)

function _by_row(evidence::Evidence)
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end

function _cpt(x::Dict{AbstractVector{Symbol},<:AbstractDiscreteProbability}, indices::AbstractVector{Symbol})
    cpt = DataFrame()
    for (i, name) in enumerate(indices)
        cpt[!, name] = map(x -> x[i], collect(keys(x)))
    end
    cpt[!, :Prob] = collect(values(x))
    return cpt
end

_cpt(x::DataFrame) = x

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