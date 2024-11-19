@auto_hash_equals struct DiscreteNode <: AbstractNode
    name::Symbol
    cpt::DataFrame
    parameters::Dict{Symbol,Vector{Parameter}}
    additional_info::Dict{AbstractVector{Symbol},Dict}

    function DiscreteNode(name::Symbol, cpt, parameters::Dict{Symbol,Vector{Parameter}}, additional_info::Dict{AbstractVector{Symbol},Dict})
        if String(name) ∉ names(cpt)
            error("defined cpt does not contain a column refered to node name $name: $cpt")
        end
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
    sub_cpts = _scenarios_cpt(cpt, name)
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

function _states(cpt::DataFrame, name::Symbol)
    unique(cpt[!, name])
end

_states(node::DiscreteNode) = _states(node.cpt, node.name)

function _scenarios(cpt::DataFrame, name::Symbol)
    scenarios = copy.(eachrow(cpt[!, Not(name, :Prob)]))
    return unique(map(s -> Dict(pairs(s)), scenarios))
end

_scenarios(node::DiscreteNode) = _scenarios(node.cpt, node.name)

function _scenarios_cpt(cpt::DataFrame, name::Symbol)
    if ncol(cpt) == 2     ## Root Nodes
        sub_cpts = [cpt]
    else    ## Child Nodes
        scenarios = unique!(map(s -> _by_row(s), _scenarios(cpt, name)))
        sub_cpts = map(e -> subset(cpt, e), scenarios)
    end
    return sub_cpts
end

_scenarios_cpt(node::DiscreteNode) = _scenarios_cpt(node.cpt, node.name)

function _cpt(x::Dict{Vector{Symbol},<:AbstractDiscreteProbability}, names::AbstractVector{Symbol})
    cpt = DataFrame()
    for (i, name) in enumerate(names)
        cpt[!, name] = map(x -> x[i], collect(keys(x)))
    end
    cpt[!, :Prob] = collect(values(x))
    return cpt
end

function _cpt(x::Dict{Symbol,<:AbstractDiscreteProbability}, name::Symbol)
    cpt = DataFrame()
    cpt[!, name] = collect(keys(x))
    cpt[!, :Prob] = collect(values(x))
    return cpt
end

_cpt(x::DataFrame) = x

function _parameters_with_evidence(node::DiscreteNode, evidence::Evidence)
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
        sub_cpts = _scenarios_cpt(node.cpt, node.name)
        dfs = map(sc -> _extreme_points_dfs(sc), sub_cpts)
        dfsa = vec(collect(Iterators.product(dfs...)))
        res = map(df -> vcat(df...), dfsa)
        return map(r -> DiscreteNode(node.name, r, node.parameters, node.additional_info), res)
    end
end

function _extreme_points_dfs(sub_cpt::DataFrame)
    ext_points = _extreme_points_probabilities(sub_cpt)
    function _replace_prob(sub_cpt, ext_points)
        res = []
        for e in ext_points
            df = deepcopy(sub_cpt)
            df[!, :Prob] = e
            push!(res, df)
        end
        return res
    end
    return _replace_prob(sub_cpt, ext_points)
end

function _extreme_points_probabilities(sub_cpt::DataFrame)
    if all(isa.(sub_cpt[!, :Prob], Vector))
        n = nrow(sub_cpt)
        A = zeros(2 * n, n)
        A[collect(1:2:2*n), :] = Matrix(-1.0I, n, n)
        A[collect(2:2:2*n), :] = Matrix(1.0I, n, n)
        A = vcat(A, [-ones(n)'; ones(n)'])

        b = collect(Iterators.flatten(sub_cpt[!, :Prob]))
        b[collect(1:2:2*n)] = -b[collect(1:2:2*n)]
        b = vcat(b, [-1 1]')

        h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
        v = doubledescription(h)
    else
        error("Precise conditional probability table does not have extreme points: $sub_cpt")
    end
    return v.points.points
end