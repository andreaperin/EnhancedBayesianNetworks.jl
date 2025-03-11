
const PreciseContinuousInput = UnivariateDistribution
const ImpreciseContinuousInput = Union{Tuple{<:Real,<:Real},UnamedProbabilityBox}
const PreciseDiscreteProbability = Real
const ImpreciseDiscreteProbability = Tuple{<:Real,<:Real}

const DiscreteProbability = Union{PreciseDiscreteProbability,ImpreciseDiscreteProbability}
const ContinuousInput = Union{PreciseContinuousInput,ImpreciseContinuousInput}

abstract type AbstractConditionalProbabilityTable end

@auto_hash_equals struct DiscreteConditionalProbabilityTable{P<:DiscreteProbability} <: AbstractConditionalProbabilityTable
    data::DataFrame
    function DiscreteConditionalProbabilityTable{P}(names::Union{Symbol,Vector{Symbol}}) where {P<:DiscreteProbability}
        names = wrap(names)
        data = DataFrame([name => Symbol[] for name in names])
        data[:, :Π] = P[]
        return new{P}(data)
    end
end

function DiscreteConditionalProbabilityTable{P}(data::DataFrame) where {P<:DiscreteProbability}
    cpt = DiscreteConditionalProbabilityTable{P}(Symbol.(names(data[!, Not(:Π)])))
    append!(cpt.data, data)
    return cpt
end

@auto_hash_equals struct ContinuousConditionalProbabilityTable{P<:ContinuousInput} <: AbstractConditionalProbabilityTable
    data::DataFrame
    function ContinuousConditionalProbabilityTable{P}(names::Union{Symbol,Vector{Symbol}}) where {P<:ContinuousInput}
        names = wrap(names)
        data = DataFrame([name => Symbol[] for name in names])
        data[:, :Π] = P[]
        return new{P}(data)
    end
end

ContinuousConditionalProbabilityTable{P}() where {P<:ContinuousInput} = ContinuousConditionalProbabilityTable{P}(Vector{Symbol}())

function ContinuousConditionalProbabilityTable{P}(data::DataFrame) where {P<:ContinuousInput}
    cpt = ContinuousConditionalProbabilityTable{P}(Symbol.(names(data[!, Not(:Π)])))
    append!(cpt.data, data)
    return cpt
end

function Base.setindex!(cpt::AbstractConditionalProbabilityTable, value, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    cpt_nodes = Symbol.(filter(i -> i != "Π", names(cpt.data)))
    if issetequal(evidence_nodes, cpt_nodes)
        cp = subset(cpt.data, selector, view=true)
        if isempty(cp)
            push!(cpt.data, (key..., Π=value))
        else
            @assert size(cp, 1) == 1
            cp.Π[1] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a CPT initialized with $cpt_nodes")
    end
    return nothing
end

function Base.getindex(cpt::AbstractConditionalProbabilityTable, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(cpt.data, selector, view=true)
    if isempty(cp)
        error("index not find in the CPT $cpt")
    else
        @assert size(cp, 1) == 1
        return cp.Π[1]
    end
end

function isprecise(cpt::AbstractConditionalProbabilityTable)
    isa(cpt, ContinuousConditionalProbabilityTable{PreciseContinuousInput}) | isa(cpt, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability})
end

function isroot(cpt::DiscreteConditionalProbabilityTable)
    length(names(cpt.data)) == 2
end

function isroot(cpt::ContinuousConditionalProbabilityTable)
    length(names(cpt.data)) == 1
end

function states(cpt::AbstractConditionalProbabilityTable, name::Symbol)
    unique(cpt.data[!, name])
end

#! figure out a better way to return scenarios other than a Vector of Ditionaries
function scenarios(cpt::AbstractConditionalProbabilityTable, name::Symbol)
    scenario = copy.(eachrow(cpt.data[!, Not(name, :Π)]))
    return unique(map(s -> Dict(pairs(s)), scenario))
end

function _scenarios_cpt(cpt::AbstractConditionalProbabilityTable, name::Symbol)
    if ncol(cpt.data) <= 2     ## Root Nodes
        sub_cpts = [cpt.data]
    else    ## Child Nodes
        scenario = unique!(map(s -> _by_row(s), scenarios(cpt, name)))
        sub_cpts = map(e -> subset(cpt.data, e), scenario)
    end
    return sub_cpts
end

function _by_row(evidence::Dict{Symbol,Symbol})
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end