using Distributions
using Discretizers

"""
    Definition of the NodeName constant
"""
const NodeName = Symbol

const NodeNames = AbstractVector{NodeName}
const NodeNameUnion = Union{NodeName,NodeNames}

nodeconvert(::Type{NodeNames}, names::NodeNameUnion) = names
nodeconvert(::Type{NodeNames}, name::NodeName) = [name]

"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const Assignment = Dict{NodeName,Any}
nodenames(a::Assignment) = collect(keys(a))

function consistent(a::Assignment, b::Assignment)
    for key in keys(a)
        if haskey(b, key) && b[key] != a[key]
            return false
        end
    end
    return true
end

"""
    Definition of the CPD AbstractType
"""
## TODO: CPD should accept as argument also a SRP
abstract type CPD{D<:Distribution} end

"""
    An Object for mapping each distribution to a MapableTypes::Union{AbstractString, Symbol}
"""
const MapableTypes = Union{AbstractString,Symbol}

struct MappedAliasTable <: Sampleable{Univariate,Discrete}
    alias::Distributions.AliasTable
    map::CategoricalDiscretizer
end

struct NamedCategorical{N<:MapableTypes} <: DiscreteUnivariateDistribution
    cat::Categorical
    map::CategoricalDiscretizer{N,Int}
end

Distributions.ncategories(s::MappedAliasTable) = Distributions.ncategories(s.alias)

## TODO Add this to logs
function NamedCategorical(items::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    if sum(probs) != 1
        println("$items => Not normalized probabilities => automatically normalized")
    end
    cat = Categorical(probs ./ sum(probs))
    map = CategoricalDiscretizer(items)
    NamedCategorical{N}(cat, map)
end

Distributions.ncategories(d::NamedCategorical) = Distributions.ncategories(d.cat)
Distributions.probs(d::NamedCategorical) = Distributions.probs(d.cat)
Distributions.params(d::NamedCategorical) = Distributions.params(d.cat)


"""
A CPD for which the distribution never changes.
    target: name of the CPD's variable
    parents: list of parent variables.
    distributions: a Distributions.jl distribution
While a RootCPD can have parents, their assignments will not affect the distribution.
"""
mutable struct RootCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    distributions::D
end

RootCPD(target::NodeName, distributions::Distribution) = RootCPD(target, NodeName[], distributions)

name(cpd::RootCPD) = cpd.target

parents(cpd::RootCPD) = cpd.parents

(cpd::RootCPD)(a::Assignment) = cpd.distributions # no update
(cpd::RootCPD)() = (cpd)(Assignment()) # cpd()
(cpd::RootCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)
nparams(cpd::RootCPD) = paramcount(params(cpd.distributions))

"""
A categorical distribution, P(x|parents(x)) where all parents are discrete integers 1:N.
The ordering of `distributions` array follows the sequence: 
X,Y,Z
1,1,1
2,1,1
1,2,1
2,2,1
1,1,2
...
"""
struct CategoricalCPD{D} <: CPD{D}
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int} # list of instantiation counts for each parent, in same order as parents
    distributions::Vector{D}  # a vector of distributions in DMU order
end

CategoricalCPD(target::NodeName, d::D) where {D<:Distribution} = CategoricalCPD(target, NodeName[], Int[], D[d])

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parents
nparams(cpd::CategoricalCPD) = sum(d -> paramcount(params(d)), cpd.distributions)

function (cpd::CategoricalCPD)(a::Assignment=Assignment())
    if isempty(cpd.parents)
        return first(cpd.distributions)
    else
        sub = [a[p] for p in cpd.parents]
        shape = ntuple(i -> cpd.parental_ncategories[i],
            length(cpd.parental_ncategories))
        ind = LinearIndices(shape)[sub...]
        return cpd.distributions[ind]
    end
end
(cpd::CategoricalCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair))
Distributions.ncategories(cpd::CategoricalCPD) = ncategories(first(cpd.distributions))