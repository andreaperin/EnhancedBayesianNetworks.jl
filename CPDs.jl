using Distributions
using Discretizers


"""
    Definition of the NodeName constant
"""
const global NodeName = Symbol

const global NodeNames = AbstractVector{NodeName}
const global NodeNameUnion = Union{NodeName,NodeNames}

# nodeconvert(::Type{NodeNames}, names::NodeNameUnion) = names
# nodeconvert(::Type{NodeNames}, name::NodeName) = [name]

"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const global Assignment = Dict{NodeName,Any}

# nodenames(a::Assignment) = collect(keys(a))

# function consistent(a::Assignment, b::Assignment)
#     for key in keys(a)
#         if haskey(b, key) && b[key] != a[key]
#             return false
#         end
#     end
#     return true
# end

"""
    Definition of the CPD AbstractType
"""
abstract type CPD{D<:Distribution} end

"""
    An Object for mapping each distribution to a MapableTypes::Union{AbstractString, Symbol}
"""
const global MapableTypes = Union{AbstractString,Symbol}
struct MappedAliasTable <: Sampleable{Univariate,Discrete}
    alias::Distributions.AliasTable
    map::CategoricalDiscretizer
end

struct NamedCategorical{N<:MapableTypes} <: DiscreteUnivariateDistribution
    cat::Categorical
    map::CategoricalDiscretizer{N,Int}
    prob_dict::Dict{Symbol,Float64}
end

Distributions.ncategories(s::MappedAliasTable) = Distributions.ncategories(s.alias)

## TODO Add this to logs
function NamedCategorical(items::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    if sum(probs) != 1
        println("$items => Not normalized probabilities => automatically normalized")
    end
    cat = Categorical(probs ./ sum(probs))
    map = CategoricalDiscretizer(items)
    prob_dict = Dict(items .=> probs ./ sum(probs))
    NamedCategorical{N}(cat, map, prob_dict)
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
mutable struct RootCPD{D<:Distribution} <: CPD{D}
    target::NodeName
    parents::NodeNames
    distributions::D
    prob_dict::Union{Dict{Symbol,D},Dict{Symbol,Float64}}
end

function RootCPD(target::NodeName, distributions::D) where {D<:Distribution}
    if isa(distributions, NamedCategorical)
        prob_dict = distributions.prob_dict
    else
        prob_dict = Dict{Symbol,D}(:nothing => distributions)
    end
    RootCPD(target, NodeName[], distributions, prob_dict)
end



name(cpd::RootCPD) = cpd.target
parents(cpd::RootCPD) = cpd.parents
(cpd::RootCPD)(a::Assignment) = cpd.distributions # no update
(cpd::RootCPD)() = (cpd)(Assignment()) # cpd()
(cpd::RootCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)
nparams(cpd::RootCPD) = paramcount(params(cpd.distributions))

"""
A categorical CPD, P(x|parents(x)) where all parents are discrete integers 1:N.
The ordering of `distributions` array follows the sequence: 
X,Y,Z
1,1,1
2,1,1
1,2,1
2,2,1
1,1,2
...
"""
struct CategoricalCPD{D<:Distribution} <: CPD{D}
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{D}
    prob_dict::Dict{Tuple,D}
end

function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}) where {D<:Distribution}
    f = x -> collect(1:1:x)
    combinations = sort(vec(collect(Iterators.product(f.(parental_ncategories)...))))
    prob_dict = Dict{Tuple,D}(combinations .=> distributions)
    CategoricalCPD(target, parents, parental_ncategories, distributions, prob_dict)
end

## Second way to define the CategoricalCPD trought a dict::(1,1) => [NamedCategorical1, NamedCategorical2]
function CategoricalCPD(target::NodeName, parents::NodeNames, prob_dict::Dict{Tuple,D}) where {D<:Distribution}
    distributions = Vector{D}(collect(values(sort(prob_dict))))
    f = x -> collect(x)
    combinations = mapreduce(permutedims, vcat, f.(collect(keys(prob_dict))))
    parental_ncategories = findmax(combinations, dims=1)[1]
    CategoricalCPD(target, parents, parental_ncategories, distributions, prob_dict)
end


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

