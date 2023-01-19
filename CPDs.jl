using Distributions
using Discretizers
using UncertaintyQuantification

"""
    Definition of the NodeName constant
"""
const global NodeName = Symbol

const global NodeNames = AbstractVector{NodeName}
const global NodeNameUnion = Union{NodeName,NodeNames}

"""
    Definition of the Assignment constant as Dict{NodeName, Any}
"""
const global Assignment = Dict{NodeName,Any}


"""
    Definition of the CPD AbstractType
"""
abstract type CPD end

"""
    Definition of the Object SystemReliabilityProblem
"""

struct SystemReliabilityProblem
    models::Union{Array{<:UQModel},UQModel}
    performances::Dict{Symbol,Function}
    inputs::Union{Array{<:UQInput},UQInput}
    sim::AbstractMonteCarlo
end


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

function NamedCategorical(items::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    ## TODO Add this to logs
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
mutable struct RootCPD <: CPD
    target::NodeName
    parents::NodeNames
    distributions::Distribution
    prob_dict::Union{Dict{Symbol,Distribution},Dict{Symbol,Float64}}
end

function RootCPD(target::NodeName, distributions::Distribution)
    if isa(distributions, NamedCategorical)
        prob_dict = distributions.prob_dict
    else
        prob_dict = Dict{Symbol,Distribution}(:nothing => distributions)
    end
    RootCPD(target, NodeName[], distributions, prob_dict)
end


name(cpd::RootCPD) = cpd.target
parents(cpd::RootCPD) = cpd.parents
# (cpd::RootCPD)(a::Assignment) = cpd.distributions # no update
# (cpd::RootCPD)() = (cpd)(Assignment()) # cpd()
# (cpd::RootCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)
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
struct CategoricalCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{Distribution}
    prob_dict::Dict{Tuple,Distribution}
end

function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}) where {D<:Distribution}
    f = x -> collect(1:1:x)
    combinations = sort(vec(collect(Iterators.product(f.(parental_ncategories)...))))
    prob_dict = Dict{Tuple,Distribution}(combinations .=> distributions)
    CategoricalCPD(target, parents, parental_ncategories, distributions, prob_dict)
end
## Second way to define the CategoricalCPD trought a dict::(1,1) => [NamedCategorical1, NamedCategorical2]
function CategoricalCPD(target::NodeName, parents::NodeNames, prob_dict::Dict{Tuple,Distribution})
    distributions = Vector{Distribution}(collect(values(sort(prob_dict))))
    f = x -> collect(x)
    combinations = mapreduce(permutedims, vcat, f.(collect(keys(prob_dict))))
    parental_ncategories = vec(findmax(combinations, dims=1)[1])
    CategoricalCPD(target, parents, parental_ncategories, distributions, prob_dict)
end

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parents
nparams(cpd::CategoricalCPD) = sum(d -> paramcount(params(d)), cpd.distributions)
# function (cpd::CategoricalCPD)(a::Assignment=Assignment())
#     if isempty(cpd.parents)
#         return first(cpd.distributions)
#     else
#         sub = [a[p] for p in cpd.parents]
#         shape = ntuple(i -> cpd.parental_ncategories[i],
#             length(cpd.parental_ncategories))
#         ind = LinearIndices(shape)[sub...]
#         return cpd.distributions[ind]
#     end
# end
# (cpd::CategoricalCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair))
# Distributions.ncategories(cpd::CategoricalCPD) = ncategories(first(cpd.distributions))

"""
ModelCPD
"""
struct ModelCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{SystemReliabilityProblem}
    prob_dict::Dict{Tuple,SystemReliabilityProblem}
end



function ModelCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{S}) where {S<:SystemReliabilityProblem}
    ## Algo for parental n-categories (Not here, but after node definition need to check that parental-ncategories is equal to combination of discrete parents and grandparents)
    f = x -> collect(1:1:x)
    combinations = sort(vec(collect(Iterators.product(f.(parental_ncategories)...))))
    prob_dict = Dict{Tuple,SystemReliabilityProblem}(combinations .=> distributions)
    ModelCPD(target, parents, parental_ncategories, distributions, prob_dict)
end
## Second way to define the ModelCPD trought a dict::(1,1) => [NamedCategorical1, NamedCategorical2]
function ModelCPD(target::NodeName, parents::NodeNames, prob_dict::Dict{Tuple,S}) where {S<:SystemReliabilityProblem}
    distributions = Vector{SystemReliabilityProblem}(collect(values(sort(prob_dict))))
    f = x -> collect(x)
    combinations = mapreduce(permutedims, vcat, f.(collect(keys(prob_dict))))
    parental_ncategories = vec(findmax(combinations, dims=1)[1])
    ModelCPD(target, parents, parental_ncategories, distributions, prob_dict)
end

name(cpd::ModelCPD) = cpd.target
parents(cpd::ModelCPD) = cpd.parents
nparams(cpd::ModelCPD) = sum(d -> paramcount(params(d)), cpd.distributions)