using Distributions
using Discretizers
using UncertaintyQuantification

"""
    Definition of the NodeName constant
"""
const global NodeName = Symbol
const global NodeNames = AbstractVector{NodeName}

"""
    Definition of the SRP Struct
"""
# struct SystemReliabilityProblem
#     models::Union{Array{<:UQModel},UQModel}
#     performances::Dict{Symbol,Function}
#     inputs::Union{Array{<:UQInput},UQInput}
#     sim::AbstractMonteCarlo
# end
struct SystemReliabilityProblem
    f::Any
end

const global ProbabilityDictionaryEvidence = Union{Dict,Nothing}
const global ProbabilityDictionaryDistribution = Dict{Symbol,Union{Float64,Distribution}}

const global ProbabilityDictionary = NamedTuple{(:evidence, :distribution),Tuple{ProbabilityDictionaryEvidence,ProbabilityDictionaryDistribution}}
const global ProbabilityDictionaryFunctional = NamedTuple{(:evidence, :distribution),Tuple{ProbabilityDictionaryEvidence,SystemReliabilityProblem}}
"""
    Definition of the CPD AbstractType
"""
abstract type CPD end

"""
    An Object for mapping each distribution to a MapableTypes::Union{AbstractString, Symbol}
"""
const global MapableTypes = Union{AbstractString,Symbol}
struct MappedAliasTable <: Sampleable{Univariate,Discrete}
    alias::Distributions.AliasTable
    map::CategoricalDiscretizer
end

struct NamedCategorical{N<:MapableTypes} <: DiscreteUnivariateDistribution
    items::AbstractVector{N}
    probs::Vector{Float64}
    cat::Categorical
    map::CategoricalDiscretizer{N,Int}
end

Distributions.ncategories(s::MappedAliasTable) = Distributions.ncategories(s.alias)

function NamedCategorical(items::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    ## TODO Add this to logs
    if sum(probs) != 1
        println("$items => Not normalized probabilities => automatically normalized")
    end
    cat = Categorical(probs ./ sum(probs))
    map = CategoricalDiscretizer(items)
    NamedCategorical{N}(items, probs, cat, map)
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
    prob_dict::Vector{ProbabilityDictionary}
end
## Creating prob_dict for RootCPD
function RootCPD(target::NodeName, distributions::Distribution)
    if isa(distributions, NamedCategorical)
        prob_dict = [ProbabilityDictionary((nothing, Dict(distributions.items .=> distributions.probs / sum(distributions.probs))))]
    else
        prob_dict = [ProbabilityDictionary((nothing, Dict(:all_states => distributions)))]
    end
    RootCPD(target, NodeName[], distributions, prob_dict)
end

name(cpd::RootCPD) = cpd.target
parents(cpd::RootCPD) = cpd.parents

"""
A categorical CPD, P(x|parents(x)) where all parents are discrete integers 1:N 
and distributions ∀ combinations are known.
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
    prob_dict::Vector{ProbabilityDictionary}
    ## Checks:
    #    - number of parental_ncategories - number of parents coherence
    #    - number of distributions - number of parents_ncategories coherence
    #    - no check on parental_ncategories - prob_dict cause prob dict is automatically generated
    function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}, prob_dict) where {D<:Distribution}
        length(parental_ncategories) == length(parents) ? new(target, parents, parental_ncategories, distributions, prob_dict) : throw(DomainError(target, "length of parental_ncategories is different from the number of defined parents"))
        prod(parental_ncategories) == length(distributions) ? new(target, parents, parental_ncategories, distributions, prob_dict) : throw(DomainError(target, "number of parental_ncategories is different from the number of  defined functions"))
    end

    ## Creating prob_dict for CategoricalCPD
    function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}) where {D<:Distribution}
        f = x -> collect(1:1:x)
        fn = x -> Dict(x.items .=> x.probs / sum(x.probs))
        fd = x -> Dict(:all_states .=> x)
        combinations = sort(vec(collect(Iterators.product(f.(parental_ncategories)...))))
        evidences_vector = map_combination2evidence.(combinations, repeat([parents], length(combinations)))
        if isa(distributions, Vector{NamedCategorical{Symbol}})
            prob_dict = ProbabilityDictionary.(tuple.(evidences_vector, fn.(distributions)))
        else
            prob_dict = ProbabilityDictionary.(tuple.(evidences_vector, fd.(distributions)))
        end
        new(target, parents, parental_ncategories, distributions, prob_dict)
    end
end

function map_combination2evidence(combination::Tuple, nodes::NodeNames)
    evidence_dict = Dict()
    if length(combination) != length(nodes)
        throw(DomainError([combination, nodes], "Assigned parents are not equals to the number of parental_ncategories"))
    else
        for i in range(1, length(combination))
            evidence_dict[nodes[i]] = combination[i]
        end
    end
    return evidence_dict
end

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parentsz


"""
A Parent Functional CPD to be used when there at least a continuos parents and/or distribution are not known.
"""

struct FunctionalCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    prob_dict::Vector{ProbabilityDictionaryFunctional}
    ## Check:
    #    - parental_ncategories - prob_dict coherence
    function FunctionalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int64}, prob_dict::Vector{ProbabilityDictionaryFunctional})
        prod(parental_ncategories) == length(prob_dict) ? new(target, parents, parental_ncategories, prob_dict) : throw(DomainError(target, "number of parental_ncategories is different from the number of  defined functions"))
    end
end

name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
