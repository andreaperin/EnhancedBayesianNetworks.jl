function Base.show(io::IO, obj::AbstractNode)
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::AbstractNode)
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end

function print_object(io::IO, obj::DiscreteRootNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        EnhancedBayesianNetworks._is_imprecise(obj) ? tp = "Imprecise" : tp = "Precise"
        print(io, "\n  ")
        print(io, "nature: $tp")
        print(io, "\n  ")
        isempty(obj.additional_info) ? info = nothing : info = obj.additional_info
        print(io, "info: $info")
        print(io, "\n  ")
        isempty(obj.parameters) ? param = nothing : param = obj.parameters
        print(io, "parameters: $param")
        print(io, "\n  ")
        d = DataFrame([i => [obj.states[i]] for i in collect(keys(obj.states))])
        # table = [permutedims(collect(keys(obj.states))); permutedims(Matrix(hcat(collect(values(obj.states)))))]
        print(io, "\r  states: $d")
    else
        Base.show_default(io, obj)
    end
end

function print_object(io::IO, obj::ContinuousRootNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        EnhancedBayesianNetworks._is_imprecise(obj) ? tp = "Imprecise" : tp = "Precise"
        print(io, "\n  ")
        print(io, "nature: $tp")
        print(io, "\n  ")
        isempty(obj.discretization.intervals) ? disc = nothing : disc = obj.discretization
        print(io, "discretization: $disc")
        print(io, "\n  ")
        isempty(obj.additional_info) ? info = nothing : info = obj.additional_info
        print(io, "info: $info")
        print(io, "\n  ")
        # table = [permutedims(collect(keys(obj.states))); permutedims(Matrix(hcat(collect(values(obj.states)))))]
        print(io, "\r  distribution: $(obj.distribution)")
    else
        Base.show_default(io, obj)
    end
end


function print_object(io::IO, obj::DiscreteChildNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        EnhancedBayesianNetworks._is_imprecise(obj) ? tp = "Imprecise" : tp = "Precise"
        print(io, "\n  ")
        print(io, "nature: $tp")
        print(io, "\n  ")
        parent_names = [i.name for i in obj.parents]
        print(io, "parents: $parent_names")
        print(io, "\n  ")
        isempty(obj.additional_info) ? info = nothing : info = obj.additional_info
        print(io, "info: $info")
        print(io, "\n  ")
        isempty(obj.parameters) ? param = nothing : param = obj.parameters
        print(io, "parameters: $param")
        print(io, "\n  ")
        print(io, "states:")
        for scenario in keys(obj.states)
            d = DataFrame([i => [obj.states[scenario][i]] for i in collect(keys(obj.states[scenario]))])
            print(io, "\n $scenario : $d")
        end
    else
        Base.show_default(io, obj)
    end
end

function print_object(io::IO, obj::ContinuousChildNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        EnhancedBayesianNetworks._is_imprecise(obj) ? tp = "Imprecise" : tp = "Precise"
        print(io, "\n  ")
        print(io, "nature: $tp")
        print(io, "\n  ")
        parent_names = [i.name for i in obj.parents]
        print(io, "parents: $parent_names")
        print(io, "\n  ")
        isempty(obj.additional_info) ? info = nothing : info = obj.additional_info
        print(io, "info: $info")
        print(io, "\n  ")
        isempty(obj.discretization.intervals) ? disc = nothing : disc = obj.discretization
        print(io, "discretization: $disc")
        print(io, "\n  ")
        print(io, "distribution:")
        for scenario in keys(obj.distribution)
            d = [obj.distribution[scenario]]
            print(io, "\n $scenario : $d")
        end
    else
        Base.show_default(io, obj)
    end
end

function print_object(io::IO, obj::DiscreteFunctionalNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        print(io, "\n  ")
        parent_names = [i.name for i in obj.parents]
        print(io, "parents: $parent_names")
        print(io, "\n  ")
        isempty(obj.parameters) ? param = nothing : param = obj.parameters
        print(io, "parameters: $param")
        print(io, "\n  ")
        print(io, "simulation: $(obj.simulation)")
        print(io, "\n  ")
        print(io, "models: $(obj.models)")
        print(io, "\n  ")
        print(io, "performance: $(obj.performance)")
    else
        Base.show_default(io, obj)
    end
end

function print_object(io::IO, obj::ContinuousFunctionalNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        print(io, "\n  ")
        parent_names = [i.name for i in obj.parents]
        print(io, "parents: $parent_names")
        print(io, "\n  ")
        isempty(obj.discretization.intervals) ? disc = nothing : disc = obj.discretization
        print(io, "discretization: $disc")
        print(io, "\n  ")
        print(io, "simulation: $(obj.simulation)")
        print(io, "\n  ")
        print(io, "models: $(obj.models)")
        print(io, "\n  ")

    else
        Base.show_default(io, obj)
    end
end



function Base.show(io::IO, obj::AbstractVector{<:AbstractNode})
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::AbstractVector{<:AbstractNode})
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end

function print_object(io::IO, obj::AbstractVector{<:AbstractNode}; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        function f(n)
            if isa(n, FunctionalNode)
                return "Functional"
            else
                EnhancedBayesianNetworks._is_imprecise(n) ? "Imprecise" : "Precise"
            end
        end
        nodes = map(i -> (i.name, typeof(i), f(i)), obj)
        print(io, "nodes:\n")
        for i in nodes
            print(io, "$i \n")
        end
    else
        Base.show_default(io, obj)
    end
end

function Base.show(io::IO, obj::AbstractNetwork)
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::AbstractNetwork)
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end

function print_object(io::IO, obj::AbstractNetwork; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        function f(n)
            if isa(n, FunctionalNode)
                return "Functional"
            else
                EnhancedBayesianNetworks._is_imprecise(n) ? "Imprecise" : "Precise"
            end
        end
        nodes = map(i -> (i.name, obj.name_to_index[i.name], typeof(i), f(i)), obj.nodes)
        print(io, "nodes:\n")
        for i in nodes
            print(io, "$i \n")
        end
    else
        Base.show_default(io, obj)
    end
end


function Base.show(io::IO, obj::Factor)
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::Factor)
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end


function print_object(io::IO, obj::Factor; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "dimensions:")
        print(io, "\n  ")
        for (i, name) in enumerate(obj.dimensions)
            print(io, "\r $i => $name \n")
        end
        print(io, "\r \r  mapping:")
        for i in keys(obj.states_mapping)
            single = (i, obj.states_mapping[i])
            print(io, "\n $single")
        end
        print(io, "\n  ")
        print(io, "\n  ")
        print(io, "potentials:")
        print(io, "\n  ")
        display(obj.potential)
    else
        Base.show_default(io, obj)
    end
end