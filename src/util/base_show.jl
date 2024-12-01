function Base.show(io::IO, obj::AbstractNode)
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::AbstractNode)
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end

function print_object(io::IO, obj::DiscreteNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        _is_root(obj) ? r = "Root" : r = "Child"
        _is_precise(obj) ? tp = "Precise" : tp = "Imprecise"
        print(io, "\n  ")
        print(io, "nature: $r $tp ")
        print(io, "\n  ")
        st = _states(obj)
        print(io, "\r  states: $st")
        print(io, "\n  ")
        isempty(obj.parameters) ? param = DataFrame() : param = DataFrame(obj.parameters)
        # print(io, "parameters: $param")
        print(io, "parameters:")
        for (k, v) in obj.parameters
            print(io, "\n $k => $v")
        end
        print(io, "\n  ")
        print(io, "\r  CPT: $(obj.cpt)")
    else
        Base.show_default(io, obj)
    end
end

function print_object(io::IO, obj::ContinuousNode; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        print(io, "name: $(obj.name)")
        _is_root(obj) ? r = "Root" : r = "Child"
        _is_precise(obj) ? tp = "Precise" : tp = "Imprecise"
        print(io, "\n  ")
        print(io, "nature: $r $tp")
        print(io, "\n  ")
        isempty(obj.discretization.intervals) ? disc = nothing : disc = obj.discretization
        print(io, "discretization: $disc")
        print(io, "\n  ")
        print("CPT => $(obj.cpt)")
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
                _is_precise(n) ? "Precise" : "Imprecise"
            end
        end
        function l(n)
            if _is_root(n)
                return "Root"
            else
                return "Child"
            end
        end
        nodes = map(i -> (i.name, typeof(i), l(i), f(i)), obj)
        print(io, "nodes:\n")
        for i in nodes
            print(io, "$i \n")
        end
    else
        Base.show_default(io, obj)
    end
end

function Base.show(io::IO, obj::EnhancedBayesianNetwork)
    print_object(io, obj, multiline=false)
end

function Base.show(io::IO, mime::MIME"text/plain", obj::EnhancedBayesianNetwork)
    multiline = get(io, :multiline, true)
    print_object(io, obj, multiline=multiline)
end

function print_object(io::IO, obj::EnhancedBayesianNetwork; multiline::Bool)
    if multiline
        print(io, summary(obj))
        print(io, "\n  ")
        function f(n)
            if isa(n, FunctionalNode)
                return "Functional"
            else
                _is_precise(n) ? "Precise" : "Imprecise"
            end
        end
        function l(n)
            if _is_root(n)
                return "Root"
            else
                return "Child"
            end
        end
        nodes = map(i -> (i.name, typeof(i), l(i), f(i), parents(obj, i)[2]), obj.nodes)
        print(io, "nodes:\n")
        for i in nodes
            print(io, "$i \n")
        end
    else
        Base.show_default(io, obj)
    end
    print(io, "\n  ")
    print(io, "adj_matrix:\n")
    display(obj.adj_matrix)
    print(io, "\n  ")
    print(io, "topology_dict: \n ")
    display(obj.topology_dict)
end

# # function Base.show(io::IO, obj::Factor)
# #     print_object(io, obj, multiline=false)
# # end

# # function Base.show(io::IO, mime::MIME"text/plain", obj::Factor)
# #     multiline = get(io, :multiline, true)
# #     print_object(io, obj, multiline=multiline)
# # end


# # function print_object(io::IO, obj::Factor; multiline::Bool)
# #     if multiline
# #         print(io, summary(obj))
# #         print(io, "\n  ")
# #         print(io, "dimensions:")
# #         print(io, "\n  ")
# #         for (i, name) in enumerate(obj.dimensions)
# #             print(io, "\r $i => $name \n")
# #         end
# #         print(io, "\r \r  mapping:")
# #         for i in keys(obj.states_mapping)
# #             single = (i, obj.states_mapping[i])
# #             print(io, "\n $single")
# #         end
# #         print(io, "\n  ")
# #         print(io, "\n  ")
# #         print(io, "potentials:")
# #         print(io, "\n  ")
# #         display(obj.potential)
# #     else
# #         Base.show_default(io, obj)
# #     end
# # end