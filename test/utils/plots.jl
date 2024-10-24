@testset "Plots Auxiliary" begin
    pt1 = (0, 0)
    pt2 = (1, 1)
    midpoint = (0.5, 0.5)
    @test EnhancedBayesianNetworks.midpoint(pt1, pt2) == midpoint

    θ = π / 4
    endx = 1
    endy = 1
    arrowlength = 0.1
    @test EnhancedBayesianNetworks.arrowcoords(θ, endx, endy, arrowlength) == ((0.9577381738259301, 0.909369221296335), (0.909369221296335, 0.9577381738259301))

    a_p = DiscreteRootNode(:Ap, Dict(:a1 => 0.5, :a2 => 0.5))
    a_i = DiscreteRootNode(:Ai, Dict(:a1 => [0.2, 0.4], :a2 => [0.6, 0.8]))
    b_p = ContinuousRootNode(:Bp, Normal())
    b_i = ContinuousRootNode(:Bp, UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]))
    model = Model(df -> df.Bp, :C)
    performance = df -> df.C
    simulation = MonteCarlo(2)
    c = DiscreteFunctionalNode(:C, [b_p], [model], performance, simulation)
    d = ContinuousFunctionalNode(:D, [b_p], [model], simulation)

    @test EnhancedBayesianNetworks.node_color.([a_p, a_i, b_p, b_i, c, d]) == ["palegreen", "green1", "paleturquoise", "cyan1", "lightsalmon", "red1"]

    locs_x = [0.537392329950904, -1.0, 0.16878664898018236, 1.0, -0.48776316663034747, 0.16714316400701024, 0.7680868152581759, 0.4231089813619797]

    locs_y = [-1.0, 1.0, -0.4966322402213703, -0.7416815056488733, 0.6475700244569838, 0.22016758762297783, -0.16192563264423787, 0.7987250612004149]

    nodesize = fill(0.1, length(locs_x))

    edges_list = [(1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (6, 8)]

    lines_res = [[(0.4783112898528361, -0.9193189569915554), (0.22786768907825028, -0.5773132832298149)], [(0.6247025513524528, -0.9512462797438745), (0.9126897785984512, -0.7904352259049987)], [(-0.9176158714770725, 0.9433179449250668), (-0.570147295153275, 0.7042520795319169)], [(0.16855736870004417, -0.39663250306895004), (0.16737244428714843, 0.12016785047055757)], [(0.9628594364244364, -0.648834434950388), (0.8052273788337394, -0.2547727033427232)], [(-0.40401905592059145, 0.5929172667824425), (0.08339905329725421, 0.2748203452975191)], [(0.2515300074270633, 0.1665125724091187), (0.6836999718381229, -0.10827061743037872)], [(0.20760241080793113, 0.31161729701727925), (0.3826497345610589, 0.7072753518061135)]]

    @test EnhancedBayesianNetworks.graphline(edges_list, locs_x, locs_y, nodesize) == lines_res

    arrowlengthfrac = 0.1
    arrowangleoffset = π / 9

    lines_cord, arrows_cord = ([[(0.4783112898528361, -0.9193189569915554), (0.28338570648676104, -0.6531286639821609)], [(0.6247025513524528, -0.9512462797438745), (0.8306450078282319, -0.8362487370655403)], [(-0.9176158714770725, 0.9433179449250668), (-0.6475630527961478, 0.7575157884168122)], [(0.16855736870004417, -0.39663250306895004), (0.16758789727448603, 0.0261988353878978)], [(0.9628594364244364, -0.648834434950388), (0.8401280923575263, -0.34202041053967736)], [(-0.40401905592059145, 0.5929172667824425), (0.0047053304290183146, 0.32617713838988593)], [(0.2515300074270633, 0.1665125724091187), (0.6044022777848832, -0.057851395565759633)], [(0.20760241080793113, 0.31161729701727925), (0.3446304788996777, 0.6213407347152726)]], [[(0.31098024838017374, -0.6329217581799901), (0.22786768907825028, -0.5773132832298149), (0.25579116459334833, -0.6733355697843316)], [(0.8473197622178915, -0.8661105915030938), (0.9126897785984512, -0.7904352259049987), (0.8139702534385724, -0.8063868826279866)], [(-0.66694945739687, 0.7293387569716402), (-0.570147295153275, 0.7042520795319169), (-0.6281766481954257, 0.7856928198619841)], [(0.201789821707886, 0.026277253862172417), (0.16737244428714843, 0.12016785047055757), (0.13338597284108605, 0.026120416913623185)], [(0.8718836607851908, -0.32931758966236696), (0.8052273788337394, -0.2547727033427232), (0.8083725239298619, -0.3547232314169877)], [(-0.013987013583971322, 0.2975349656422545), (0.08339905329725421, 0.2748203452975191), (0.02339767444200795, 0.35481931113751736)], [(0.5860511817912981, -0.08671339584708693), (0.6836999718381229, -0.10827061743037872), (0.6227533737784683, -0.02898939528443234)], [(0.3759081216138074, 0.6075028573255731), (0.3826497345610589, 0.7072753518061135), (0.31335283618554793, 0.6351786121049723)]])

    @test EnhancedBayesianNetworks.graphline(edges_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset) == (lines_cord, arrows_cord)

    lines, larrows = EnhancedBayesianNetworks.build_straight_edges(edges_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset)

    function get_point_as_float(l)
        points = l.points
        map(v -> (v[1].value, v[2].value), points)
    end

    lines_points = [[(0.4783112898528361, -0.9193189569915554), (0.28338570648676104, -0.6531286639821609)], [(0.6247025513524528, -0.9512462797438745), (0.8306450078282319, -0.8362487370655403)], [(-0.9176158714770725, 0.9433179449250668), (-0.6475630527961478, 0.7575157884168122)], [(0.16855736870004417, -0.39663250306895004), (0.16758789727448603, 0.0261988353878978)], [(0.9628594364244364, -0.648834434950388), (0.8401280923575263, -0.34202041053967736)], [(-0.40401905592059145, 0.5929172667824425), (0.0047053304290183146, 0.32617713838988593)], [(0.2515300074270633, 0.1665125724091187), (0.6044022777848832, -0.057851395565759633)], [(0.20760241080793113, 0.31161729701727925), (0.3446304788996777, 0.6213407347152726)]]

    @test get_point_as_float.(lines.primitives) == lines_points

    larrows_points = [[(0.31098024838017374, -0.6329217581799901), (0.22786768907825028, -0.5773132832298149), (0.25579116459334833, -0.6733355697843316)], [(0.8473197622178915, -0.8661105915030938), (0.9126897785984512, -0.7904352259049987), (0.8139702534385724, -0.8063868826279866)], [(-0.66694945739687, 0.7293387569716402), (-0.570147295153275, 0.7042520795319169), (-0.6281766481954257, 0.7856928198619841)], [(0.201789821707886, 0.026277253862172417), (0.16737244428714843, 0.12016785047055757), (0.13338597284108605, 0.026120416913623185)], [(0.8718836607851908, -0.32931758966236696), (0.8052273788337394, -0.2547727033427232), (0.8083725239298619, -0.3547232314169877)], [(-0.013987013583971322, 0.2975349656422545), (0.08339905329725421, 0.2748203452975191), (0.02339767444200795, 0.35481931113751736)], [(0.5860511817912981, -0.08671339584708693), (0.6836999718381229, -0.10827061743037872), (0.6227533737784683, -0.02898939528443234)], [(0.3759081216138074, 0.6075028573255731), (0.3826497345610589, 0.7072753518061135), (0.31335283618554793, 0.6351786121049723)]]

    @test get_point_as_float.(larrows.primitives) == larrows_points

    arrowlengthfrac = 0.0
    lines, larrows = EnhancedBayesianNetworks.build_straight_edges(edges_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset)
    @test isnothing(larrows)
end