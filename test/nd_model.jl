module TestIdealPoints
    using Base.Test

    import RollCallDataIO
    import IdealPoints
    import FiniteDiff

    io = open(joinpath("data", "senate", "112.ord"), "r")
    ord_file = read(RollCallDataIO.ORDFile, io)
    close(io)

    roll_call = convert(
        RollCallDataIO.SparseRollCall,
        RollCallDataIO.roll_call(ord_file),
    )

    n_legislators, n_bills = roll_call.n_legislators, roll_call.n_bills

    for d in 1:2
        a, b, c, misc = IdealPoints.ideal_points(ord_file, d)

        θ = fill(
            0.0,
            d * n_legislators + d * n_bills + n_bills
        )

        nlp = IdealPoints.make_nlp(
            d,
            1.0,
            n_legislators,
            n_bills,
            roll_call.legislators,
            roll_call.bills,
            roll_call.votes,
        )

        nlp_gr! = IdealPoints.make_nlp_gr(
            d,
            1.0,
            n_legislators,
            n_bills,
            roll_call.legislators,
            roll_call.bills,
            roll_call.votes,
        )

        θstar = vcat(a, b, c)
        @test nlp(θstar) < nlp(θ)

        gr = copy(θ)
        nlp_gr!(θ, gr)
        @test norm(FiniteDiff.gradient(nlp, θ) - gr) < 1e-4

        nlp_gr!(θstar, gr)
        @test norm(FiniteDiff.gradient(nlp, θstar) - gr) < 1e-4
        @test norm(gr) < 1e-4

        # Check in a region around θstar that θstar produces a smaller nlp.
        for _ in 1:10
            for i in 1:length(θ)
                θ[i] = θstar[i] + 0.01 * randn()
            end
            @test nlp(θstar) < nlp(θ)

            nlp_gr!(θ, gr)
            n1 = norm(gr)

            nlp_gr!(θstar, gr)
            n2 = norm(gr)
            @test n2 < n1
        end
    end
end
