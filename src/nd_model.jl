@doc """
Generate a closure over the roll-call data set that will compute the
unnormalized negative log-posterior at any set of parameters, `θ`.
""" ->
function make_nlp(
    D::Int,
    σ_x::Float64,
    σ_y::Float64,
    n_legislators::Int,
    n_bills::Int,
    legislators::Vector{Int},
    bills::Vector{Int},
    votes::Vector{Float64},
)
    function nlp(θ::Vector)
        n = length(legislators)

        lp = 0.0

        # Log-posterior contribution from the likelihood.
        for i in 1:n
            l = legislators[i]
            b = bills[i]
            v = votes[i]
            z = 0.0
            for d in 1:D
                x_l_d = θ[D * (l - 1) + d]
                x_b_d = θ[D * n_legislators + D * (b - 1) + d]
                z += x_l_d * x_b_d
            end
            y_b = θ[D * n_legislators + D * n_bills + b]
            z += y_b
            p = logistic(z)
            lp += log(v * p + (1 - v) * (1 - p))
        end

        # Log-posterior contribution from the prior.
        for i in 1:(D * n_legislators + D * n_bills)
            lp += logpdf(Normal(0, σ_x), θ[i])
        end
        for i in (D * n_legislators + D * n_bills + 1):(D * n_legislators + D * n_bills + n_bills)
            lp += logpdf(Normal(0, σ_y), θ[i])
        end

        -lp
    end

    nlp
end

@doc """
Generate a closure over the roll-call data set that will compute the gradient
of the unnormalized negative log-posterior at any set of parameters, `θ`. The
gradient will be stored in the second argument, `gr`, which will be mutated.
""" ->
function make_nlp_gr(
    D::Int,
    σ_x::Float64,
    σ_y::Float64,
    n_legislators::Int,
    n_bills::Int,
    legislators::Vector{Int},
    bills::Vector{Int},
    votes::Vector{Float64},
)
    function nlp_gr!(θ::Vector, gr::Vector)
        n = length(legislators)

        fill!(gr, 0.0)

        # Log-posterior contribution from the likelihood.
        for i in 1:n
            l = legislators[i]
            b = bills[i]
            v = votes[i]
            z = 0.0
            for d in 1:D
                x_l_d = θ[D * (l - 1) + d]
                x_b_d = θ[D * n_legislators + D * (b - 1) + d]
                z += x_l_d * x_b_d
            end
            y_b = θ[D * n_legislators + D * n_bills + b]
            z += y_b
            p = logistic(z)
            for d in 1:D
                x_l_d = θ[D * (l - 1) + d]
                x_b_d = θ[D * n_legislators + D * (b - 1) + d]
                gr[D * (l - 1) + d] += (p - v) * x_b_d
                gr[D * n_legislators + D * (b - 1) + d] += (p - v) * x_l_d
            end
            gr[D * n_legislators + D * n_bills + b] += (p - v)
        end

        # Log-posterior contribution from the prior.
        for i in 1:(D * n_legislators + D * n_bills)
            gr[i] += θ[i] / σ_x^2
        end
        for i in (D * n_legislators + D * n_bills + 1):(D * n_legislators + D * n_bills + n_bills)
            gr[i] += θ[i] / σ_y^2
        end

        nothing
    end

    nlp_gr!
end

@doc """
Fit an ideal points model to roll call data from an ORD file. Defaults to
fitting latent space positions in 1 dimension; this can changed by increasing
the value of the second argument. To ensure that left-wing candidates take on
negative valued latent space positions, all legislators from a specific party
idea are not in the non-negative orthant. For long-running model fits, it is
possible to enable a trace of the optimization process to ensure that the
model fit is improving with each optimization pass.
""" ->
function ideal_points(
    ord_file::ORDFile,
    D::Integer = 1,
    σ_x::Real = 1.0,
    σ_y::Real = 25.0,
    party_id::Integer = 100,
    show_progress::Bool = false,
)
    parties = RollCallDataIO.parties(ord_file)
    roll_call = convert(SparseRollCall, RollCallDataIO.roll_call(ord_file))
    n_legislators, n_bills = roll_call.n_legislators, roll_call.n_bills

    nlp = make_nlp(
        D,
        σ_x,
        σ_y,
        n_legislators,
        n_bills,
        roll_call.legislators,
        roll_call.bills,
        roll_call.votes,
    )

    nlp_gr! = make_nlp_gr(
        D,
        σ_x,
        σ_y,
        n_legislators,
        n_bills,
        roll_call.legislators,
        roll_call.bills,
        roll_call.votes,
    )

    # Force members of party with ID == party_id to one part of the latent
    # space to ensure consistent placement across runs. The default party_id
    # will always place US Democrats to the left for consistency with
    # left-wing terminology.
    Θ₀ = fill(0.0, D * n_legislators + D * n_bills + n_bills)
    for l in 1:n_legislators
        for d in 1:D
            Θ₀[D * (l - 1) + d] = ifelse(
                parties[l] == party_id,
                -1.0 + d / l,
                1.0 + d / l,
            )
        end
    end

    fit = Optim.optimize(
        nlp,
        nlp_gr!,
        Θ₀,
        method = :l_bfgs,
        show_trace = show_progress,
    )

    legislator_slopes = fit.minimum[1:(D * n_legislators)]
    bill_slopes = fit.minimum[
        (D * n_legislators + 1):(D * n_legislators + D * n_bills)
    ]
    bill_intercepts = fit.minimum[
        (D * n_legislators + D * n_bills + 1):(D * n_legislators + D * n_bills + n_bills)
    ]

    (
        legislator_slopes,
        bill_slopes,
        bill_intercepts,
        fit,
    )
end
