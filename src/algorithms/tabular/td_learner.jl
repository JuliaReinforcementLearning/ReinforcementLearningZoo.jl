export TDLearner

using LinearAlgebra:dot
using Distributions:pdf

Base.@kwdef struct TDLearner{A} <: AbstractLearner
    approximator::A
    γ::Float64
    method::Symbol
    n::Int
end

(L::TDLearner)(env::AbstractEnv) = L.approximator(state(env))
(L::TDLearner)(s) = L.approximator(s)
(L::TDLearner)(s, a) = L.approximator(s, a)

## update policies

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage
)
    if p.learner.method === :ExpectedSARSA && s === PRE_ACT_STAGE
        # A special case
        update!(p.learner, (t, pdf(prob(p, e))), e, s)
    else
        update!(p.learner, t, e, s)
    end
end


function RLBase.update!(L::TDLearner, t::AbstractTrajectory, ::AbstractEnv, s::Union{PreActStage, PostEpisodeStage})
    _update!(L, L.approximator, Val(L.method), t, s)
end

# for ExpectedSARSA
function RLBase.update!(L::TDLearner, t::Tuple, ::AbstractEnv, s::Union{PreActStage, PostEpisodeStage})
    _update!(L, L.approximator, Val(L.method), t, s)
end

## update trajectories

function RLBase.update!(
    t::AbstractTrajectory,
    ::Union{QBasedPolicy{<:TDLearner}, NamedPolicy{<:QBasedPolicy{<:TDLearner}}},
    ::AbstractEnv,
    ::PreEpisodeStage
)
    empty!(t)
end

## implementations

function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Union{Val{:SARSA}, Val{:ExpectedSARSA}, Val{:SARS}},
    t::Trajectory,
    ::PostEpisodeStage
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.
    for i in 1:min(n+1, length(R))
        G = R[end-i+1] + γ * G
        s, a = S[end-i], A[end-i]
        update!(Q, (s,a) => Q(s, a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:SARSA},
    t::Trajectory,
    ::PreActStage
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n+1
        s, a, s′, a′ = S[end-n-1], A[end-n-1], S[end], A[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n+1) * Q(s′, a′)
        update!(Q, (s,a) => Q(s ,a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:ExpectedSARSA},
    experience,
    ::PreActStage
)
    t, p = experience

    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n+1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n+1) * dot(Q(s′), p)
        update!(Q, (s,a) => Q(s ,a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:SARS},
    t::AbstractTrajectory,
    ::PreActStage
)
    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n+1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n+1) * maximum(Q(s′))
        update!(Q, (s,a) => Q(s ,a) - G)
    end
end
