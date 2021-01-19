export TDLearner

using LinearAlgebra:dot
using Distributions:pdf

Base.@kwdef struct TDLearner{A} <: AbstractLearner
    approximator::A
    γ::Float64 = 1.0
    method::Symbol
    n::Int = 0
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
    ::Union{QBasedPolicy{<:TDLearner}, NamedPolicy{<:QBasedPolicy{<:TDLearner}}, VBasedPolicy{<:TDLearner}},
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

function _update!(
    L::TDLearner,
    ::TabularVApproximator,
    ::Val{:SRS},
    t::Trajectory,
    ::PostEpisodeStage
)
    S, R = t[:state], t[:reward]
    n, γ, V = L.n, L.γ, L.approximator
    G = 0.
    for i in 1:min(n+1, length(R))
        G = R[end-i+1] + γ * G
        s = S[end-i]
        update!(V, s => V(s) - G)
    end
end

function _update!(
    L::TDLearner,
    ::TabularVApproximator,
    ::Val{:SRS},
    t::AbstractTrajectory,
    ::PreActStage
)
    S = t[:state]
    R = t[:reward]

    n, γ, V = L.n, L.γ, L.approximator

    if length(R) >= n+1
        s, s′ = S[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n+1) * V(s′)
        update!(V, s => V(s) - G)
    end
end

#####
# DynaAgent
#####

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    m::Union{ExperienceBasedSamplingModel, TimeBasedSamplingModel},
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::Union{PreActStage, PostEpisodeStage}
)
    if p.learner.method == :SARS
        transition = sample(m)
        if !isnothing(transition)
            s, a, r, t, s′ = transition
            traj = VectorSARTTrajectory()
            push!(traj; state=s, action=a, reward=r, terminal=t)
            push!(traj; state=s′, action=a)  # here a is a dummy one
            update!(p.learner, traj, env, t ? POST_EPISODE_STAGE : PRE_ACT_STAGE)
        end
    else
        @error "unsupported method $(p.learner.method)"
    end
end

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    m::PrioritizedSweepingSamplingModel,
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::Union{PreActStage, PostEpisodeStage}
)
    if p.learner.method == :SARS
        transition = sample(m)
        if !isnothing(transition)
            s, a, r, t, s′ = transition
            traj = VectorSARTTrajectory()
            push!(traj; state=s, action=a, reward=r, terminal=t)
            push!(traj; state=s′, action=a)  # here a is a dummy one
            update!(p.learner, traj, env, t ? POST_EPISODE_STAGE : PRE_ACT_STAGE)
            
            # update priority
            for (s̄, ā, r̄, d̄) in m.predecessors[s]
                P = RLBase.priority(p.learner, (s̄, ā, r̄, d̄, s))
                if P ≥ m.θ
                    m.PQueue[(s̄, ā)] = P
                end
            end
        end
    else
        @error "unsupported method $(p.learner.method)"
    end
end

function RLBase.priority(L::TDLearner, transition::Tuple)
    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        Δ = d ? (r - Q(s, a)) : (r + γ^(L.n + 1) * maximum(Q(s′)) - Q(s, a))
        Δ = [Δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
        abs(Δ[])
    else
        @error "unsupported method"
    end
end