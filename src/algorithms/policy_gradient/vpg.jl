using Distributions: Categorical, Normal
using Flux
using Random

using ReinforcementLearningBase
using ReinforcementLearningCore

export VPGPolicy

"""
Vanilla Policy Gradient
"""
Base.@kwdef mutable struct VPGPolicy{
    A<:NeuralNetworkApproximator,
    B<:Union{NeuralNetworkApproximator,Nothing},
    R<:AbstractRNG,
} <: AbstractPolicy
    approximator::A
    baseline::B = nothing
    dist::Any = Categorical
    γ::Float32 = 0.99f0 # discount factor
    α = 1.0f0 # step size
    fα = 0.999f0
    mini_batches::Int = 128
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
    baseline_loss::Float32 = 0.0f0
end

function (agent::Agent{<:VPGPolicy,<:AbstractTrajectory})(::Training{PreActStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (agent::Agent{<:VPGPolicy,<:AbstractTrajectory})(::Training{PostEpisodeStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (π::VPGPolicy)(env)
    to_dev = x -> send_to_device(device(π.approximator), x)

    logits = env |> get_state |> to_dev |> π.approximator
    π(logits, get_actions(env))
end

function (π::VPGPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function (π::VPGPolicy)(logits, actions::DiscreteSpace)
    dist = logits |> softmax |> π.dist
    action = actions[rand(π.rng, dist)]
end

"""
See
* [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
* [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564.pdf)
"""
function (π::VPGPolicy)(logits, actions::ContinuousSpace)
    dist = π.dist(logits[1], exp(logits[2]))
    action = clamp(rand(π.rng, dist), actions.low, actions.high)
end

function (π::VPGPolicy)(logits, actions::MultiDiscreteSpace)
    error("not implemented")
    # TODO
end

function (π::VPGPolicy)(logits, actions::MultiContinuousSpace)
    error("not implemented")
    # TODO
end

function RLBase.update!(π::VPGPolicy, traj::ElasticCompactSARTSATrajectory)
    error("not supported")
end

function RLBase.update!(π::VPGPolicy, traj::ElasticCompactSARTSATrajectory, ::DiscreteSpace)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev = x -> send_to_device(device(model), x)

    states = traj[:state] |> to_dev
    actions = traj[:action]
    gains = traj[:reward] # |> x -> discount_rewards(x, π.γ)

    if typeof(π.baseline) != Nothing
        baseline = π.baseline(states)
        gs = gradient(Flux.params(π.baseline)) do
            loss = mse(baseline, gains')
            ignore() do
                π.baseline_loss = loss
            end
            loss
        end
        update!(π.baseline, gs)
        gains -= baseline[1, :]
    end
    gains =
        gains |> x -> discount_rewards(x, π.γ) |> x -> Flux.normalise(x; dims = 1) |> to_dev

    # TODO: use mini batches.
    gs = gradient(Flux.params(model)) do
        log_prob = states |> model |> logsoftmax
        log_probₐ = log_prob[CartesianIndex.(actions, 1:length(actions))]
        loss = -mean(log_probₐ .* gains) * π.α
        ignore() do
            π.loss = loss
        end
        loss
    end

    update!(model, gs)
    empty!(traj)
    π.α *= π.fα # decrease α
end

function RLBase.update!(
    π::VPGPolicy,
    traj::ElasticCompactSARTSATrajectory,
    ::ContinuousSpace,
)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev = x -> send_to_device(device(model), x)

    states = traj[:state] |> to_dev
    actions = traj[:action]
    gains =
        traj[:reward] |>
        x -> discount_rewards(x, π.γ) |> x -> Flux.normalise(x; dims = 1) |> to_dev

    # TODO: fix bug
    gs = gradient(Flux.params(model)) do
        logits = states |> model
        @views μ, σ = logits[1, :], exp.(logits[2, :])
        dist = π.dist.(μ, σ)
        log_probₐ = logpdf.(dist, actions)
        loss = -mean(log_probₐ .* gains) * π.α
        ignore() do
            π.loss = loss
        end
        loss
    end

    update!(model, gs)
    empty!(traj)
    π.α *= π.fα # decrease α

end
