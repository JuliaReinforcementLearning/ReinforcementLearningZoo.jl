using Distributions: Categorical, Normal
using Flux
using Random

using ReinforcementLearningBase
using ReinforcementLearningCore

export PGPolicy
Base.@kwdef mutable struct PGPolicy{A<:NeuralNetworkApproximator,R<:AbstractRNG} <:
                           AbstractPolicy
    approximator::A
    dist::Any = Categorical
    γ::Float32 = 0.99f0 # discount factor
    mini_batches::Int = 128
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
end

function (agent::Agent{<:PGPolicy,<:AbstractTrajectory})(::Training{PreActStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (agent::Agent{<:PGPolicy,<:AbstractTrajectory})(::Training{PostEpisodeStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (π::PGPolicy)(env)
    to_dev = x -> send_to_device(device(π.approximator), x)

    logits = env |> get_state |> to_dev |> π.approximator
    π(logits, get_actions(env))
end

function (π::PGPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function (π::PGPolicy)(logits, actions::DiscreteSpace)
    dist = logits |> softmax |> send_to_host |> π.dist
    action = actions[rand(π.rng, dist)]
end

"""
See Diagonal Gaussian Policies from https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
"""
function (π::PGPolicy)(logits, actions::ContinuousSpace)
    dist = logits |> send_to_host |> x -> π.dist(x[1], exp(x[2]))
    action = clamp(rand(π.rng, dist), actions.low, actions.high)
end

function (π::PGPolicy)(logits, actions::MultiDiscreteSpace)
    error("not implemented")
    # TODO
end

function (π::PGPolicy)(logits, actions::MultiContinuousSpace)
    error("not implemented")
    # TODO
end

function RLBase.update!(π::PGPolicy, traj::ElasticCompactSARTSATrajectory)
    error("not supported")
end

function RLBase.update!(π::PGPolicy, traj::ElasticCompactSARTSATrajectory, ::DiscreteSpace)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev = x -> send_to_device(device(model), x)

    states = traj[:state] |> to_dev
    actions = traj[:action]
    gains =
        traj[:reward] |>
        x -> discount_rewards(x, π.γ) |> to_dev |> x -> Flux.normalise(x; dims = 1)

    # TODO: use mini batches.
    gs = gradient(Flux.params(model)) do
        log_prob = states |> model |> logsoftmax
        log_probₐ = log_prob[CartesianIndex.(actions, 1:length(actions))]
        loss = -mean(log_probₐ .* gains)
        ignore() do
            π.loss = loss
        end
        loss
    end
    update!(model, gs)
    empty!(traj)
end

function RLBase.update!(π::PGPolicy, traj::ElasticCompactSARTSATrajectory, ::ContinuousSpace)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev = x -> send_to_device(device(model), x)

    states = traj[:state] |> to_dev
    actions = traj[:action]
    gains =
        traj[:reward] |>
        x -> discount_rewards(x, π.γ) |> to_dev |> x -> Flux.normalise(x; dims = 1)

    # TODO: fix bug
    gs = gradient(Flux.params(model)) do
        logits = states |> model
        @views μ, σ = logits[1, :], exp.(logits[2, :])
        dist = π.dist.(μ, σ)
        log_probₐ = logpdf.(dist, actions)
        loss = -mean(log_probₐ .* gains)
        ignore() do
            π.loss = loss
        end
        loss
    end

    update!(model, gs)
    empty!(traj)
end
