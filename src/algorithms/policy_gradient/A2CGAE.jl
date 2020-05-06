export A2CGAELearner

using Flux

"""
    A2CGAELearner(;kwargs...)
# Keyword arguments
- `approximator`, an [`ActorCritic`](@ref) based [`NeuralNetworkApproximator`](@ref)
- `γ::Float32`, reward discount rate.
- 'λ::Float32', lambda for GAE-lambda
- `actor_loss_weight::Float32`
- `critic_loss_weight::Float32`
- `entropy_loss_weight::Float32`
"""
Base.@kwdef struct A2CGAELearner{A} <: AbstractLearner
    approximator::A
    γ::Float32
    λ::Float32
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
end

(learner::A2CGAELearner)(obs::BatchObs) =
    learner.approximator(
        send_to_device(device(learner.approximator), get_state(obs)),
        Val(:Q),
    ) |> send_to_host

function RLBase.update!(learner::A2CGAELearner, experience)
    AC = learner.approximator
    γ = learner.γ
    λ = learner.λ
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    states, actions, rewards, terminals, rollout = experience
    states = send_to_device(device(AC), states)
    rollout = flatten_batch(rollout)
    rollout = send_to_device(device(AC), rollout)

    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_step)
    actions = flatten_batch(actions)
    actions = CartesianIndex.(actions, 1:length(actions))

    rollout_values = AC(rollout, Val(:V))
    rollout_values = send_to_host(rollout_values)
    rollout_values = reshape(
        rollout_values,
        size(states, ndims(states) - 1),
        size(states, ndims(states)) + 1,
    )
    advantages = generalized_advantage_estimation(
        rewards,
        rollout_values,
        γ,
        λ;
        dims = 2,
        terminal = terminals,
    )

    gains = advantages + select_last_dim(rollout_values, 1:(nframes(rollout_values)-1))
    gains = send_to_device(device(AC), gains)
    advantages = flatten_batch(advantages)
    advantages = send_to_device(device(AC), advantages)

    gs = gradient(Flux.params(AC)) do
        probs = AC(states_flattened, Val(:Q))
        log_probs = log.(probs)
        log_probs_select = log_probs[actions]
        values = AC(states_flattened, Val(:V))
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* advantages)
        critic_loss = mean(advantage .^ 2)
        entropy_loss = sum(probs .* log_probs) * 1 // size(probs, 2)
        loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
        loss
    end
    update!(AC, gs)
end

function RLCore.extract_experience(
    t::CircularCompactSARTSATrajectory,
    learner::A2CGAELearner,
)
    if isfull(t)
        (
            states = get_trace(t, :state),
            actions = get_trace(t, :action),
            rewards = get_trace(t, :reward),
            terminals = get_trace(t, :terminal),
            rollout = t[:state],
        )
    else
        nothing
    end
end

function (agent::Agent{<:QBasedPolicy{<:A2CGAELearner},<:CircularCompactSARTSATrajectory})(
    ::PreActStage,
    obs,
)
    action = agent.policy(obs)
    state = get_state(obs)
    push!(agent.trajectory; state = state, action = action)
    update!(agent.policy, agent.trajectory)

    # the main difference is we'd like to flush the buffer after each update!
    if isfull(agent.trajectory)
        empty!(agent.trajectory)
        push!(agent.trajectory; state = state, action = action)
    end

    action
end
