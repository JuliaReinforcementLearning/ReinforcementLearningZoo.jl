export A2CGAELearner

using Flux

"""
    A2CGAELearner(;kwargs...)

# Keyword arguments

- `approximator`, an [`ActorCritic`](@ref) based [`NeuralNetworkApproximator`](@ref)
- `γ::Float32`, reward discount rate.
- 'λ::Float32', lambda for GAE-Lamda
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
    states, actions, rewards, terminals, rollout, next_state = experience
    states = send_to_device(device(AC), states)
    rollout = reshape(rollout, : , size(rollout,2)*size(rollout,3))
    rollout = send_to_device(device(AC), rollout)
    next_state = send_to_device(device(AC), next_state)

    states_flattened = flatten_batch(states) # (state_size..., n_thread * update_step)
    actions = flatten_batch(actions)

    actions = CartesianIndex.(actions, 1:length(actions))

    rollout_values = AC(rollout, Val(:V))
    rollout_values = send_to_host(rollout_values)
    rollout_values = reshape(rollout_values,size(states,2),size(states,3)+1)

    advantages = generalized_advantage_estimation(
        rewards,
        rollout_values,
        γ,
        λ;
        dims = 2,
        terminal = terminals,
    )

    advantages = reshape(advantages,size(states,2)*size(states,3))
    advantages = send_to_device(device(AC), advantages)

    next_state_values = AC(next_state, Val(:V))
    gains = discount_rewards(
        rewards,
        γ;
        dims = 2,
        init = send_to_host(next_state_values),
        terminal = terminals,
    )
    gains = send_to_device(device(AC), gains)

    gs = gradient(Flux.params(AC)) do
        probs = AC(states_flattened, Val(:Q))
        log_probs = log.(probs)
        log_probs_select = probs[actions]
        values = AC(states_flattened, Val(:V))
        advantage = vec(gains) .- vec(values)
        actor_loss = -mean(log_probs_select .* Zygote.dropgrad(advantages))
        critic_loss = mean(advantage .^ 2)
        entropy_loss = sum(probs .* log_probs) * 1 // size(probs, 2)
        loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
        loss
    end
    update!(AC, gs)
end

function RLBase.extract_experience(t::CircularCompactSARTSATrajectory, learner::A2CGAELearner)
    if isfull(t)
        (
            states = get_trace(t, :state),
            actions = get_trace(t, :action),
            rewards = get_trace(t, :reward),
            terminals = t[:terminal],
            rollout = t[:state],
            next_state = select_last_frame(get_trace(t, :next_state)),
        )
    else
        nothing
    end
end
