export Experiment

import .GridWorlds
using Dates
using ReinforcementLearningBase
using ReinforcementLearningCore
using Flux
using BSON
using TensorBoardLogger
using Logging
using Random

RLBase.get_state(w::GridWorlds.EmptyGridWorld) = Float64.(vec(GridWorlds.get_agent_view(w)))

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:EmptyGridWorld},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_BasicDQN_EmptyGridWorld_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = StableRNG(seed)

    inner_env = GridWorlds.EmptyGridWorld()

    integer_actions_wrapper = ActionTransformedEnv(a -> get_actions(inner_env)[a]; mapping = x -> Dict(action => i for (i, action) in enumerate([get_actions(inner_env)...]))[x])

    env = inner_env |> integer_actions_wrapper

    ns, na = length(get_state(env)), length(get_actions(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, 128, relu; initW = glorot_uniform(rng)),
                        Dense(128, na; initW = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1000,
            reward_type = Float64,
            state_type = Float64,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
        DoEveryNStep(10000) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )

    description = """
    The environment is EmptyGridWorld.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
