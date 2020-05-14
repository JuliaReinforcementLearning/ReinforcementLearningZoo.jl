export Experiment

using Dates
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Flux
using BSON
using TensorBoardLogger
using Logging

function RLCore.Experiment(::Val{:juliarl}, ::Val{:BasicDQN}, ::Val{:CartPole}, ::Nothing; save_dir=nothing)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyymmddHHMMSS")
        save_dir = joinpath(pwd(), "juliarl_BasicDQN_CartPole_$(t)")
    end

    lg=TBLogger(joinpath(save_dir, "tb_log"), min_level=Logging.Info)

    env = CartPoleEnv(; T = Float32 #= , seed = 11 =# )
    ns, na = length(rand(get_observation_space(env))), length(get_action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 128, relu; initW = seed_glorot_uniform(#= seed = 17 =#)),
                        Dense(128, 128, relu; initW = seed_glorot_uniform(#= seed = 23 =#)),
                        Dense(128, na; initW = seed_glorot_uniform(#= seed = 39 =#)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                #= seed = 22, =#
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                #= seed = 33, =#
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 1000,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    stop_condition = StopAfterStep(10000)

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env, obs
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                with_logger(lg) do
                    @info "training" loss=agent.policy.learner.loss
                end
            end
        end,
        DoEveryNStep(10000) do t, agent, env, obs
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end
        )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is CartPoleEnv.

    Agent and statistic info will be saved to: $save_dir
    You can also view the tensorboard logs with `tensorboard --logdir $(joinpath(save_dir, "tb_log"))`
    To load the agent and statistic info:
    ```
    agent = RLCore.load("$save_dir", Agent)
    BSON.@load joinpath("$save_dir", "stats.bson") total_reward_per_episode time_per_step
    ```
    """

    Experiment(agent, env, stop_condition, hook, description)
end
