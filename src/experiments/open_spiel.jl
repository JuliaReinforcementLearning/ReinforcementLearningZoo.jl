function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:Minimax},
    ::Val{:OpenSpielEnv},
    game;
)
    env = OpenSpielEnv(game)
    agents = (
        Agent(policy=MinimaxPolicy(), role=0),
        Agent(policy=MinimaxPolicy(), role=1)
    )
    hooks = (TotalRewardPerEpisode(), TotalRewardPerEpisode())
    Experiment(agents, env, StopAfterEpisode(1), hooks)
end