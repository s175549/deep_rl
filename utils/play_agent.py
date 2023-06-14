
def play_agent(env,agent,agent_name):

    state = env.reset()

    for _ in range(10):

        terminated = False
        while not terminated:
            env.render("human")
            # Take an action and observe next_state, reward and if terminated
            action, _, _ = agent.take_action(state)
            next_state, reward, terminated, _ = env.step(action)
            state = next_state if not terminated else env.reset()