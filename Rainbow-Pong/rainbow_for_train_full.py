from Common.logger import Logger
from Brain.agent import Agent
from Common.utils import *
from Common.config import get_params
import time

if __name__ == '__main__':
    #参数初始化
    params = get_params()
    test_env = make_atari(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    print(f"Environment: {params['env_name']}\n"
          f"Number of actions:{params['n_actions']}")
    # 创建训练环境
    env = make_atari(params["env_name"])
    env.seed(int(time.time()))

    agent = Agent(**params)
    logger = Logger(agent, **params)
    # 使用预训练模型
    if not params["train_from_scratch"]:
        chekpoint = logger.load_weights()
        agent.online_model.load_state_dict(chekpoint["online_model_state_dict"])
        agent.hard_update_of_target_network()
        params.update({"beta": chekpoint["beta"]})
        min_episode = chekpoint["episode"]
        print("Keep training from previous run.")
    # 从头开始训练模型
    else:
        min_episode = 0
        print("Train from scratch.")

    #执行训练
    if params["do_train"]:
        stacked_states = np.zeros(shape=params["state_shape"], dtype=np.uint8)
        state = env.reset()
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        beta = params["beta"]
        loss = 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):
            stacked_states_copy = stacked_states.copy()
            action = agent.choose_action(stacked_states_copy)
            next_state, reward, done, _ = env.step(action)

            # 累积状态
            stacked_states = stack_states(stacked_states, next_state, False)
            reward = np.clip(reward, -1.0, 1.0)
            agent.store(stacked_states_copy, action, reward, stacked_states, done)
            episode_reward += reward

            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # ---------------------             2 填空           -----------------------------#
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            #2 使用多步回报，在训练的前期目标价值可以估计地更准，从而加快模型的训练
            if step % params["train_period"] == 0:
                beta = min(1.0, params["beta"] + step * (1.0 - params["beta"]) / params["final_annealing_beta_steps"])
                loss += agent.train(beta)
            agent.soft_update_of_target_network()

            if done:
                logger.off()
                logger.log(episode, episode_reward, loss, step, beta)
                episode += 1
                state = env.reset()
                stacked_frames = stack_states(stacked_states, state, True)
                episode_reward = 0
                episode_loss = 0
                logger.on()

