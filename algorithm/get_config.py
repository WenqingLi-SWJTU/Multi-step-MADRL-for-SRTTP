import argparse
import pickle
import numpy as np


# with open("26_10_act.pkl", "rb") as f:
#     act_26_10 = pickle.load(f)
#
# with open("26_10_timeslot.pkl", "rb") as f:
#     timeslot_26_10 = pickle.load(f)

def parse_args(parser):
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mat', choices=["mat", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru"])
    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=False,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=12, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='xxx',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False,
                        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='TTLE', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state_raw or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=2,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument("--critic_lr", type=float, default=5e-3,
                        help='critic learning rate (default: 5e-3)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')

    # parser.add_argument("--use_teaching_force", type=bool,
    #                     default=False, help='use a linear schedule on the learning rate')
    # parser.add_argument("--teaching_force_rate", type=float,
    #                     default=0.01, help='use a linear schedule on the learning rate')

    # save parameters
    parser.add_argument("--save_interval", type=int, default=100,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1,
                        help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    # parser.add_argument("--model_dir", type=str, default=None,
    #                     help="by default None. set the path to pretrained model.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")

    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)

    return parser.parse_args([])


def get_3_4_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='3_4',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=2,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=1,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=3,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=4,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=3,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=60,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[9, 8, 7],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0, 17, 47],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0, 0, 1],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=2,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=4,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(4, 15)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--stop_plan", type=list, default=[[0, 1], [1, 0], [0, 0]])
    parser.add_argument("--if_shuffle_stop_plan", type=bool, default=False)
    parser.add_argument("--use_action_mask", type=bool, default=False)

    return parser


def get_6_6_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='6_6',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=3,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=3,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=6,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=6,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=5,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=250,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7, 8, 8],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0, 30, 60, 90, 120, 140],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0, 0, 0, 1, 1, 1],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(8, 30)),
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((6, 6))
    stop_plan[0, :] = 1
    stop_plan[-1, :] = 1
    stop_plan = stop_plan.tolist()

    # stop_plan = [[0, 0, 0, 0],
    #              [0, 0, 0, 1],
    #              [1, 0, 1, 0],
    #              [1, 1, 1, 0],
    #              [0, 1, 0, 0],
    #              [0, 0, 0, 0]]

    parser.add_argument("--stop_plan", type=list, default=stop_plan)
    parser.add_argument("--if_shuffle_stop_plan", type=bool, default=False)
    parser.add_argument("--use_action_mask", type=bool, default=True)

    return parser


def get_10_10_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='10_10',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=5,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=5,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=10,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=10,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=9,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=570,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[8.1, 10.2, 9, 11.1, 10.3, 7.3, 6.7, 7.5, 10.3],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[10, 9, 8, 10, 8, 8, 7, 8, 10],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[10, 10, 9, 11, 10, 8, 8, 8, 16],
                        help="the running time of upstream trains in each section.")
    # parser.add_argument("--startTime", type=list, default=[0, 30, 60, 100, 170,
    #                                                        320, 450, 480, 510, 540],
    #                     help="the start time of trains at the origin station.")
    parser.add_argument("--startTime", type=list, default=[1, 75, 131, 239, 351,
                                                           184, 294, 403, 493, 562],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(5)] + [1 for _ in range(5)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(4, 20)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--teaching_action", type=list, default=None,
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((10, 10))
    stop_plan[:, 0] = 1
    stop_plan[:, -1] = 1
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist(),
                        help="the start time of trains at the origin station.")
    return parser


def get_20_10_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='20_10',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=10,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=10,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=20,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=10,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=9,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=600,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[8.1, 10.2, 9, 11.1, 10.3, 7.3, 6.7, 7.5, 10.3],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7, 9, 8, 7, 7, 6, 11],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7, 9, 9, 7, 6, 6, 9],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0, 30, 60, 90, 120, 160, 195, 217, 248, 282,
                                                           157, 192, 246, 280, 314, 351, 382, 400, 433, 467],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(10)] + [1 for _ in range(10)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=2,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=4,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(4, 60)),
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((20, 8)).tolist()

    parser.add_argument("--stop_plan", type=list, default=stop_plan)
    parser.add_argument("--if_shuffle_stop_plan", type=bool, default=False)
    parser.add_argument("--use_action_mask", type=bool, default=True)

    return parser


def get_26_10_config():
    parser = argparse.ArgumentParser(description="make the time tabling learning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='26_10',
                        help="the scale of the experiment")
    parser.add_argument("--numDownT", type=int, default=13,
                        help="the number of downstream trains.")
    parser.add_argument("--numUpT", type=int, default=13,
                        help="the number of upstream trains.")
    parser.add_argument("--numT", type=int, default=26,
                        help="the number of downstream and upstream trains.")
    parser.add_argument("--numS", type=int, default=10,
                        help="the number of stations.")
    parser.add_argument("--numB", type=int, default=9,
                        help="the number of sections.")
    parser.add_argument("--timeLossOfAc", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--timeLossOfDc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--timeZone", type=int, default=1440 - 6 * 60,
                        help="the time horizon.")
    parser.add_argument("--distance", type=list, default=[8.1, 10.2, 9, 11.1, 10.3, 7.3, 6.7, 7.5, 10.3],
                        help="the running time in each section.")
    parser.add_argument("--downRunTime", type=list, default=[9, 8, 7, 9, 8, 7, 7, 6, 11],
                        help="the running time of downstream trains in each section.")
    parser.add_argument("--upRunTime", type=list, default=[9, 8, 7, 9, 9, 7, 6, 6, 9],
                        help="the running time of upstream trains in each section.")
    parser.add_argument("--startTime", type=list, default=[0., 90., 170., 240., 320., 400., 480., 560.,
                                                           640., 720., 800., 880., 960., 121., 201., 292., 370., 457.,
                                                           537., 617., 697., 777., 857., 937., 1017., 1080.],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--direction", type=list, default=[0 for _ in range(13)] + [1 for _ in range(13)],
                        help="the start time of trains at the origin station.")
    parser.add_argument("--staHeadway", type=int, default=4,
                        help="the station headway.")
    parser.add_argument("--secHeadway", type=int, default=2,
                        help="the section headway.")
    parser.add_argument("--cHeadwayWhenLaterStop", type=int, default=2,
                        help="the consecutive headway when later train stop at the backward station.")
    parser.add_argument("--cHeadwayWhenLaterPass", type=int, default=4,
                        help="the consecutive headway when later train pass through the backward station.")
    parser.add_argument("--ava_actions", type=list, default=[0] + list(range(10, 26)),
                        help="the start time of trains at the origin station.")
    parser.add_argument("--teaching_action", type=list, default=None,
                        help="the start time of trains at the origin station.")
    stop_plan = np.zeros((26, 10))
    stop_plan[0, :] = 1
    stop_plan[-1, :] = 1
    parser.add_argument("--stop_plan", type=list, default=stop_plan.tolist())
    return parser


def get_small_net_config():
    parser = argparse.ArgumentParser(description="make the integrated line planning, timetabling learning,"
                                                 "and rolling stock planning environment")

    # environment
    parser.add_argument('--scenario', type=str, default='small_net_IOLTR',
                        help="the scale of the experiment")

    parser.add_argument("--num_station", type=int, default=5,
                        help="the number of stations.")
    parser.add_argument("--num_block", type=int, default=7,
                        help="the number of blocks.")
    parser.add_argument("--num_rs_compo", type=int, default=2,
                        help="the number of blocks.")

    parser.add_argument("--station_block_conn", type=list, default=[(0, 3, 4), (0, 1, 5), (1, 2, 6), (2, 3, 7),
                                                                    (4, 5, 6)],  # connected blocks
                        help="the connection from stations to blocks, the first line is IDs, and the second"
                             "line is connected blocks.")
    parser.add_argument("--depot", type=list, default=[0],
                        help="the location of depot.")
    parser.add_argument("--block_station_conn", type=list,
                        default=[(0, 1), (1, 2), (3, 2), (0, 3), (0, 4), (4, 1),
                                 (4, 3)],  # connected stations
                        help="the connection from blocks to stations.")
    parser.add_argument("--distance", type=list, default=[3, 3, 3, 3, 2, 2, 2],
                        help="the distance between each block.")
    parser.add_argument("--max_allowed_speed_block", type=list, default=[1.5, 1.5, 1.5, 1.5, 1, 1, 1],
                        help="max allowed speed in each block.")

    parser.add_argument("--train_path_compo", type=list, default=[
        [(0, 4, 1), (0, 4, 3), (0, 1, 2), (0, 3, 2)],  # outbound trains
        [(1, 4, 0), (3, 4, 0), (2, 1, 0), (2, 3, 0)],  # inbound trains
    ],
                        help="the train path compositions, and the first line is about outbound trains, "
                             "and the second line is about inbound trains.")
    parser.add_argument("--num_train_path_compo", type=int, default=4)

    parser.add_argument("--passenger_flow", type=list, default=[[0, 979, 830, 925, 1172],
                                                                [714, 0, 995, 693, 941],
                                                                [1139, 990, 0, 934, 682],
                                                                [675, 931, 796, 0, 974],
                                                                [894, 1100, 854, 783, 0]],
                        help="passenger flow.")

    parser.add_argument("--rolling_stock_compo", type=list, default=[0, 1],
                        help="rolling_stock_compositions.")
    parser.add_argument("--num_seats", type=list, default=[50, 100],
                        help="the number of seats of each rolling stock composition.")
    parser.add_argument("--max_allowed_speed_rs", type=list, default=[1, 1.5],
                        help="max allowed speed of each rolling stock composition.")
    parser.add_argument("--inventory", type=list, default=[10, 10],
                        help="the inventory of each rolling stock composition.")

    parser.add_argument("--time_loss_of_ac", type=int, default=2,
                        help="the time loss of train acceleration.")
    parser.add_argument("--time_loss_of_dc", type=int, default=3,
                        help="the time loss of train deceleration.")
    parser.add_argument("--time_zone", type=int, default=600,
                        help="the time horizon.")
    parser.add_argument("--headway", type=int, default=5,
                        help="the tracking headway between two trains running in the same direction.")

    parser.add_argument("--max_stop_time", type=int, default=5,
                        help="the tracking headway between two trains running in the same direction.")
    parser.add_argument("--min_stop_time", type=int, default=2,
                        help="the tracking headway between two trains running in the same direction.")

    parser.add_argument("--punish_factor", type=int, default=-500)

    parser.add_argument("--transfer_pf_board_ratio", type=float, default=0.3,
                        help="the board ratio of transfer passenger on each train.")
    parser.add_argument("--min_transfer_time", type=int, default=20)
    parser.add_argument("--max_transfer_time", type=int, default=60)

    parser.add_argument("--min_turnaround_time", type=int, default=20)
    parser.add_argument("--max_turnaround_time", type=int, default=40)

    return parser
