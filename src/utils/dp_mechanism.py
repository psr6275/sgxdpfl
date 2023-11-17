import numpy as np
from .tensorflow_privacy.compute_noise_from_budget_lib import compute_noise


def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size


#     return 2 * lr * clip

def cal_sensitivity_MA(lr, clip, dataset_size):
    return lr * clip / dataset_size

def calculate_noise_scale_cdp(args, times):
    if args.dp_mechanism == 'Laplace':
        epsilon_single_query = args.dp_epsilon / times
        return Laplace(epsilon = epsilon_single_query)
    elif args.dp_mechanism == 'Gaussian':
        epsilon_single_query = args.dp_epsilon / times
        delta_single_query = args.dp_delta / times
        return Gaussian_Simple(epsilon=epsilon_single_query, delta = delta_single_query)
    elif args.dp_mechanism == 'MA':
        return Gaussian_MA(epsilon = args.dp_epsilon, delta = args.dp_delta, 
                           q = args.dp_sample, epoch = times)
# def Laplace(epsilon, sensitivity, size):
#     noise_scale = sensitivity / epsilon
#     return np.random.laplace(0, scale=noise_scale, size=size)

def Laplace(epsilon):
    return 1 / epsilon


def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def Gaussian_MA(epsilon, delta, q, epoch):
    return compute_noise(1, q, epsilon, epoch, delta, 1e-5)
