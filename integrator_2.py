from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import math
from numba import jit

@jit(nopython=True)
def integrand(x, N, mu, var):
    return np.exp(-N*(((x-mu)**2)/(2*var)))


@jit(nopython=True) 
def get_a_b(coalition_size, q, w):
    a = (q - w)/coalition_size
    b = (q-10.0**-20)/coalition_size
    return (a,b)

@jit(nopython=True) 
def get_marginal(var, coalition_size, I):
    return (1.0/(((2.0*math.pi*var)/coalition_size)**0.5))*I

def marginal_calc(coalition_size, q, w, mu, var):
    (a, b) = get_a_b(coalition_size, q, w)
    I = quad(integrand, a, b, args=(coalition_size, mu, var))[0]
    marginal = get_marginal(var, coalition_size, I)
    return marginal


player_count = 100
game_count = 10000

q = player_count/4


X = np.random.uniform(0,1,(game_count,player_count))


def calculate_shapley(X, q):

    player_count = X.shape[1]
    game_count = X.shape[0]
    
    shapley = np.zeros(player_count)
    
    for data_point in tqdm(range(game_count)):
        mu = float(np.mean(X[data_point,:]))
        var = float(np.var(X[data_point,:]))
        for player_index in range(player_count):
            w = float(X[data_point, player_index])
            shap_in_game = sum([marginal_calc(coalition_size, q, w, mu, var) for coalition_size in range(1, player_count)])
            
            shapley[player_index] = shapley[player_index] + shap_in_game

    shapley = shapley/(game_count*player_count)
    return shapley


shapley = calculate_shapley(X, q)

print(np.sum(shapley))
