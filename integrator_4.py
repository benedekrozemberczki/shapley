from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import math
from scipy.stats import norm
from numba import jit

player_count = 100
game_count = 500

q = player_count/4
X = np.random.uniform(0,1,(game_count,player_count))
shapley = np.zeros(player_count)
pi = math.pi

@jit
def create_vectors(var, player_count, q, w):
    var_s = np.power(var/np.linspace(1,player_count-1,player_count-1),0.5)
    a_s = (q-w)/np.linspace(1,player_count-1,player_count-1)
    b_s = (q-10**-20)/np.linspace(1,player_count-1,player_count-1)
    return var_s, a_s, b_s

for data_point in tqdm(range(game_count)):
    mu = float(np.mean(X[data_point,:]))
    var = float(np.var(X[data_point,:]))
    for player_index in range(player_count):
        w = X[data_point, player_index]
        var_s, a_s, b_s = create_vectors(var, player_count, q, w)
        shap_in_game = norm.cdf(b_s, loc=mu, scale=var_s)-norm.cdf(a_s, loc=mu, scale=var_s)
        shapley[player_index] = shapley[player_index] + np.sum(shap_in_game)

shapley = shapley/(game_count*player_count)
print(shapley)
print(np.sum(shapley))
