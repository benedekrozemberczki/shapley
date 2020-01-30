from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import math

def integrand(x, N, mu, sigma):
    return np.exp(-N*(((x-mu)**2)/(2*var)))

player_count = 100
game_count = 50

q = player_count/4
X = np.random.uniform(0,1,(game_count,player_count))
shapley = np.zeros(player_count)
pi = math.pi
for data_point in tqdm(range(game_count)):
    mu = np.mean(X[data_point,:])
    var = np.var(X[data_point,:])
    for player_index in range(player_count):
        shap_in_game = 0
        for coalition_size in range(1, player_count):
            a = (q - X[data_point, player_index])/coalition_size
            b = (q-10**-20)/coalition_size
            I = quad(integrand, a, b, args=(coalition_size, mu, var))
            marginal = (1.0/(((2*pi*var)/coalition_size)**0.5))*I[0]
            shap_in_game = shap_in_game + marginal
        shapley[player_index] = shapley[player_index] + shap_in_game

shapley = shapley/(game_count*player_count)

print(np.sum(shapley))
            
