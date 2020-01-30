from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import math
from scipy.stats import norm

player_count = 100
game_count = 5000

q = player_count/4
X = np.random.uniform(0,1,(game_count,player_count))
shapley = np.zeros(player_count)
pi = math.pi
for data_point in tqdm(range(game_count)):
    mu = float(np.mean(X[data_point,:]))
    var = float(np.var(X[data_point,:]))
    for player_index in range(player_count):
        var_s = np.power(var/np.linspace(1,player_count-1,player_count-1),0.5)
        a_s = (q-X[data_point, player_index])/np.linspace(1,player_count-1,player_count-1)
        #var_s = [(var/coalition_size)**0.5 for coalition_size in range(1, player_count)]
        #a_s = [(q - X[data_point, player_index])/coalition_size for coalition_size in range(1, player_count)]
        #b_s = [(q-10**-20)/coalition_size for coalition_size in range(1, player_count)]
        b_s = (q-10**-20)/np.linspace(1,player_count-1,player_count-1)
        shap_in_game = norm.cdf(b_s, loc=mu, scale=var_s)-norm.cdf(a_s, loc=mu, scale=var_s)
        shapley[player_index] = shapley[player_index] + np.sum(shap_in_game)

shapley = shapley/(game_count*player_count)
print(shapley)
print(np.sum(shapley))
