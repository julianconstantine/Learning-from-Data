# HOEFFDING'S INEQUALITY
# Run a computer simulation for flipping 1000 virtual fair coins. Flip each coin independently 10 times. Focus
# on 3 coins as follows: c_1 is the first coin flipped, c_rand is a coin chosen randomly from the 1000 coins,
# and c_min is the coin which had the minimum frequency of heads (pick the earlier one in case of a tie). Let nu_1,
# nu_rand, and nu_min be the fraction of heads obtained for the 3 respective coins out of the 10 tosses.

# Run the experiment 100,000 times in order to get a full distribution of nu_1, nu_rand, and nu_min (note that c_rand
# and c_min will change from run to run).

import numpy as np


def flip_coins(ncoins, nflips, ntrials):
    coin_flips = np.random.randint(low=0, high=2, size=[ncoins, nflips, ntrials])

    return coin_flips


NUM_COINS = 1000
NUM_FLIPS = 10
NUM_TRIALS = 100000

c = flip_coins(ncoins=NUM_COINS, nflips=NUM_FLIPS, ntrials=NUM_TRIALS)

nu = np.mean(a=c, axis=1)

nu_1 = nu[0:NUM_COINS][1]
nu_min = np.min(a=nu, axis=0)
nu_rand = nu[np.random.randint(low=0, high=NUM_COINS)]

print("Average value of nu_1: " + str(np.mean(nu_1)))
print("Average value of nu_min: " + str(np.mean(nu_min)))
print("Average value of nu_rand: " + str(np.mean(nu_rand)))

# 1. The average value of nu_min is closest to:
#   [a] 0
#   [b] 0.01
#   [c] 0.1
#   [d] 0.5
#   [e] 0.67

# The average value of nu_min is about 0.03-0.04, so closest to [b] 0.01
#   CHECK: CORRECT!
print("Average value of nu_min: " + str(np.mean(nu_min)))

# 2. Which coin(s) has a distribution of nu that satisfies the (single-bin) Hoeffding Inequality?
#   [a] c_1 only
#   [b] c_rand only
#   [c] c_min only
#   [d] c_1 and c_rand
#   [e] c_min and c_rand

# The average values of nu_1 and nu_rand are approximately 0.5, so [d] c_1 and c_rand satisfy the single-bin
# Hoeffding Inequality
#   CHECK: CORRECT!
print("Average value of nu_1: " + str(np.mean(nu_1)))
print("Average value of nu_min: " + str(np.mean(nu_min)))
print("Average value of nu_rand: " + str(np.mean(nu_rand)))