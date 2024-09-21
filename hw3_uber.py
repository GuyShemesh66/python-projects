# Skeleton file for HW3 questions 9 and 10
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
import random

from hw3_matchingmarket import market_eq

def _manhattan_distance(point1, point2):
    """Calculate the Manhattan distance between two points."""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


# === Problem 9(a) ===
def exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs):
    '''Given a market scenario for ridesharing, with n riders, and
    m drivers, output an exchange network representing the problem.
    -   The grid is size l x l. Points on the grid are tuples (x,y) where
        both x and y are in (0...l).
    -   rider_vals is a list of numbers, where rider_vals[i] is the value
        of rider i's trip
    -   rider_locs is a list of points, where rider_locs[i] is the current
        location of rider i (in 0...n-1)
    -   rider_dests is a list of points, where rider_dests[i] is the desired
        destination of rider i (in 0...n-1)
    -   driver_locs is a list of points, where driver_locs[j] is the current
        location of driver j (in 0...m-1)
    Output a tuple (n, m, V) representing a bipartite exchange network, where:
    -   V is an n x m list, with V[i][j] is the value of the edge between
        rider i (in 0...n-1) and driver j (in 0...m-1)'''
    V = [[0] * m for _ in range(n)]
    
    for rider_index in range(n):
        for driver_index in range(m):
            # Calculate the total distance the driver needs to travel
            distance_to_rider = _manhattan_distance(driver_locs[driver_index], rider_locs[rider_index])
            distance_of_trip = _manhattan_distance(rider_locs[rider_index], rider_dests[rider_index])
            total_distance = distance_to_rider + distance_of_trip
            
            # Calculate the value of this edge
            # The value is the rider's value minus the cost of the trip
            # We assume the cost is proportional to the distance
            V[rider_index][driver_index] = rider_vals[rider_index] - total_distance
            
            # Ensure the value is non-negative
            V[rider_index][driver_index] = max(0, V[rider_index][driver_index])
    
    return (n, m, V)


# === Problem 10 ===
def stable_outcome(n, m, V):
    '''Given a bipartite exchange network, with n riders, m drivers, and
    edge values V, output a stable outcome (M, A_riders, A_drivers).
    -   V is defined as in exchange_network_from_uber.
    -   M is an n-element list, where M[i] = j if rider i is 
        matched with driver j, and M[i] = None if there is no matching.
    -   A_riders is an n-element list, where A_riders[i] is the value
        allocated to rider i.
    -   A_drivers is an m-element list, where A_drivers[j] is the value
        allocated to driver j.'''
    # Use the market equilibrium algorithm to find a stable outcome
    P, M = market_eq(n, m, V)
    A_riders = [0] * n
    A_drivers = [0] * m
    
    for i in range(n):
        if M[i] is not None:
            edge_value = V[i][M[i]]
            driver_value = P[M[i]]
            rider_value = edge_value - driver_value
            A_riders[i] = max(0, rider_value)
            A_drivers[M[i]] = driver_value

    print(f"Riders' Allocation: {A_riders}")
    print(f"Drivers' Allocation: {A_drivers}")
    return (M, A_riders, A_drivers)

# === Problem 10(a) ===
    
def rider_driver_example_1():
    n = 5
    m = 5
    l = 10
    rider_vals = [80, 100, 90, 120, 110]
    rider_locs = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
    rider_dests = [(9, 9), (7, 7), (5, 5), (3, 3), (1, 1)]
    driver_locs = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8)]
    n1, m1, V1 =exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
    matching, A_riders1, A_drivers1 = stable_outcome(n1, m1, V1)
    return (matching, A_riders1, A_drivers1,n1,m1)

def rider_driver_example_2():
    n = 6
    m = 5
    l = 12
    rider_vals = [90, 110, 100, 130, 120, 95]
    rider_locs = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]
    rider_dests = [(11, 11), (9, 9), (7, 7), (5, 5), (3, 3), (1, 1)]
    driver_locs = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
    n1, m1, V1 =exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
    matching, A_riders1, A_drivers1 = stable_outcome(n1, m1, V1)
    return (matching, A_riders1, A_drivers1,n1,m1)

def rider_driver_example_3():
    n = 6
    m = 5
    l = 15
    rider_vals = [90, 100, 110, 120, 130, 140]
    rider_locs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    rider_dests = [(10, 10), (9, 9), (8, 8), (7, 7), (6, 6), (5, 5)]
    driver_locs = [(0, 0), (3, 3), (6, 6), (9, 9), (12, 12)]
    n1, m1, V1 =exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
    matching, A_riders1, A_drivers1 = stable_outcome(n1, m1, V1)
    return (matching, A_riders1, A_drivers1,n1,m1)

M1, A_riders1, A_drivers1,n1, m1, = rider_driver_example_1()
    
print("Example 1:")
print("Drivers number:", m1)
print("Riders number:", n1)
print("Matching:", M1)
print("Riders' Allocation:", A_riders1)
print("Drivers' Allocation:", A_drivers1)
    
#     # דוגמה 2
M2, A_riders2, A_drivers2 ,n2, m2, = rider_driver_example_2()
    
print("\nExample 2:")
print("Drivers number:", m2)
print("Riders number:", n2)
print("Matching:", M2)
print("Riders' Allocation:", A_riders2)
print("Drivers' Allocation:", A_drivers2)

M3, A_riders3, A_drivers3 ,n3, m3, = rider_driver_example_3()
print("\nExample 3:")
print("Drivers number:", m3)
print("Riders number:", n3)
print("Matching:", M3)
print("Riders' Allocation:", A_riders3)
print("Drivers' Allocation:", A_drivers3)

# === Problem 10(b) ===
def random_riders_drivers_stable_outcomes(n, m):
    '''Generates n riders, m drivers, each located randomly on the grid,
    with random destinations, each rider with a ride value of 100, 
    and returns the stable outcome.'''
    l = 100  # grid size
    value = 100

    rider_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    rider_dests = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    driver_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(m)]
    
    rider_vals = [value] * n
    n, m, V = exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
    M, A_riders, A_drivers = stable_outcome(n, m, V)
    return (M, A_riders, A_drivers)

def public_transport_stable_outcome(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b):
    '''Given an l x l grid, n riders, m drivers, and public transportation
    parameters (a,b), output a stable outcome (M, A_riders, A_drivers), where:
    -   rider_vals, rider_locs, rider_dests, driver_locs are defined the same
        way as in exchange_network_from_uber
    -   the cost of public transport is a + b * dist(start, end) where dist is
        manhattan distance
    -   M is an n-element list, where M[i] = j if rider i is 
        matched with driver j, and M[i] = -1 if rider i takes public transportation, and M[i] = None if there is no match for rider i.
    -   A_riders, A_drivers are defined as before.
    -   If there is no stable outcome, return None.
    '''
    _, _, V = exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
    max_iterations = 1000

    # Initialize variables
    M = [None] * n
    A_riders = [0] * n
    A_drivers = [0] * m
    driver_prices = [[1+_manhattan_distance(driver_locs[j], rider_locs[i]) + 
                      _manhattan_distance(rider_locs[i], rider_dests[i]) for j in range(m)] for i in range(n)]

    public_transport_costs = [ a + b * _manhattan_distance(rider_locs[i], rider_dests[i]) for i in range(n)]

    def assign_rides():
        for i in range(n):
            best_option_value = max(V[i][j] - driver_prices[i][j] for j in range(m))
            best_option_index = max(range(m), key=lambda j: V[i][j] - driver_prices[i][j])

            if best_option_value > rider_vals[i] - public_transport_costs[i]:
                M[i] = best_option_index
                A_riders[i] = best_option_value
                A_drivers[best_option_index] += driver_prices[i][best_option_index]
            else:
                M[i] = -1  # Choose public transport
                A_riders[i] = rider_vals[i] - public_transport_costs[i]

    def adjust_prices():
        for j in range(m):
            # If driver j is oversubscribed
            if M.count(j) > 1:
                for i in range(n):
                    if M[i] == j:
                        driver_prices[i][j] += 1
            # If driver j is unmatched
            elif M.count(j) == 0:
                for i in range(n):
                    driver_prices[i][j] = max(0, driver_prices[i][j] - 1)

    def check_stability(tolerance=5):
        for i in range(n):
            for j in range(m):
                if M[i] != j and M[i] != -1:
                    if V[i][j] - driver_prices[i][j] > A_riders[i] + tolerance:
                        return False
        return True

    # Main algorithm loop
    for _ in range(max_iterations):
        assign_rides()
        if check_stability():
            print("Stable outcome found.")
            return M, A_riders, A_drivers
        
        adjust_prices()

    print("No stable outcome found within max_iterations.")
    return None  # No stable outcome found within max_iterations

######################## Bonus Question 3(b) ##################################
print("Bonus 3(b)")
def generate_scenario(n, m, l):
    rider_vals = [random.randint(100, 300) for _ in range(n)]
    rider_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    rider_dests = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    driver_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(m)]
    return rider_vals, rider_locs, rider_dests, driver_locs

def analyze_public_transport_impact(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a_values, b_values):
    results = []
    for a in a_values:
        for b in b_values:
            print(f"\nAnalyzing scenario with a={a}, b={b}")
            result = public_transport_stable_outcome(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b)
            if result is not None:
                M, A_riders, A_drivers = result
                riders_using_public = M.count(-1)
                avg_rider_cost = sum(rider_vals[i] - A_riders[i] for i in range(n)) / n
                avg_driver_profit = sum(A_drivers) / m if m > 0 else 0
                results.append((a, b, riders_using_public, avg_rider_cost, avg_driver_profit))
    return results


def _print_analysis_results():
    print("\n")
    print(" Problem 10(b)")

    n=10
    m=10
    M, A_riders, A_drivers =random_riders_drivers_stable_outcomes(n, m)
    print("Drivers number:", m)
    print("Riders number:", n)
    print("Matching:", M)
    print("Riders' Allocation:", A_riders)
    print("Drivers' Allocation:", A_drivers)
    print("\n")
    n=5
    m=20
    M, A_riders, A_drivers =random_riders_drivers_stable_outcomes(n, m)
    print("Drivers number:", m)
    print("Riders number:", n)
    print("Matching:", M)
    print("Riders' Allocation:", A_riders)
    print("Drivers' Allocation:", A_drivers)
    print("\n")
    n=20
    m=5
    M, A_riders, A_drivers =random_riders_drivers_stable_outcomes(n, m)
    print("Drivers number:", m)
    print("Riders number:", n)
    print("Matching:", M)
    print("Riders' Allocation:", A_riders)
    print("Drivers' Allocation:", A_drivers)
    print("\n")


def main():
    _print_analysis_results()
    # Code for question 3a
    print("Bonus 3(a)")
    n = 10
    m = 5
    l = 100
    rider_vals = [random.randint(50, 150) for _ in range(n)]
    rider_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    rider_dests = [(random.randint(0, l), random.randint(0, l)) for _ in range(n)]
    driver_locs = [(random.randint(0, l), random.randint(0, l)) for _ in range(m)]
    a_values = [5, 10, 15, 20]
    b_values = [0.5, 1, 1.5, 2]
    a = random.choice(a_values)
    b = random.choice(b_values)
    public_transport_stable_outcome(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b)
    
    # Code for question 3b
    n, m, l = 20, 15, 50  # 20 riders, 15 drivers, 50x50 grid
    rider_vals, rider_locs, rider_dests, driver_locs = generate_scenario(n, m, l)

    a_values = [5, 10, 15, 20]
    b_values = [0.5, 1, 1.5, 2]

    results = analyze_public_transport_impact(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a_values, b_values)

    for a, b, riders_public, avg_rider_cost, avg_driver_profit in results:
        print(f"\na = {a}, b = {b}")
        print(f"Riders using public transport: {riders_public} out of {n}")
        print(f"Average rider cost: {avg_rider_cost:.2f}")
        print(f"Average driver profit: {avg_driver_profit:.2f}")

if __name__ == "__main__":
    main()