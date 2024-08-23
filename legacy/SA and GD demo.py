import numpy as np

def cost_function(x):
    """Example cost function: f(x) = x^2 + 4*sin(5x) + 6*cos(3x)"""
    return x**2 + 4*np.sin(5*x) + 6*np.cos(3*x)

def gradient(x):
    """Gradient of the cost function"""
    return 2*x + 20*np.cos(5*x) - 18*np.sin(3*x)

def simulated_annealing(cost_function, initial_temp, cooling_rate, max_iterations):
    # Initialize parameters
    current_temp = initial_temp
    current_solution = np.random.uniform(-10, 10)  # Random initial solution
    current_cost = cost_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost

    global sa_current_solution; sa_current_solution = []
    for iteration in range(max_iterations):
        # Generate a new solution in the neighborhood
        new_solution = current_solution + np.random.uniform(-1, 1)
        new_cost = cost_function(new_solution)
        
        # Calculate acceptance probability
        if new_cost < current_cost:
            acceptance_probability = 1.0
        else:
            acceptance_probability = np.exp((current_cost - new_cost) / current_temp)
        
        # Accept or reject the new solution
        if acceptance_probability > np.random.rand():
            current_solution = new_solution
            current_cost = new_cost
        
        # Update the best solution found
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        
        # Cool down the temperature
        current_temp *= cooling_rate

        sa_current_solution.append(current_solution)
        
        # Optionally: print the progress
        print(f"Iteration {iteration}: Current Cost = {current_cost}, Best Cost = {best_cost}")

    return best_solution, best_cost

def gradient_descent(cost_function, gradient, initial_guess, learning_rate, max_iterations, tolerance):
    # Initialize parameters
    current_solution = initial_guess
    current_cost = cost_function(current_solution)

    for iteration in range(max_iterations):
        # Calculate gradient
        grad = gradient(current_solution)
        
        # Update solution
        new_solution = current_solution - learning_rate * grad
        new_cost = cost_function(new_solution)
        
        # Optionally: print the progress
        print(f"Iteration {iteration}: Current Cost = {new_cost}, Solution = {new_solution}")

        # Check for convergence
        if abs(new_cost - current_cost) < tolerance:
            break
        
        current_solution = new_solution
        current_cost = new_cost

    return current_solution, current_cost

sa_best_solutions = []
sa_best_costs = []

gd_best_solutions = []
gd_best_costs = []
max_iterations = 1000
for i in range(10):
    # Parameters for simulated annealing
    initial_temp = 1000
    cooling_rate = 0.995

    # Parameters for gradient descent
    initial_guess = np.random.uniform(-10, 10)
    learning_rate = 0.0001
    tolerance = 1e-6
    # Perform simulated annealing
    best_solution_sa, best_cost_sa = simulated_annealing(cost_function, initial_temp, cooling_rate, max_iterations)
    sa_best_solutions.append(best_solution_sa)
    sa_best_costs.append(best_cost_sa)

    # Perform gradient descent
    best_solution_gd, best_cost_gd = gradient_descent(cost_function, gradient, initial_guess, learning_rate, max_iterations, tolerance)
    gd_best_solutions.append(best_solution_gd)
    gd_best_costs.append(best_cost_gd)

def average_similar_values(x_values, threshold):
    # Sort the input array to simplify grouping
    x_values = np.sort(x_values)
    
    # Initialize variables
    averages = []
    counts = []
    i = 0
    n = len(x_values)
    
    while i < n:
        # Initialize the group
        group = [x_values[i]]
        count = 1
        
        # Find all values within the threshold
        j = i + 1
        while j < n and abs(x_values[j] - x_values[i]) <= threshold:
            group.append(x_values[j])
            count += 1
            j += 1
        
        # Calculate the average of the group
        group_average = np.mean(group)
        
        # Append the results
        averages.append(group_average)
        counts.append(count)
        
        # Move to the next unprocessed value
        i = j
    
    return np.array(averages), np.array(counts)

threshold = 0.3
gd_average_value, gd_count = average_similar_values(gd_best_solutions, threshold)
sa_average_value, sa_count = average_similar_values(sa_best_solutions, threshold)

# Create a plot of the cost function and the best solutions found
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 1000)
y = cost_function(x)
plt.plot(x, y)
plt.scatter(gd_best_solutions, gd_best_costs,label='Gradient Descent', color='blue', marker="o")
for i in range (len(gd_average_value)):
    plt.text(gd_average_value[i],  cost_function(gd_average_value[i]) - 7, gd_count[i], horizontalalignment='center', size='medium', color='blue', weight='semibold')
    
plt.scatter(sa_best_solutions, sa_best_costs,label='Simulated Annealing', color='red', marker="*")
for i in range (len(sa_average_value)):
    plt.text(sa_average_value[i],  cost_function(sa_average_value[i]) + 7, sa_count[i], horizontalalignment='center', size='medium', color='red', weight='semibold')

# plt.scatter(gd_best_solutions, gd_best_costs, color='blue', label='Gradient Descent', marker="o")
# plt.scatter(sa_best_solutions, sa_best_costs, color='red', label='Simulated Annealing', marker="*")

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Cost Function: f(x) = x^2 + 4*sin(5x) + 6*cos(3x)')
plt.show()