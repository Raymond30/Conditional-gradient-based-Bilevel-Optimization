import sympy as sp
import numpy as np
from sympy import Matrix, solve, symbols, diff, hessian

def find_six_hump_camel_minima():
    # Define the variables
    x1, x2 = symbols('x1 x2')
    
    # Define the function
    f = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
    
    # Calculate partial derivatives
    df_dx1 = diff(f, x1)
    df_dx2 = diff(f, x2)
    
    # Find critical points by solving system of equations
    # We'll use numerical solving due to the complexity of the equations
    from sympy.solvers.solvers import nsolve
    from itertools import product
    
    # Create a grid of initial guesses within the bounds
    x1_guesses = np.linspace(-3, 3, 10)
    x2_guesses = np.linspace(-2, 2, 10)
    critical_points = []
    
    for x1_guess, x2_guess in product(x1_guesses, x2_guesses):
        try:
            sol = nsolve([df_dx1, df_dx2], [x1, x2], [x1_guess, x2_guess], verify=False)
            x1_val = float(sol[0])
            x2_val = float(sol[1])
            
            # Check if point is within bounds
            if -3 <= x1_val <= 3 and -2 <= x2_val <= 2:
                # Check if this point is unique (not already found)
                is_unique = True
                for existing_point in critical_points:
                    if abs(existing_point[0] - x1_val) < 1e-4 and abs(existing_point[1] - x2_val) < 1e-4:
                        is_unique = False
                        break
                if is_unique:
                    critical_points.append((x1_val, x2_val))
        except:
            continue
    
    # Calculate Hessian matrix
    H = hessian(f, (x1, x2))
    
    print(f"Critical points found: {len(critical_points)}")
    
    # Check each critical point
    local_minima = []
    for point in critical_points:
        x1_val, x2_val = point
        
        # Evaluate Hessian at this point
        H_eval = H.subs({x1: x1_val, x2: x2_val}).evalf()
        
        # Convert to numpy array for eigenvalue computation
        H_np = np.array(H_eval).astype(np.float64)
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(H_np)
        
        # Check if all eigenvalues are positive (local minimum)
        if all(ev > 0 for ev in eigenvals):
            f_val = float(f.subs({x1: x1_val, x2: x2_val}).evalf())
            local_minima.append({
                'point': (x1_val, x2_val),
                'value': f_val,
                'eigenvalues': eigenvals
            })
    
    # Sort by function value to identify global minima
    local_minima.sort(key=lambda x: x['value'])
    
    # Print results
    print("\nLocal Minima Found:")
    for i, min_point in enumerate(local_minima, 1):
        is_global = abs(min_point['value'] - (-1.0316)) < 1e-4
        print(f"\nMinimum {i}:")
        print(f"Point (x1, x2) = {min_point['point']}")
        print(f"Function value = {min_point['value']}")
        print(f"Is global minimum: {is_global}")
    
    return local_minima

if __name__ == "__main__":
    local_minima = find_six_hump_camel_minima()