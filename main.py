import numpy as np
from typing import Callable, Tuple, List, Optional

class DIRECTOptimizer:
    """
    Implementation of the DIRECT (DIviding RECTangles) algorithm for global optimization.
    
    DIRECT is a deterministic sampling algorithm for bound-constrained global optimization
    that doesn't require gradient information.
    
    Parameters:
    -----------
    func : Callable
        The objective function to minimize
    bounds : List[Tuple[float, float]]
        List of (lower, upper) bounds for each dimension
    max_iter : int, default=1000
        Maximum number of iterations
    max_feval : int, default=10000
        Maximum number of function evaluations
    min_feval : int, default=10
        Maximum number of function evaluations
    eps : float, default=1e-4
        Convergence tolerance
    vol_tol : float, default=1e-16
        Minimum hyperrectangle volume tolerance
    size_tol : float, default=1e-12
        Minimum hyperrectangle size tolerance
    """
    
    def __init__(self, 
                 func: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 max_iter: int = 1000,
                 max_feval: int = 10000,
                 min_feval: int = 10,
                 eps: float = 1e-4,
                 vol_tol: float = 1e-16,
                 size_tol: float = 1e-12):
        
        self.func = func
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.max_iter = max_iter
        self.max_feval = max_feval
        self.min_feval = min_feval
        self.eps = eps
        self.vol_tol = vol_tol
        self.size_tol = size_tol
        
        # Validate that max_feval >= min_feval
        if max_feval < min_feval:
            print(f"ERROR: max_feval ({max_feval}) must be ≥ min_feval ({min_feval})")
            print("Terminating program.")
            exit()  # Exit with error code 1        

        # Initialize center point
        self.center = self.bounds[:, 0] + 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.center_value = func(self.center)
        
        # Initialize rectangle structure
        self.rectangles = []
        self.feval_count = 1
        self.iteration_count = 0
        
        # Best solution found
        self.best_x = self.center.copy()
        self.best_value = self.center_value
        
    class Rectangle:
        """Class to represent a hyperrectangle in DIRECT algorithm."""
        def __init__(self, center: np.ndarray, value: float, diameter: float, 
                     side_lengths: np.ndarray, index: int):
            self.center = center
            self.value = value
            self.diameter = diameter
            self.side_lengths = side_lengths
            self.index = index  # Used for tracking which dimension was divided last
            
        def __repr__(self):
            return f"Rectangle(center={self.center}, value={self.value:.6f}, diameter={self.diameter:.6f})"
    
    def _normalize_coordinates(self, x: np.ndarray) -> np.ndarray:
        """Convert from unit hypercube [0,1]^n to actual bounds."""
        return self.bounds[:, 0] + x * (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _get_potentially_optimal_rectangles(self, rectangles: List[Rectangle]) -> List[Rectangle]:
        """
        Identify potentially optimal rectangles based on the DIRECT algorithm.
        
        A rectangle is potentially optimal if there exists some rate-of-change constant
        such that the rectangle lies on the lower convex hull of the set of points
        (diameter, function value).
        """
        if not rectangles:
            return []
        
        # Sort rectangles by diameter
        sorted_rects = sorted(rectangles, key=lambda r: r.diameter)
        
        # Extract diameters and values
        diameters = np.array([r.diameter for r in sorted_rects])
        values = np.array([r.value for r in sorted_rects])
        
        potentially_optimal = []
        n = len(sorted_rects)
        
        # Start with the rectangle having the smallest function value
        min_idx = np.argmin(values)
        potentially_optimal.append(sorted_rects[min_idx])
        
        # Find other potentially optimal rectangles using convex hull method
        j = min_idx
        while j < n - 1:
            found = False
            k = j + 1
            while k < n:
                # Check if diameters are equal (to avoid division by zero)
                if abs(diameters[k] - diameters[j]) < 1e-12:
                    # If diameters are equal, compare values directly
                    if values[k] < values[j]:
                        # Rectangle k has same diameter but better value
                        potentially_optimal.append(sorted_rects[k])
                        j = k
                        found = True
                        break
                    else:
                        # Skip rectangle k since it's worse or equal
                        k += 1
                        continue
                
                # Calculate slope (safe now since we know diameters differ)
                slope = (values[k] - values[j]) / (diameters[k] - diameters[j])
                valid = True
                
                # Check if this slope makes rectangle j potentially optimal
                for m in range(j + 1, k):
                    # Also check for equal diameters here
                    if abs(diameters[m] - diameters[j]) < 1e-12:
                        # If diameters are equal at intermediate point
                        if values[m] < values[j]:
                            valid = False
                            break
                    elif values[m] < values[j] + slope * (diameters[m] - diameters[j]):
                        valid = False
                        break
                
                if valid:
                    potentially_optimal.append(sorted_rects[k])
                    j = k
                    found = True
                    break
                k += 1
            
            if not found:
                break

        return potentially_optimal
    
    def _divide_rectangle(self, rectangle: Rectangle) -> List[Rectangle]:
        """
        Divide a potentially optimal rectangle along only one dimension.
        """
        new_rectangles = []
        
        # Find dimensions with maximum side length
        max_length = np.max(rectangle.side_lengths)
        max_dims = np.where(rectangle.side_lengths == max_length)[0]
        
        # Choose which dimension to divide
        if len(max_dims) == 1:
            divide_dim = max_dims[0]
        else:
            # If multiple dimensions have max length, prefer the one not divided last
            if rectangle.index >= 0 and rectangle.index in max_dims:
                # Avoid dividing the same dimension consecutively
                available_dims = [d for d in max_dims if d != rectangle.index]
                divide_dim = available_dims[0] if available_dims else max_dims[0]
            else:
                divide_dim = max_dims[0]  # Choose first available
        
        # Calculate new side lengths
        new_side_lengths = rectangle.side_lengths.copy()
        new_side_lengths[divide_dim] = max_length / 3.0
        
        # Calculate new diameter
        new_diameter = np.linalg.norm(new_side_lengths)
        
        # Calculate offset for new centers
        offset = max_length / 3.0
        
        # Create two new rectangles
        for direction in [-1, 1]:
            new_center = rectangle.center.copy()
            new_center[divide_dim] += direction * offset
            
            # Ensure new center is within bounds [0, 1] (normalized coordinates)
            new_center = np.clip(new_center, 0, 1)
            
            # Evaluate at new center
            actual_coords = self._normalize_coordinates(new_center)
            new_value = self.func(actual_coords)
            self.feval_count += 1
            
            # Update best solution if found
            if new_value < self.best_value:
                self.best_value = new_value
                self.best_x = actual_coords.copy()
            
            # Create new rectangle
            new_rect = self.Rectangle(
                center=new_center,
                value=new_value,
                diameter=new_diameter,
                side_lengths=new_side_lengths.copy(),
                index=divide_dim
            )
            new_rectangles.append(new_rect)
                
        return new_rectangles
    
    def optimize(self) -> dict:
        """
        Run the DIRECT optimization algorithm.
        
        Returns:
        --------
        dict: Dictionary containing optimization results:
            - 'x': Best solution found
            - 'fun': Best function value
            - 'nit': Number of iterations
            - 'nfev': Number of function evaluations
            - 'success': Whether optimization converged
            - 'message': Description of termination reason
        """
        # Initialize with unit hypercube
        initial_center = np.ones(self.n_dim) * 0.5  # Center of unit hypercube
        initial_side_lengths = np.ones(self.n_dim)  # All sides have length 1 initially
        initial_diameter = np.linalg.norm(initial_side_lengths)
        
        initial_rect = self.Rectangle(
            center=initial_center,
            value=self.center_value,
            diameter=initial_diameter,
            side_lengths=initial_side_lengths,
            index=-1
        )
        
        self.rectangles = [initial_rect]
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            self.iteration_count = iteration + 1
            
            # Check termination criteria
            if self.feval_count >= self.max_feval:
                return {
                    'x': self.best_x,
                    'fun': self.best_value,
                    'nit': self.iteration_count,
                    'nfev': self.feval_count,
                    'success': False,
                    'message': 'Maximum number of function evaluations reached'
                }
            
            # Find potentially optimal rectangles
            pot_optimal = self._get_potentially_optimal_rectangles(self.rectangles)
            
            if not pot_optimal:
                return {
                    'x': self.best_x,
                    'fun': self.best_value,
                    'nit': self.iteration_count,
                    'nfev': self.feval_count,
                    'success': True,
                    'message': 'No potentially optimal rectangles found'
                }
            
            # Divide potentially optimal rectangles
            new_rectangles = []
            for rect in pot_optimal:
                # Check if rectangle is large enough to divide
                if rect.diameter < self.size_tol or np.prod(rect.side_lengths) < self.vol_tol:
                    new_rectangles.append(rect)
                else:
                    divided_rects = self._divide_rectangle(rect)
                    new_rectangles.extend(divided_rects)
            
            # Keep non-potentially optimal rectangles
            non_pot_optimal = [r for r in self.rectangles if r not in pot_optimal]
            self.rectangles = non_pot_optimal + new_rectangles
            
            # Check convergence
            if self.feval_count > self.min_feval:
                # Check improvement
                prev_best = getattr(self, '_prev_best', self.best_value + 2 * self.eps)
                if abs(prev_best - self.best_value) < self.eps:
                    return {
                        'x': self.best_x,
                        'fun': self.best_value,
                        'nit': self.iteration_count,
                        'nfev': self.feval_count,
                        'success': True,
                        'message': f'Converged (|f_change| < {self.eps})'
                    }
            
            self._prev_best = self.best_value
        
        return {
            'x': self.best_x,
            'fun': self.best_value,
            'nit': self.iteration_count,
            'nfev': self.feval_count,
            'success': False,
            'message': 'Maximum number of iterations reached'
        }


# Example usage and test functions
def test_function_1(x: np.ndarray) -> float:
    """
    Sphere function (convex, single minimum at origin).
    f(x) = sum(x_i^2)
    """
    return np.sum(x**2)

def test_function_2(x: np.ndarray) -> float:
    """
    Rastrigin function (highly multimodal).
    f(x) = 10n + sum(x_i^2 - 10*cos(2π*x_i))
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def test_function_3(x: np.ndarray) -> float:
    """
    Ackley function (multimodal with many local minima).
    f(x) = -20*exp(-0.2*sqrt(0.5*sum(x_i^2))) - 
           exp(0.5*sum(cos(2π*x_i))) + 20 + exp(1)
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.exp(1)

def test_function_4(x: np.ndarray) -> float:
    """
    Rosenbrock function (banana-shaped valley).
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


if __name__ == "__main__":
    # Example 1: Minimize Sphere function in 2D
    print("=" * 60)
    print("Example 1: Minimizing Sphere function (2D)")
    print("=" * 60)
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    optimizer = DIRECTOptimizer(test_function_1, bounds, max_iter=100, max_feval=1000)
    result = optimizer.optimize()
    print(f"Best solution: {result['x']}")
    print(f"Best value: {result['fun']:.8f}")
    print(f"Iterations: {result['nit']}")
    print(f"Function evaluations: {result['nfev']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    
    # Example 2: Minimize Rastrigin function in 3D
    print("\n" + "=" * 60)
    print("Example 2: Minimizing Rastrigin function (3D)")
    print("=" * 60)
    bounds = [(-5.12, 5.12) for _ in range(3)]
    optimizer = DIRECTOptimizer(test_function_2, bounds, max_iter=100, max_feval=1000)
    result = optimizer.optimize()
    print(f"Best solution: {result['x']}")
    print(f"Best value: {result['fun']:.8f}")
    print(f"Iterations: {result['nit']}")
    print(f"Function evaluations: {result['nfev']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    
    # Example 3: Minimize Ackley function (3D)
    print("\n" + "=" * 60)
    print("Example 3: Minimizing Ackley function (3D)")
    print("=" * 60)
    bounds = [(-30, 30) for _ in range(3)]
    optimizer = DIRECTOptimizer(test_function_3, bounds, max_iter=100, max_feval=1000)
    result = optimizer.optimize()
    print(f"Best solution: {result['x']}")
    print(f"Best value: {result['fun']:.8f}")
    print(f"Iterations: {result['nit']}")
    print(f"Function evaluations: {result['nfev']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")

    # Example 4: Minimize Rosenbrock function in 2D
    print("\n" + "=" * 60)
    print("Example 4: Minimizing Rosenbrock function (2D)")
    print("=" * 60)
    bounds = [(-2.0, 1.0), (-1.0, 2.0)]
    optimizer = DIRECTOptimizer(test_function_4, bounds, max_iter=500, max_feval=50000, min_feval=1000)
    result = optimizer.optimize()
    print(f"Best solution: {result['x']}")
    print(f"Best value: {result['fun']:.8f}")
    print(f"Iterations: {result['nit']}")
    print(f"Function evaluations: {result['nfev']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")


    