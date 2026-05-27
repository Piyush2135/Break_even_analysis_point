import numpy as np
import pandas as pd

# ================= CONSTANTS =================
DEFAULT_RANDOM_SEED = 42
DEFAULT_DATA_POINTS = 50
PREDICTION_FUTURE_DAYS = 10

# ================= DATA GENERATION =================
def generate_data(num_days=DEFAULT_DATA_POINTS, seed=DEFAULT_RANDOM_SEED):
    """
    Generate synthetic demand data for analysis.
    
    Args:
        num_days (int): Number of data points to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with 'Day' and 'Demand' columns
    """
    np.random.seed(seed)
    days = np.arange(1, num_days + 1)
    demand = 200 + 2 * days + np.random.normal(0, 20, len(days))
    
    return pd.DataFrame({
        "Day": days,
        "Demand": np.maximum(demand, 0)  # Ensure non-negative demand
    })

# ================= ML MODEL =================
def linear_regression(X, y):
    """
    Perform simple linear regression using least squares method.
    
    Formula: y = mx + c
    where:
        m = slope (rate of change)
        c = y-intercept (baseline value)
    
    Args:
        X (array-like): Independent variable (days/time)
        y (array-like): Dependent variable (demand)
    
    Returns:
        tuple: (slope m, intercept c)
    """
    if len(X) < 2:
        raise ValueError("At least 2 data points required for regression")
    
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    
    numerator = np.sum((X - mean_x) * (y - mean_y))
    denominator = np.sum((X - mean_x) ** 2)
    
    if denominator == 0:
        raise ValueError("Cannot calculate regression: denominator is zero")
    
    m = numerator / denominator
    c = mean_y - m * mean_x
    
    return m, c

# ================= PREDICTION =================
def predict_demand(m, c, start_day=None, num_days=PREDICTION_FUTURE_DAYS):
    """
    Predict future demand based on linear regression model.
    
    Args:
        m (float): Slope from regression model
        c (float): Intercept from regression model
        start_day (int): Starting day for prediction (default: 51)
        num_days (int): Number of days to predict into the future
    
    Returns:
        tuple: (future_days array, predicted_demand array)
    """
    if start_day is None:
        start_day = DEFAULT_DATA_POINTS + 1
    
    future_days = np.arange(start_day, start_day + num_days)
    predicted = m * future_days + c
    
    return future_days, np.maximum(predicted, 0)  # Ensure non-negative predictions

# ================= BREAK EVEN =================
def break_even(FC, SP, VC):
    """
    Calculate break-even point in units.
    
    Break-even occurs when: Total Revenue = Total Cost
    Units = Fixed Cost / Contribution Margin
    where Contribution Margin = Selling Price - Variable Cost
    
    Args:
        FC (float): Fixed Cost
        SP (float): Selling Price per unit
        VC (float): Variable Cost per unit
    
    Returns:
        float: Break-even point in units
    
    Raises:
        ValueError: If SP <= VC (invalid margin)
    """
    contribution_margin = SP - VC
    
    if contribution_margin <= 0:
        raise ValueError("Selling Price must be greater than Variable Cost")
    
    return FC / contribution_margin

# ================= PROFIT =================
def calculate_profit(SP, VC, FC, demand):
    """
    Calculate total profit based on demand.
    
    Formula: Profit = (SP - VC) * Demand - FC
    where:
        (SP - VC) = Contribution Margin per unit
        Demand = Total units sold
        FC = Fixed Cost
    
    Args:
        SP (float): Selling Price per unit
        VC (float): Variable Cost per unit
        FC (float): Fixed Cost
        demand (float or array): Demand in units
    
    Returns:
        float or array: Profit (same type as demand input)
    """
    return (SP - VC) * demand - FC

# ================= OPTIMAL PRICE =================
def optimal_price(FC, VC, target_profit, demand):
    """
    Calculate optimal selling price to achieve target profit.
    
    Formula: SP = VC + (FC + Target Profit) / Demand
    
    Args:
        FC (float): Fixed Cost
        VC (float): Variable Cost per unit
        target_profit (float): Target profit to achieve
        demand (float): Average or expected demand in units
    
    Returns:
        float: Recommended selling price
    
    Raises:
        ValueError: If demand is zero or negative
    """
    if demand <= 0:
        raise ValueError("Demand must be positive for price optimization")
    
    optimal_sp = VC + (FC + target_profit) / demand
    
    return max(optimal_sp, VC)  # Ensure price doesn't go below variable cost

# ================= MARGIN ANALYSIS =================
def calculate_margin_percent(SP, VC):
    """
    Calculate profit margin percentage.
    
    Formula: Margin % = ((SP - VC) / SP) * 100
    
    Args:
        SP (float): Selling Price
        VC (float): Variable Cost
    
    Returns:
        float: Margin percentage
    """
    if SP <= 0:
        raise ValueError("Selling Price must be positive")
    
    return ((SP - VC) / SP) * 100

# ================= SENSITIVITY ANALYSIS =================
def sensitivity_analysis(FC, SP, VC, demand, variable="price", variation_range=(-20, 20)):
    """
    Perform sensitivity analysis on profit for a given variable.
    
    Args:
        FC (float): Fixed Cost
        SP (float): Selling Price
        VC (float): Variable Cost
        demand (float): Demand
        variable (str): 'price', 'cost', or 'demand'
        variation_range (tuple): Percentage variation range (min, max)
    
    Returns:
        dict: Contains 'values' and 'profits' arrays for plotting
    """
    variations = np.linspace(variation_range[0], variation_range[1], 50)
    profits = []
    
    for var in variations:
        factor = 1 + (var / 100)
        
        if variable == "price":
            profit = calculate_profit(SP * factor, VC, FC, demand)
        elif variable == "cost":
            profit = calculate_profit(SP, VC * factor, FC, demand)
        elif variable == "demand":
            profit = calculate_profit(SP, VC, FC, demand * factor)
        else:
            raise ValueError(f"Unknown variable: {variable}")
        
        profits.append(profit)
    
    return {"variations": variations, "profits": np.array(profits)}
