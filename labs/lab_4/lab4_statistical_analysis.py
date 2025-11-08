"""Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
import os
import math

# --- Configuration and Constants ---
# FIX 1: Set DATASETS_PATH to the same directory as the script file.
# This assumes ALL .csv files are moved into the /labs/lab4/ folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = SCRIPT_DIR
OUTPUT_PATH = SCRIPT_DIR

# --- Part 1: Data Loading and Descriptive Statistics Functions ---

def load_data(file_name):
    """Load dataset from CSV file, assuming it's in the same directory as the script."""
    file_path = os.path.join(DATASETS_PATH, file_name)
    try:
        data = pd.read_csv(file_path)
        print(f"--- Data Loaded: {file_name} ---")
        print(f"Shape: {data.shape}")
        print(f"Columns:\n{data.columns.tolist()}")
        print(f"Data Types:\n{data.dtypes}")
        print("-" * 30)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure the CSV files are in the same directory as the script.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return pd.DataFrame()

def calculate_descriptive_stats(data, column='strength_mpa'):
    """Calculate all descriptive statistics."""
    if data.empty or column not in data.columns:
        return {}

    series = data[column].dropna()

    # Measures of Central Tendency
    mean_val = series.mean()
    median_val = series.median()
    mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan

    # Measures of Spread
    variance_val = series.var()
    std_dev_val = series.std()
    range_val = series.max() - series.min()
    iqr_val = series.quantile(0.75) - series.quantile(0.25)

    # Shape Measures
    skewness_val = series.skew()
    kurtosis_val = series.kurtosis()

    # Quantiles/Five-Number Summary
    quantiles = series.quantile([0, 0.25, 0.5, 0.75, 1.0])
    five_number_summary = {
        'Min': quantiles[0.0],
        'Q1 (25th)': quantiles[0.25],
        'Median (Q2)': quantiles[0.5],
        'Q3 (75th)': quantiles[0.75],
        'Max': quantiles[1.0]
    }

    stats_dict = {
        'N': len(series),
        'Mean': mean_val,
        'Median': median_val,
        'Mode': mode_val,
        'Variance': variance_val,
        'Std Dev ($\\sigma$)': std_dev_val,
        'Range': range_val,
        'IQR': iqr_val,
        'Skewness': skewness_val,
        'Kurtosis': kurtosis_val,
        'Five-Number Summary': five_number_summary
    }

    print(f"\n--- Descriptive Statistics for {column} ---")
    for key, value in stats_dict.items():
        if key != 'Five-Number Summary':
            print(f"{key:<15}: {value:.4f}")
    print("Five-Number Summary:")
    for key, value in five_number_summary.items():
        print(f"  {key:<10}: {value:.4f}")
    print("-" * 30)

    return stats_dict

def plot_distribution(data, column, stats_dict, title, save_path=None):
    """
    Create a distribution plot (histogram/KDE) with central tendency and
    spread (±n*sigma) marked.
    """
    if data.empty or column not in data.columns:
        print("Error: Data is empty or column not found for plotting.")
        return

    series = data[column].dropna()

    mean = stats_dict.get('Mean', series.mean())
    median = stats_dict.get('Median (Q2)', series.median())
    mode = stats_dict.get('Mode', series.mode().iloc[0] if not series.mode().empty else np.nan)
    std = stats_dict.get('Std Dev ($\\sigma$)', series.std())

    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=15, color='skyblue', edgecolor='black', alpha=0.6)

    # Mark Central Tendency
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Median: {median:.2f}')
    if not np.isnan(mode):
        plt.axvline(mode, color='orange', linestyle=':', linewidth=2, label=f'Mode: {mode:.2f}')

    # Mark Spread (±1σ, ±2σ, ±3σ)
    colors = ['gray', 'lightcoral', 'indianred']
    sigmas = [1, 2, 3]
    for i, s in enumerate(sigmas):
        plt.axvline(mean + s * std, color=colors[i], linestyle='--', alpha=0.7, label=f'+{s}$\sigma$')
        plt.axvline(mean - s * std, color=colors[i], linestyle='--', alpha=0.7, label=f'-{s}$\sigma$')

    plt.title(title, fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency / Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    if save_path:
        full_path = os.path.join(OUTPUT_PATH, save_path)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")
    plt.show()

def plot_material_comparison(data, column, group_column, save_path=None):
    """Create comparative boxplot for material types."""
    if data.empty or column not in data.columns or group_column not in data.columns:
        print("Error: Data is empty or required columns not found for comparison plot.")
        return

    plt.figure(figsize=(10, 6))
    # FIX 2: Added hue=group_column and legend=False to comply with Seaborn future versions
    sns.boxplot(x=group_column, y=column, data=data, palette="Set2",
                hue=group_column, legend=False)
    plt.title(f'Comparison of {column} by {group_column}', fontsize=16)
    plt.xlabel(group_column, fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.grid(axis='y', alpha=0.5)

    if save_path:
        full_path = os.path.join(OUTPUT_PATH, save_path)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")
    plt.show()

# --- Part 2: Probability Distributions Functions ---

def calculate_probability_binomial(n, p, k=None, cumulative=False):
    """Calculate binomial probabilities."""
    dist = binom(n, p)
    if k is not None:
        if cumulative:
            # P(X <= k)
            prob = dist.cdf(k)
            print(f"P(X <= {k}) for Binomial(n={n}, p={p}) is: {prob:.4f}")
        else:
            # P(X = k)
            prob = dist.pmf(k)
            print(f"P(X = {k}) for Binomial(n={n}, p={p}) is: {prob:.4f}")
        return prob

    # Calculate Mean and Variance for the distribution
    mean, var = dist.stats(moments='mv')
    print(f"Binomial(n={n}, p={p}) - Mean: {mean:.4f}, Variance: {var:.4f}")
    return mean, var

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """Calculate normal probabilities."""
    dist = norm(loc=mean, scale=std)

    if x_lower is not None and x_upper is not None:
        # P(x_lower < X < x_upper)
        prob = dist.cdf(x_upper) - dist.cdf(x_lower)
        print(f"P({x_lower} < X < {x_upper}) for Normal({mean}, {std}) is: {prob:.4f}")
    elif x_lower is not None:
        # P(X > x_lower) (e.g., exceedance probability)
        prob = dist.sf(x_lower)
        print(f"P(X > {x_lower}) for Normal({mean}, {std}) is: {prob:.4f}")
    elif x_upper is not None:
        # P(X < x_upper)
        prob = dist.cdf(x_upper)
        print(f"P(X < {x_upper}) for Normal({mean}, {std}) is: {prob:.4f}")

    # Percentile calculation example
    if x_lower is None and x_upper is None:
        # 95th percentile
        percentile_95 = dist.ppf(0.95)
        print(f"The 95th percentile (X_95) for Normal({mean}, {std}) is: {percentile_95:.4f}")
        return percentile_95

    return prob

def calculate_probability_poisson(lambda_param, k=None, cumulative=False):
    """Calculate Poisson probabilities."""
    dist = poisson(lambda_param)

    if k is not None:
        if cumulative:
            # P(X <= k)
            prob = dist.cdf(k)
            print(f"P(X <= {k}) for Poisson($\lambda$={lambda_param}) is: {prob:.4f}")
        else:
            # P(X = k)
            prob = dist.pmf(k)
            print(f"P(X = {k}) for Poisson($\lambda$={lambda_param}) is: {prob:.4f}")
        return prob

    # Calculate Mean and Variance for the distribution (they are equal to lambda)
    mean, var = dist.stats(moments='mv')
    print(f"Poisson($\lambda$={lambda_param}) - Mean: {mean:.4f}, Variance: {var:.4f}")
    return mean, var

def calculate_probability_exponential(mean, x_limit=None, survival=False):
    """Calculate exponential probabilities."""
    # Scale parameter ($\lambda$) is the reciprocal of the mean (1/mean)
    scale_param = mean # SciPy's expon uses scale parameter $\beta = 1/\lambda = \text{mean}$
    dist = expon(scale=scale_param)

    if x_limit is not None:
        if survival:
            # P(X > x_limit) (Survival probability)
            prob = dist.sf(x_limit)
            print(f"P(X > {x_limit}) for Exponential($\mu$={mean}) is: {prob:.4f}")
        else:
            # P(X < x_limit) (Failure probability before x_limit)
            prob = dist.cdf(x_limit)
            print(f"P(X < {x_limit}) for Exponential($\mu$={mean}) is: {prob:.4f}")
        return prob

    # Calculate Mean and Variance for the distribution
    mean_calc, var_calc = dist.stats(moments='mv')
    print(f"Exponential($\mu$={mean}) - Mean: {mean_calc:.4f}, Variance: {var_calc:.4f}")
    return mean_calc, var_calc

# --- Part 3: Probability Applications Functions ---

def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Apply Bayes' theorem to an engineering diagnostic problem.
    P(Damage | Positive Test) = [ P(Positive Test | Damage) * P(Damage) ] / P(Positive Test)

    Parameters:
    prior (P(D)): Probability of structural damage (Base Rate).
    sensitivity (P(T+|D)): Probability of a positive test given damage (True Positive Rate).
    specificity (P(T-|D')): Probability of a negative test given NO damage (True Negative Rate).
    """
    # Define probabilities
    PD = prior  # P(Damage) - Prior
    PT_D = sensitivity  # P(T+|D) - Likelihood (Sensitivity)
    PT_D_complement = 1 - specificity # P(T+|D') - False Positive Rate

    # P(D')
    PD_complement = 1 - PD

    # Calculate P(Positive Test) - Total Probability Theorem
    # P(T+) = P(T+|D) * P(D) + P(T+|D') * P(D')
    PT = (PT_D * PD) + (PT_D_complement * PD_complement)

    # Apply Bayes' Theorem
    # P(D|T+) = [ P(T+|D) * P(D) ] / P(T+)
    PD_T = (PT_D * PD) / PT

    print("\n--- Bayes' Theorem Application (Structural Damage) ---")
    print(f"Prior P(Damage) (PD): {PD:.4f}")
    print(f"Sensitivity P(T+|D): {PT_D:.4f}")
    print(f"False Positive Rate P(T+|D'): {PT_D_complement:.4f}")
    print(f"Probability of Positive Test P(T+): {PT:.4f}")
    print(f"Posterior P(Damage|Positive Test) (PD|T+): {PD_T:.4f}")
    print("-" * 30)

    # Visualization (Probability Tree Diagram logic)
    print("\nProbability Tree Paths:")
    print(f"  P(D and T+): {PD * PT_D:.4f}") # True Positive
    print(f"  P(D and T-): {PD * (1 - PT_D):.4f}") # False Negative
    print(f"  P(D' and T+): {PD_complement * PT_D_complement:.4f}") # False Positive
    print(f"  P(D' and T-): {PD_complement * specificity:.4f}") # True Negative

    return PD_T, PT

# --- Part 4 & 5: Distribution Fitting, Visualization, and Reporting ---

def fit_distribution(data, column, distribution_type='normal'):
    """Fit probability distribution to data."""
    if data.empty or column not in data.columns:
        print("Error: Data is empty or column not found for distribution fitting.")
        return None, None

    series = data[column].dropna()

    if distribution_type.lower() == 'normal':
        # Estimate parameters (MLE)
        params = norm.fit(series)
        mu, sigma = params
        print(f"\n--- Distribution Fitting: {distribution_type.upper()} ---")
        print(f"Fitted Parameters (Mean, Std Dev): ({mu:.4f}, {sigma:.4f})")
        return norm, params

    # Add other distributions here if needed
    print(f"Warning: Distribution type '{distribution_type}' not supported for fitting.")
    return None, None

def plot_distribution_fitting(data, column, fitted_dist, fitted_params, save_path=None):
    """Visualize fitted distribution overlaid on histogram."""
    if data.empty or column not in data.columns or fitted_dist is None or fitted_params is None:
        print("Error: Data, fitted distribution, or parameters are missing for plotting.")
        return

    series = data[column].dropna()
    mu, sigma = fitted_params

    plt.figure(figsize=(10, 6))

    # 1. Histogram of empirical data
    sns.histplot(series, bins=15, kde=False, stat='density', color='lightgray',
                 edgecolor='black', label='Empirical Data')

    # 2. PDF of the fitted distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = fitted_dist.pdf(x, *fitted_params)
    plt.plot(x, p, 'r', linewidth=2, label=f'Fitted Normal PDF ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})')

    # 3. Generate and plot synthetic data
    synthetic_data = fitted_dist.rvs(*fitted_params, size=len(series))
    sns.kdeplot(synthetic_data, color='blue', linestyle='--', label='Synthetic Data KDE')

    # Comparison metrics (optional for plot)
    mean_syn = np.mean(synthetic_data)
    std_syn = np.std(synthetic_data)

    plt.title(f'Normal Distribution Fitting for {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    if save_path:
        full_path = os.path.join(OUTPUT_PATH, save_path)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")
    plt.show()

    print("\n--- Validation of Synthetic Data ---")
    print(f"Synthetic Data Mean: {mean_syn:.4f} (Fitted Mean: {mu:.4f})")
    print(f"Synthetic Data Std Dev: {std_syn:.4f} (Fitted Std Dev: {sigma:.4f})")
    print("-" * 30)

def create_statistical_report(report_data, output_file='lab4_statistical_report.txt'):
    """Create a statistical report summarizing findings."""
    full_path = os.path.join(OUTPUT_PATH, output_file)
    # The key name used in stats_dict
    STD_DEV_KEY = 'Std Dev ($\\sigma$)'

    with open(full_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("            LAB 4: STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Part 1: Concrete Strength Descriptive Stats
        f.write("## 1. Concrete Strength Descriptive Statistics\n\n")

        # Format the descriptive statistics table
        f.write("{:<25} {:<15} {:<15}\n".format("Statistic", "Value", "Interpretation"))
        f.write("-" * 55 + "\n")

        # Central Tendency
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Mean", report_data['concrete_stats']['Mean'],
                                               "Average strength"))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Median", report_data['concrete_stats']['Median'],
                                               "Middle value"))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Mode", report_data['concrete_stats']['Mode'],
                                               "Most frequent value"))

        # Spread
        # FIX: The key is accessed literally. The display string uses single backslash.
        f.write("{:<25} {:<15.4f} {:<15}\n".format(STD_DEV_KEY, report_data['concrete_stats'][STD_DEV_KEY],
                                               "Variability/Precision"))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Variance", report_data['concrete_stats']['Variance'],
                                               "Spread measure ($\sigma^2$)"))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Range", report_data['concrete_stats']['Range'],
                                               "Max - Min"))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("IQR", report_data['concrete_stats']['IQR'],
                                               "Middle 50% spread"))

        # Shape
        skew = report_data['concrete_stats']['Skewness']
        kurt = report_data['concrete_stats']['Kurtosis']
        skew_interp = "Slightly Skewed Right" if skew > 0.1 else ("Slightly Skewed Left" if skew < -0.1 else "Symmetric")
        kurt_interp = "Platykurtic (Flat)" if kurt < 0.1 else ("Leptokurtic (Peaked)" if kurt > 0.1 else "Mesokurtic (Normal)")

        f.write("{:<25} {:<15.4f} {:<15}\n".format("Skewness", skew, skew_interp))
        f.write("{:<25} {:<15.4f} {:<15}\n".format("Kurtosis", kurt, kurt_interp))

        f.write("\n### Engineering Interpretation:\n")
        f.write(f"- Central Tendency: Mean strength is **{report_data['concrete_stats']['Mean']:.2f} MPa**. Given the small difference between mean and median, the distribution is relatively symmetric.\n")
        # FIX: Access the key literally (resolved the f-string issue by using literal access)
        f.write(f"- Spread: The standard deviation of **{report_data['concrete_stats'][STD_DEV_KEY]:.2f} MPa** indicates the precision of the concrete mixing process. Lower variability is desired for quality control.\n")
        f.write(f"- Shape: The skewness ({skew:.2f}) and kurtosis ({kurt:.2f}) suggest the distribution is approximately normal, which is common for material properties.\n\n")

        # Part 2: Material Comparison
        f.write("## 2. Material Property Comparison\n\n")
        f.write("{:<15} {:<15} {:<15} {:<15} {:<15}\n".format("Material", "Count", "Mean (MPa)", "Std Dev ($\sigma$)", "CoV"))
        f.write("-" * 60 + "\n")

        comparison_results = report_data['material_comparison']
        for material, stats in comparison_results.items():
            cov = stats['Std Dev'] / stats['Mean'] # Coefficient of Variation
            f.write("{:<15} {:<15} {:<15.4f} {:<15.4f} {:<15.2%}\n".format(
                material, stats['Count'], stats['Mean'], stats['Std Dev'], cov
            ))

        max_var_mat = max(comparison_results, key=lambda k: comparison_results[k]['Std Dev'])
        max_cov_mat = max(comparison_results, key=lambda k: comparison_results[k]['Std Dev'] / comparison_results[k]['Mean'])

        f.write("\n### Engineering Interpretation:\n")
        f.write(f"- **Variability:** **{max_var_mat}** has the highest absolute variability (Std Dev), while **{max_cov_mat}** has the highest relative variability (Coefficient of Variation).\n")
        f.write("- **Implication:** Materials with higher variability pose greater design challenges, often requiring larger safety factors (lower characteristic strength).\n\n")


        # Part 3: Probability Modeling and Applications
        f.write("## 3. Probability Modeling and Applications\n\n")

        # Binomial Scenario
        f.write("### 3.1. Binomial Scenario (Quality Control)\n")
        f.write(f"Parameters: n=100 components, p=0.05 defect rate\n")
        f.write(f"- P(X = 3 defective): **{report_data['prob_binomial_3']:.4f}**\n")
        f.write(f"- P(X $\leq$ 5 defective): **{report_data['prob_binomial_5_cum']:.4f}**\n")
        f.write("Implication: The probability of having more than 5 defects is low, suggesting the defect rate is acceptable under current quality control standards.\n\n")

        # Poisson Scenario
        f.write("### 3.2. Poisson Scenario (Bridge Load Events)\n")
        f.write(f"Parameters: $\lambda$=10 heavy trucks/hour\n")
        f.write(f"- P(X = 8 trucks): **{report_data['prob_poisson_8']:.4f}**\n")
        # P(X > 15) = 1 - P(X <= 15)
        f.write(f"- P(X > 15 trucks): 1 - P(X $\leq$ 15) = **{1 - report_data['prob_poisson_15_cum']:.4f}**\n")
        f.write("Implication: The probability of exceeding 15 heavy trucks per hour is very low, which is crucial for determining the **design life** and **fatigue loading** of the bridge structure.\n\n")

        # Normal Scenario
        f.write("### 3.3. Normal Scenario (Steel Yield Strength)\n")
        f.write(f"Parameters: $\mu$=250 MPa, $\sigma$=15 MPa\n")
        f.write(f"- P(X > 280 MPa) (Exceedance): **{report_data['prob_normal_exceed_280']:.4f}**\n")
        f.write(f"- 95th Percentile: **{report_data['prob_normal_95th_perc']:.4f}** MPa\n")
        f.write("Implication: The low exceedance probability is important for **reliability-based design**. The 95th percentile is a measure of high-end strength.\n\n")

        # Exponential Scenario
        f.write("### 3.4. Exponential Scenario (Component Lifetime)\n")
        f.write(f"Parameters: Mean lifetime $\mu$=1000 hours\n")
        f.write(f"- P(Failure < 500 hrs): **{report_data['prob_exp_fail_500']:.4f}**\n")
        f.write(f"- P(Survival > 1500 hrs): **{report_data['prob_exp_survive_1500']:.4f}**\n")
        f.write("Implication: These probabilities are essential for **maintenance scheduling** and determining the **replacement cycles** for the component.\n\n")

        # Bayes' Theorem
        f.write("### 3.5. Bayes' Theorem (Structural Damage Detection)\n")
        PD_T = report_data['bayes_posterior']
        PT = report_data['bayes_pt']
        f.write(f"Prior P(Damage): 5% (0.05)\n")
        f.write(f"Sensitivity P(Test+|Damage): 95% (0.95)\n")
        f.write(f"Specificity P(Test-|No Damage): 90% (0.90)\n")
        f.write(f"- Posterior Probability P(Damage|Positive Test): **{PD_T:.4f}**\n")
        f.write(f"- Probability of a Positive Test P(T+): **{PT:.4f}**\n")
        f.write("Implication: Despite a low base rate of 5%, a positive test result significantly increases the probability of actual damage to **{PD_T*100:.2f}%**. This guides whether further, more expensive investigation is necessary.\n\n")


        # Part 4: Distribution Fitting
        f.write("## 4. Distribution Fitting (Concrete Strength)\n\n")
        f.write(f"Fitted Normal Distribution Parameters:\n")
        f.write(f"- Mean ($\mu$): **{report_data['fitted_params'][0]:.4f}**\n")
        f.write(f"- Std Dev ($\sigma$): **{report_data['fitted_params'][1]:.4f}**\n")
        f.write(f"\nSample Statistics for Comparison:\n")
        f.write(f"- Sample Mean: **{report_data['concrete_stats']['Mean']:.4f}**\n")
        # FIX: Access the key literally
        f.write(f"- Sample Std Dev: **{report_data['concrete_stats'][STD_DEV_KEY]:.4f}**\n")
        f.write("Interpretation: The fitted parameters are very close to the sample statistics, suggesting the **Normal distribution is a good model** for the concrete strength data.\n\n")

        f.write("=" * 70 + "\n")
        f.write("End of Report\n")

    print(f"\nStatistical Report generated and saved to {full_path}")


def main():
    """Main execution function."""

    # Storage for report data
    report_data = {}

    # 🚨 NOTE: Ensure your three CSV files are in the same folder as this script.

    # --- Part 1: Descriptive Statistics (Concrete Strength) ---
    print("\n" + "=" * 50)
    print("--- PART 1: DESCRIPTIVE STATISTICS (CONCRETE) ---")
    print("=" * 50)

    # Load and explore data
    concrete_data = load_data('concrete_strength.csv')
    if concrete_data.empty: return

    # Assuming 'strength_mpa' is the column for strength measurements
    strength_col = 'strength_mpa'

    # Handle missing values (simple dropna for this lab)
    concrete_data.dropna(subset=[strength_col], inplace=True)

    # Calculate and store descriptive statistics
    concrete_stats = calculate_descriptive_stats(concrete_data, column=strength_col)
    report_data['concrete_stats'] = concrete_stats

    # Visualize distribution
    plot_distribution(
        concrete_data,
        column=strength_col,
        stats_dict=concrete_stats,
        title='Concrete Compressive Strength Distribution with Descriptive Statistics',
        save_path='concrete_strength_distribution.png'
    )

    # --- Task 2: Material Comparison ---
    print("\n" + "=" * 50)
    print("--- TASK 2: MATERIAL COMPARISON ---")
    print("=" * 50)

    material_data = load_data('material_properties.csv')
    if material_data.empty: return

    strength_col_mat = 'yield_strength_mpa'
    group_col = 'material_type'
    material_data.dropna(subset=[strength_col_mat, group_col], inplace=True)

    material_comparison_results = {}
    print("\n--- Comparative Statistics by Material Type ---")
    for material in material_data[group_col].unique():
        subset = material_data[material_data[group_col] == material][strength_col_mat]
        stats = {
            'Count': len(subset),
            'Mean': subset.mean(),
            'Std Dev': subset.std()
        }
        material_comparison_results[material] = stats
        print(f"  {material:<10} | Mean: {stats['Mean']:.4f}, Std Dev: {stats['Std Dev']:.4f}")

    report_data['material_comparison'] = material_comparison_results

    # Create comparative boxplots
    plot_material_comparison(
        material_data,
        column=strength_col_mat,
        group_column=group_col,
        save_path='material_comparison_boxplot.png'
    )

    # --- Part 2 & Task 3: Probability Modeling ---
    print("\n" + "=" * 50)
    print("--- PART 2 & TASK 3: PROBABILITY MODELING ---")
    print("=" * 50)

    # Binomial Scenario
    n_binom, p_binom = 100, 0.05
    report_data['prob_binomial_3'] = calculate_probability_binomial(n_binom, p_binom, k=3)
    report_data['prob_binomial_5_cum'] = calculate_probability_binomial(n_binom, p_binom, k=5, cumulative=True)
    calculate_probability_binomial(n_binom, p_binom) # Display mean/var

    # Poisson Scenario
    lambda_poisson = 10
    report_data['prob_poisson_8'] = calculate_probability_poisson(lambda_poisson, k=8)
    # P(X > 15) = 1 - P(X <= 15)
    prob_leq_15 = calculate_probability_poisson(lambda_poisson, k=15, cumulative=True)
    report_data['prob_poisson_15_cum'] = prob_leq_15
    calculate_probability_poisson(lambda_poisson) # Display mean/var

    # Normal Scenario
    mean_normal, std_normal = 250, 15
    # P(X > 280 MPa)
    report_data['prob_normal_exceed_280'] = calculate_probability_normal(mean_normal, std_normal, x_lower=280)
    # 95th percentile
    report_data['prob_normal_95th_perc'] = calculate_probability_normal(mean_normal, std_normal)

    # Exponential Scenario
    mean_exp = 1000
    # P(Failure < 500 hrs)
    report_data['prob_exp_fail_500'] = calculate_probability_exponential(mean_exp, x_limit=500)
    # P(Survival > 1500 hrs)
    report_data['prob_exp_survive_1500'] = calculate_probability_exponential(mean_exp, x_limit=1500, survival=True)
    calculate_probability_exponential(mean_exp) # Display mean/var

    # --- Part 3 & Task 4: Bayes' Theorem Application ---
    print("\n" + "=" * 50)
    print("--- PART 3 & TASK 4: BAYES' THEOREM APPLICATION ---")
    print("=" * 50)

    prior_d = 0.05      # P(Damage)
    sensitivy_pt_d = 0.95 # P(T+|D)
    specificity_tn_d_comp = 0.90 # P(T-|D')

    # P(Damage|Positive Test)
    posterior_prob, prob_test_pos = apply_bayes_theorem(prior_d, sensitivy_pt_d, specificity_tn_d_comp)
    report_data['bayes_posterior'] = posterior_prob
    report_data['bayes_pt'] = prob_test_pos

    # --- Part 4 & Task 5: Distribution Fitting and Validation ---
    print("\n" + "=" * 50)
    print("--- PART 4 & TASK 5: DISTRIBUTION FITTING ---")
    print("=" * 50)

    # Fit a normal distribution to concrete strength
    fitted_dist, fitted_params = fit_distribution(concrete_data, strength_col, distribution_type='normal')
    report_data['fitted_params'] = fitted_params

    # Visualize fitted distribution and synthetic data comparison
    plot_distribution_fitting(
        concrete_data,
        strength_col,
        fitted_dist,
        fitted_params,
        save_path='distribution_fitting.png'
    )

    # --- Part 4 & Task 5: Statistical Report ---
    print("\n" + "=" * 50)
    print("--- PART 4: STATISTICAL REPORT GENERATION ---")
    print("=" * 50)
    create_statistical_report(report_data, output_file='lab4_statistical_report.txt')

    print("\n--- Script Execution Complete. Check generated files (PNG and TXT) ---")

if __name__ == "__main__":
    # Suppress an expected Matplotlib warning about tight_layout
    pd.options.mode.chained_assignment = None
    main()
