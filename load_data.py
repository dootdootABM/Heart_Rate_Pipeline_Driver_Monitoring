import pandas as pd
import sys # Import sys to check Python version
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    print(f"[INFO] Attempting to load data from: {file_path}")
    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path)
        print("[INFO] Data loaded successfully.")

        # --- Basic Data Verification ---
        print("\n--- Data Info ---")
        df.info() # Prints column names, non-null counts, and data types

        print("\n--- First 5 Rows ---")
        print(df.head()) # Shows the first few rows of your data

        print("\n--- Basic Statistics (for numeric columns) ---")
        print(df.describe()) # Shows count, mean, std dev, min, max, etc.

        return df

    except FileNotFoundError:
        print(f"[ERROR] File not found at the specified path.")
        print("Please ensure the path is correct and the file exists.")
        return None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The CSV file appears to be empty.")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading the data: {e}")
        return None

# --- VISUALIZATION FUNCTIONS (NOW SAVE TO FILE) ---

def plot_correlation_heatmap(df, numeric_cols):
    """
    Plots and saves a heatmap of the correlation matrix for numerical columns.
    """
    if not numeric_cols.empty:
        print("\n[INFO] Generating Correlation Heatmap...")
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        try:
            plt.savefig('correlation_heatmap.png')
            print("[INFO] Saved plot to 'correlation_heatmap.png'")
        except Exception as e:
            print(f"[ERROR] Could not save 'correlation_heatmap.png': {e}")
        plt.close() # Close the plot figure to free memory
    else:
        print("[INFO] No numerical columns found to plot correlation heatmap.")

def plot_numerical_distributions(df, numeric_cols):
    """
    Plots and saves histograms for all numerical columns.
    """
    if not numeric_cols.empty:
        print(f"[INFO] Generating distributions for {len(numeric_cols)} numerical features...")
        num_plots = len(numeric_cols)
        num_cols_grid = 3
        num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
        
        fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 5 * num_rows_grid))
        axes = axes.flatten() 

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        try:
            plt.savefig('numerical_distributions.png')
            print("[INFO] Saved plot to 'numerical_distributions.png'")
        except Exception as e:
            print(f"[ERROR] Could not save 'numerical_distributions.png': {e}")
        plt.close()
    else:
        print("[INFO] No numerical columns found to plot distributions.")

def plot_numerical_boxplots(df, numeric_cols):
    """
    Plots and saves box plots for all numerical columns.
    """
    if not numeric_cols.empty:
        print(f"[INFO] Generating box plots for {len(numeric_cols)} numerical features...")
        num_plots = len(numeric_cols)
        num_cols_grid = 3
        num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
        
        fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(15, 5 * num_rows_grid))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=df[col], ax=axes[i])
            axes[i].set_title(f'Box Plot of {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        try:
            plt.savefig('numerical_boxplots.png')
            print("[INFO] Saved plot to 'numerical_boxplots.png'")
        except Exception as e:
            print(f"[ERROR] Could not save 'numerical_boxplots.png': {e}")
        plt.close()
    else:
        print("[INFO] No numerical columns found to plot boxplots.")

def plot_categorical_counts(df, categorical_cols):
    """
    Plots and saves count plots (bar charts) for categorical columns.
    """
    if not categorical_cols.empty:
        print(f"[INFO] Generating count plots for {len(categorical_cols)} categorical features...")
        num_plots = len(categorical_cols)
        num_cols_grid = 2
        num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
        
        fig, axes = plt.subplots(nrows=num_rows_grid, ncols=num_cols_grid, figsize=(18, 7 * num_rows_grid))
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            if df[col].nunique() > 50:
                top_20 = df[col].value_counts().nlargest(20).index
                sns.countplot(y=col, data=df[df[col].isin(top_20)], order=top_20, ax=ax)
                ax.set_title(f'Top 20 Counts for {col} (High Cardinality)')
            else:
                sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
                ax.set_title(f'Counts for {col}')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        try:
            plt.savefig('categorical_counts.png')
            print("[INFO] Saved plot to 'categorical_counts.png'")
        except Exception as e:
            print(f"[ERROR] Could not save 'categorical_counts.png': {e}")
        plt.close()
    else:
        print("[INFO] No categorical columns found to plot counts.")

def plot_pairplot(df, numeric_cols):
    """
    Plots and saves a pairplot for (a sample of) numerical features.
    """
    if not numeric_cols.empty:
        print("[INFO] Generating Pairplot... (this may take a moment)")
        
        if len(numeric_cols) > 5:
            print(f"[INFO] Too many numerical cols ({len(numeric_cols)}). Plotting pairplot for a sample of 5.")
            cols_to_plot = numeric_cols.sample(5)
        else:
            cols_to_plot = numeric_cols
        
        # Create the pairplot
        g = sns.pairplot(df[cols_to_plot], corner=True)
        g.fig.suptitle('Pairplot of Numerical Features', y=1.02)
        
        try:
            # Save the figure from the PairGrid object
            g.savefig('pairplot.png')
            print("[INFO] Saved plot to 'pairplot.png'")
        except Exception as e:
            print(f"[ERROR] Could not save 'pairplot.png': {e}")
        plt.close()
    else:
        print("[INFO] No numerical columns found to plot pairplot.")


# --- Main execution block ---
if __name__ == "__main__":
    # Print Python and Pandas versions for debugging
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")

    # --- Use the specific file path you provided ---
    csv_file_path = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\drowsiness_dataset.csv'

    # Call the function to load the data
    data_frame = load_data(csv_file_path)

    if data_frame is not None:
        print(f"\n[SUCCESS] Loaded DataFrame with {len(data_frame)} rows and {len(data_frame.columns)} columns.")
        
        print("\n--- Starting Data Visualization (Saving to PNG) ---")
        
        # Identify column types
        numeric_cols = data_frame.select_dtypes(include=['number']).columns
        categorical_cols = data_frame.select_dtypes(include=['object', 'category']).columns

        # --- Call the plotting functions ---
        
        # 1. Correlation Heatmap
        plot_correlation_heatmap(data_frame, numeric_cols)
        
        # 2. Numerical Distributions
        plot_numerical_distributions(data_frame, numeric_cols)

        # 3. Numerical Box Plots
        plot_numerical_boxplots(data_frame, numeric_cols)
        
        # 4. Categorical Counts
        plot_categorical_counts(data_frame, categorical_cols)
        
        # 5. Pairplot
        plot_pairplot(data_frame, numeric_cols)
        
        # --- [CHANGED] No plt.show() needed ---
        print("\n[INFO] All plots saved to .png files in the script directory.")

    else:
        print("\n[FAILURE] Data loading failed. Please check the errors above.")