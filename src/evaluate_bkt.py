import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
ML_TRAINING_DIR = SRC_DIR.parent
MODELS_DIR = ML_TRAINING_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def simulate_bkt_convergence():
    np.random.seed(42)  # For reproducible simulation results
    
    # 1. Mock 100 students
    num_students = 100
    num_interactions = 15
    
    # True hidden mastery level for a skill (between 0.1 and 0.9)
    true_mastery = np.random.uniform(0.1, 0.9, size=num_students)
    
    # Initialize belief states at Alpha=2, Beta=2
    alpha = np.full(num_students, 2.0)
    beta = np.full(num_students, 2.0)
    
    rmse_history = []
    
    # 2. Simulate 15 interactions
    for step in range(num_interactions):
        # Probability of correct response is based on True Mastery plus slight Gaussian noise
        prob_correct = np.clip(true_mastery + np.random.normal(0, 0.05, size=num_students), 0.0, 1.0)
        
        # Simulate binomial trial
        correct = np.random.rand(num_students) < prob_correct
        
        # Bayesian update
        alpha += correct.astype(float)
        beta += (~correct).astype(float)
        
        # Recalculate Expected Mastery E[K]
        predicted_mastery = alpha / (alpha + beta)
        
        # Compute RMSE for this interaction step
        rmse = np.sqrt(np.mean((true_mastery - predicted_mastery) ** 2))
        rmse_history.append(rmse)
        
        print(f"Interaction Step {step + 1:02d}: Aggregate RMSE = {rmse:.4f}")
        
    # 3. Plot convergence curve
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6), dpi=300)
    
    steps = np.arange(1, num_interactions + 1)
    plt.plot(steps, rmse_history, marker='o', color='#00f0ff', linewidth=2.5, markersize=6, label='BKT E[K] Estimate')
    
    # Style plot professionally
    plt.title("Bayesian Parameter Convergence Rate", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Interaction / Sequence Length", fontsize=11, fontweight='bold')
    plt.ylabel("Root Mean Square Error (RMSE)", fontsize=11, fontweight='bold')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight final error convergence
    plt.axhline(y=rmse_history[-1], color='#f43f5e', linestyle=':', alpha=0.8, label=f"Converged Error ({rmse_history[-1]:.3f})")
    plt.legend(frameon=True, facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    # Customize dark mode vibes
    plt.gcf().patch.set_facecolor('#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    
    output_path = MODELS_DIR / "bkt_rmse_convergence.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0f172a')
    plt.close()
    
    print(f"[BKT Eval] Saved convergence graph to {output_path}")

if __name__ == "__main__":
    simulate_bkt_convergence()
