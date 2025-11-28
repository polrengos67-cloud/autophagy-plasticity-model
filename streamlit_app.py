import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from io import BytesIO
import json
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Neural Plasticity Model",
    page_icon="🧠",
    layout="wide"
)

def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    """Enhanced model of autophagy-plasticity dynamics"""
    S, D, A, E = y
    
    # Apply bounds to state variables
    S = max(0, S)
    D = max(0, min(1.0, D))
    A = max(0.1, A)
    E = max(0.1, min(5.0, E))
    
    # Autophagic Flux
    target_A = A0 / (1 + sigma)
    dAdt = 0.2 * (target_A - A)
    
    # Energy Dynamics
    production = k_atp * (1 - 0.5 * D)
    consumption = k_energy * S
    dEdt = production - consumption
    
    # Synaptic Strength
    maintenance = k_maintain * S * E
    dSdt = -k_decay * S - k_damage * D * S + P_val + maintenance
    
    # Synaptic Damage
    clearance = alpha * A * D
    dDdt = beta - clearance
    
    return [dSdt, dDdt, dAdt, dEdt]

def run_simulation(params):
    """Run simulation with stimulus protocol"""
    y0 = [1.0, 0.05, params['A0'], 1.0]
    
    t_stim = params['stim_time']
    t_end = params['t_end']
    stim_strength = params['stim_strength']
    
    # Pre-stimulus
    t_pre = np.linspace(0, t_stim, 300)
    sol_pre = solve_ivp(
        lambda t, y: model(t, y, **{k: v for k, v in params.items() 
                                   if k not in ['stim_time', 't_end', 'stim_strength']}, P_val=0),
        [0, t_stim], y0, t_eval=t_pre, method='LSODA'
    )
    
    # Apply stimulus
    y_stim = sol_pre.y[:, -1].copy()
    y_stim[0] += stim_strength
    
    # Post-stimulus
    t_post = np.linspace(t_stim, t_end, 1000)
    sol_post = solve_ivp(
        lambda t, y: model(t, y, **{k: v for k, v in params.items() 
                                    if k not in ['stim_time', 't_end', 'stim_strength']}, P_val=0),
        [t_stim, t_end], y_stim, t_eval=t_post, method='LSODA'
    )
    
    # Combine results
    t_all = np.concatenate([sol_pre.t, sol_post.t])
    y_all = np.concatenate([sol_pre.y, sol_post.y], axis=1)
    
    return t_all, y_all, t_stim

def create_figure(t, y, t_stim, title="Neural Plasticity Dynamics", params=None):
    """Create publication-quality figure"""
    S, D, A, E = y
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.3)
    
    colors = {'S': '#2E7D32', 'D': '#D32F2F', 'A': '#1976D2', 'E': '#F57C00'}
    
    # Synaptic Strength
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, S, color=colors['S'], linewidth=2.5, label='Synaptic Strength')
    ax1.axvline(t_stim, color='black', linestyle='--', alpha=0.5, label='LTP Stimulus')
    ax1.fill_between(t, 0, S, alpha=0.2, color=colors['S'])
    ax1.set_ylabel('Strength (S)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 2.2])
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(True, alpha=0.2)
    
    # Damage
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, D, color=colors['D'], linewidth=2.5, label='Synaptic Damage')
    ax2.fill_between(t, 0, D, alpha=0.2, color=colors['D'])
    ax2.set_ylabel('Damage (D)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(0.6, np.max(D)*1.1)])
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(True, alpha=0.2)
    
    # Autophagy
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, A, color=colors['A'], linewidth=2.5, label='Autophagic Flux')
    ax3.fill_between(t, 0, A, alpha=0.2, color=colors['A'])
    ax3.set_ylabel('Autophagy (A)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    ax3.legend(loc='upper right', frameon=False)
    ax3.grid(True, alpha=0.2)
    
    # Energy
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, E, color=colors['E'], linewidth=2.5, label='Energy/ATP')
    ax4.fill_between(t, 0, E, alpha=0.2, color=colors['E'])
    ax4.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Energy (E)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 4.0])
    ax4.legend(loc='upper right', frameon=False)
    ax4.grid(True, alpha=0.2)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if params and params['sigma'] > 0:
        param_text = f"sigma = {params['sigma']:.1f} (Chronic Stress)"
        fig.text(0.99, 0.96, param_text, ha='right', fontsize=10, style='italic', color='red')
    elif params:
        param_text = f"sigma = {params['sigma']:.1f} (Control)"
        fig.text(0.99, 0.96, param_text, ha='right', fontsize=10, style='italic', color='green')
    
    plt.tight_layout()
    return fig

# Streamlit Interface
st.title("🧠 Computational Model of Stress-Autophagy-Neural Plasticity")
st.markdown("""
### Interactive simulation demonstrating the AMPK-mTOR-BDNF axis
Based on: **Stress → Autophagy → Clearance → Capacity → Plasticity**
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Simulation", "📊 Analysis", "📚 Theory"])

with tab1:
    with st.sidebar:
        st.header("⚙️ Model Parameters")
        
        st.markdown("---")
        st.subheader("🧹 Autophagy & Stress")
        col1, col2 = st.columns(2)
        with col1:
            sigma = st.number_input("sigma (Stress)", 0.0, 10.0, 0.0, 0.5,
                                   help="Chronic stress level")
        with col2:
            A0 = st.number_input("A0", 0.5, 2.0, 1.0, 0.1,
                               help="Baseline autophagy")
        
        st.markdown("---")
        st.subheader("🔗 Synaptic Parameters")
        k_decay = st.number_input("k_decay", 0.001, 0.01, 0.003, 0.001)
        k_damage = st.number_input("k_damage", 0.0, 1.0, 0.3, 0.05)
        k_maintain = st.number_input("k_maintain", 0.0, 0.02, 0.006, 0.001)
        
        st.markdown("---")
        st.subheader("🗑️ Damage & Clearance")
        beta = st.number_input("beta", 0.001, 0.1, 0.02, 0.005)
        alpha = st.number_input("alpha", 0.0, 1.0, 0.3, 0.05)
        
        st.markdown("---")
        st.subheader("⚡ Energy Metabolism")
        k_atp = st.number_input("k_atp", 0.1, 1.0, 0.3, 0.05)
        k_energy = st.number_input("k_energy", 0.0, 0.5, 0.1, 0.01)
        
        st.markdown("---")
        st.subheader("💉 LTP Protocol")
        stim_strength = st.number_input("Stimulus", 0.0, 2.0, 0.8, 0.1)
        stim_time = st.number_input("Stim Time", 10, 100, 30, 5)
        
        st.markdown("---")
        if st.button("🔄 Reset to Defaults"):
            st.rerun()
    
    # Main area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Effective Autophagy", f"{A0/(
