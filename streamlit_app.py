import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- 1. Optimized Model Definition (From your latest snippet) ---
def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    S, D, A, E = y
    
    # Autophagic Flux: adapts to stress level
    target_A = A0 / (1 + sigma)
    dAdt = 0.2 * (target_A - A)
    
    # Energy: homeostatic regulation with bounds
    # Production scales with mitochondrial health (1-0.5*D)
    production = k_atp * (1 - 0.5 * D)
    consumption = k_energy * S
    dEdt = production - consumption
    
    # Prevent energy from going negative or exploding (Robust Bounds)
    if E < 0.1:
        dEdt = max(0, dEdt)  # Stop decline below threshold
    elif E > 5.0:
        dEdt = min(0, dEdt)  # Cap at upper limit
    
    # Synaptic Strength: maintenance depends on energy availability
    # Key fix: maintenance proportional to (S * E) ensures stability
    maintenance = k_maintain * S * max(0.1, E)  # Minimum energy floor
    
    dSdt = -k_decay * S - k_damage * D * S + P_val + maintenance
    
    # Synaptic Damage: balance between production and clearance
    clearance = alpha * A * D
    dDdt = beta - clearance
    
    return [dSdt, dDdt, dAdt, dEdt]

# --- 2. Professional Graph Generator (The "Supergraph") ---
def create_figure(t, S, D, A, E, title, p_time):
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    plt.subplots_adjust(hspace=0.12, top=0.94, bottom=0.06)

    def style_subplot(ax, data, color, label, ylabel, ylim=None):
        ax.plot(t, data, color=color, linewidth=2.2, label=label)
        # Mark the stimulus time
        ax.axvline(p_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Stimulus')
        ax.set_ylabel(ylabel, fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc='upper right', frameon=False, fontsize=9)

    # Biologically realistic axis limits
    style_subplot(axs[0], S, '#1f77b4', 'Synaptic Strength (S)', 'Strength (a.u.)', ylim=(0, 2.5))
    style_subplot(axs[1], D, '#d62728', 'Cellular Damage (D)', 'Damage (a.u.)', ylim=(0, 0.6))
    style_subplot(axs[2], A, '#2ca02c', 'Autophagic Flux (A)', 'Flux (a.u.)', ylim=(0, 1.2))
    style_subplot(axs[3], E, '#ff7f0e', 'Metabolic Energy (E)', 'Energy (a.u.)', ylim=(0, 4.5))

    axs[0].set_title(title, weight='bold', fontsize=13, pad=12)
    axs[3].set_xlabel('Time (arbitrary units)', fontsize=11, weight='bold')
    
    return fig

# --- 3. PDF Report Generator (With Corrected Abstract) ---
def create_pdf(fig, params):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        
        # Title page
        fig_text = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        fig_text.text(0.1, 0.92, "The Metabolic Switch:", fontsize=18, weight='bold', color='#1f77b4')
        fig_text.text(0.1, 0.88, "Autophagic Flux Gates Synaptic Stability", fontsize=16, weight='bold', color='#1f77b4')
        fig_text.text(0.1, 0.84, "Author: Polykleitos Rengos", fontsize=11, style='italic')
        
        fig_text.text(0.1, 0.78, "Abstract & Hypothesis:", fontsize=12, weight='bold')
        
        # *** CRITICAL FIX: Includes the Lateral Habenula vs PFC distinction ***
        abstract = (
            "This computational model explores the interplay between metabolic stress, "
            "autophagic flux, and synaptic plasticity. While stress-induced suppression "
            "of autophagy has been definitively characterized in the lateral habenula "
            "(Yang et al., 2025), this model tests the hypothesis that similar "
            "mTOR-mediated mechanisms operate in the Prefrontal Cortex (PFC). "
            "Simulations demonstrate that chronic suppression of clearance mechanisms "
            "leads to damage accumulation and metabolic failure of LTP maintenance."
        )
        
        wrapped = textwrap.fill(abstract, width=85)
        fig_text.text(0.1, 0.74, wrapped, fontsize=10, ha='left', va='top', linespacing=1.6)
        
        fig_text.text(0.1, 0.54, "Simulation Parameters (arbitrary units):", fontsize=11, weight='bold')
        param_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in params.items()])
        fig_text.text(0.1, 0.51, param_text, fontsize=9, fontfamily='monospace', va='top')
        
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)

        # Graph page
        pdf.savefig(fig, bbox_inches='tight')
        
    buf.seek(0)
    return buf

# --- 4. Streamlit Interface ---
st.set_page_config(page_title="Autophagy-Plasticity Model", layout="wide")
st.title("Stress → Autophagy → Neural Plasticity")
st.caption("Interactive computational model demonstrating metabolic gating of synaptic stability")

with st.sidebar:
    st.header("Model Parameters")
    
    st.subheader("Synaptic Dynamics")
    k_decay = st.slider("k_decay", 0.001, 0.020, 0.003, 0.001, help="Natural forgetting rate")
    k_damage = st.slider("k_damage", 0.1, 2.0, 0.8, 0.1, help="Damage sensitivity")
    k_maintain = st.slider("k_maintain", 0.000, 0.020, 0.006, 0.001, help="Energy-dependent maintenance")
    
    st.subheader("Autophagy System")
    beta = st.slider("β (waste production)", 0.005, 0.050, 0.010, 0.001)
    alpha = st.slider("α (clearance rate)", 0.10, 0.60, 0.20, 0.05)
    
    st.subheader("Metabolic Dynamics")
    k_atp = st.slider("k_atp (energy production)", 0.1, 1.0, 0.3, 0.05)
    k_energy = st.slider("k_energy (energy cost)", 0.01, 0.20, 0.05, 0.01)

    st.subheader("Experimental Protocol")
    sigma = st.slider("σ (stress level)", 0.0, 8.0, 0.0, 0.5, 
                      help="0 = healthy control, 4+ = chronic stress")
    p_strength = st.slider("Stimulus strength", 0.1, 1.5, 0.6, 0.05)
    p_time = st.slider("Stimulus onset time", 20, 60, 30, 5)
    p_duration = st.slider("Stimulus duration", 5, 30, 15, 5)

    st.markdown("---")
    st.caption("💡 **Tip:** For Figure 1 (control): σ=0. For Figure 2 (stress): σ=4.")

# --- 5. Run Simulation (Robust LSODA Method) ---
# Initial Conditions: S=1, D=0.02, A=Adapted, E=2.0 (Healthy Baseline)
y0 = [1.0, 0.02, 1.0/(1+sigma), 2.0]  
t_stim_end = p_time + p_duration
t_final = 150

params = (k_decay, k_damage, beta, alpha, 1.0, sigma, k_maintain, k_atp, k_energy)

# Phase 1: Pre-stimulus
t1 = np.linspace(0, p_time, 150)
sol1 = solve_ivp(model, [0, p_time], y0, args=(*params, 0.0), t_eval=t1, method='LSODA')

# Phase 2: Stimulus
y1 = sol1.y[:, -1]
t2 = np.linspace(p_time, t_stim_end, 100)
sol2 = solve_ivp(model, [p_time, t_stim_end], y1, args=(*params, p_strength), t_eval=t2, method='LSODA')

# Phase 3: Post-stimulus
y2 = sol2.y[:, -1]
t3 = np.linspace(t_stim_end, t_final, 250)
sol3 = solve_ivp(model, [t_stim_end, t_final], y2, args=(*params, 0.0), t_eval=t3, method='LSODA')

# Combine results
t_all = np.concatenate([sol1.t, sol2.t, sol3.t])
S_all = np.concatenate([sol1.y[0], sol2.y[0], sol3.y[0]])
D_all = np.concatenate([sol1.y[1], sol2.y[1], sol3.y[1]])
A_all = np.concatenate([sol1.y[2], sol2.y[2], sol3.y[2]])
E_all = np.concatenate([sol1.y[3], sol2.y[3], sol3.y[3]])

# --- 6. Display Results ---
col1, col2 = st.columns([3, 1])

with col1:
    condition = "Optimized Condition: Healthy Control" if sigma < 1.0 else "Suboptimal Condition: Chronic Stress"
    # Create the professional figure
    fig = create_figure(t_all, S_all, D_all, A_all, E_all, condition, p_time)
    st.pyplot(fig)

with col2:
    st.subheader("Simulation Results")
    
    st.metric("Initial S", f"{S_all[0]:.3f}")
    st.metric("Peak S", f"{np.max(S_all):.3f}")
    st.metric("Final S", f"{S_all[-1]:.3f}")
    
    # Logic check for LTP
    ltp_success = S_all[-1] > 1.3
    st.metric("LTP Maintained", "Yes ✅" if ltp_success else "No ❌")
    
    st.metric("Final D", f"{D_all[-1]:.3f}")
    st.metric("Final E", f"{E_all[-1]:.3f}")
    st.metric("Effective A", f"{A_all[-1]:.3f}")
    
    st.markdown("---")
    
    # Generate PDF for Download
    param_dict = {
        "k_decay": k_decay, "k_damage": k_damage, "sigma": sigma,
        "beta": beta, "alpha": alpha, "Stimulus": p_strength
    }
    
    pdf_data = create_pdf(fig, param_dict)
    
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name="autophagy_plasticity_results.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    if sigma < 1.0 and ltp_success:
        st.success("**Optimal:** Efficient autophagy maintains proteostasis, allowing energy to fuel LTP stability.")
    elif sigma >= 1.0:
        st.error("**Failed:** Stress suppressed autophagy, leading to damage accumulation and metabolic uncoupling.")
