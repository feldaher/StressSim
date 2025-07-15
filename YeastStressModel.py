import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, List, Tuple

# Your existing yeast stress response model (simplified)
class YeastStressModel:
    """
    Simplified version of your yeast stress response model
    Based on your research parameters
    """
    
    def __init__(self):
        self.reset_state()
        
        # Model parameters from your research
        self.params = {
            'snf1_activation_rate': 2.0,
            'snf1_deactivation_rate': 0.5,
            'msn24_import_rate': 1.0,
            'msn24_export_rate': 0.3,
            'energy_threshold': 0.4,
            'condensation_threshold': 0.3,
            'autophagy_delay': 30,  # time steps
            'recovery_rate': 0.001
        }
        
    def reset_state(self):
        """Reset to healthy cell state"""
        self.state = {
            'energy': 1.0,
            'stress_level': 0.0,
            'snf1_active': 0.0,
            'msn24_nuclear': 0.0,
            'condensate_level': 0.0,
            'autophagy': 0.0,
            'trehalose': 0.0,
            'cell_size': 1.0,
            'time_step': 0,
            'stress_start_time': None
        }
        self.history = [self.state.copy()]
        
    def apply_stress(self, stress_type: str, intensity: float = 0.5):
        """Apply different types of stress"""
        stress_effects = {
            'heat': {'energy_loss': 0.3, 'stress_increase': 0.4},
            'osmotic': {'energy_loss': 0.2, 'stress_increase': 0.3, 'size_change': -0.1},
            'starvation': {'energy_loss': 0.5, 'stress_increase': 0.5},
            'oxidative': {'energy_loss': 0.4, 'stress_increase': 0.6}
        }
        
        if stress_type in stress_effects:
            effects = stress_effects[stress_type]
            self.state['energy'] = max(0, self.state['energy'] - effects['energy_loss'] * intensity)
            self.state['stress_level'] = min(1, self.state['stress_level'] + effects['stress_increase'] * intensity)
            
            if 'size_change' in effects:
                self.state['cell_size'] = max(0.7, self.state['cell_size'] + effects['size_change'] * intensity)
                
            if not self.state['stress_start_time']:
                self.state['stress_start_time'] = self.state['time_step']
    
    def update(self, dt: float = 0.01):
        """Update cell state for one time step"""
        self.state['time_step'] += 1
        
        # SNF1 activation (energy sensing)
        if self.state['energy'] < self.params['energy_threshold']:
            activation_rate = self.params['snf1_activation_rate'] * (1 - self.state['energy'] / self.params['energy_threshold'])
            self.state['snf1_active'] = min(1, self.state['snf1_active'] + activation_rate * dt)
        else:
            self.state['snf1_active'] = max(0, self.state['snf1_active'] - self.params['snf1_deactivation_rate'] * dt)
        
        # Msn2/4 nuclear translocation
        import_driving = self.state['snf1_active'] * self.params['msn24_import_rate']
        export_rate = self.params['msn24_export_rate']
        net_import = import_driving * (1 - self.state['msn24_nuclear']) - export_rate * self.state['msn24_nuclear']
        self.state['msn24_nuclear'] = max(0, min(1, self.state['msn24_nuclear'] + net_import * dt))
        
        # Protein condensation
        if self.state['stress_level'] > self.params['condensation_threshold']:
            condensation_rate = (self.state['stress_level'] - self.params['condensation_threshold']) * 0.05
            self.state['condensate_level'] = min(1, self.state['condensate_level'] + condensation_rate)
        else:
            self.state['condensate_level'] = max(0, self.state['condensate_level'] - 0.02)
        
        # Autophagy (with delay)
        time_since_stress = (self.state['time_step'] - self.state['stress_start_time']) if self.state['stress_start_time'] else 0
        if self.state['energy'] < self.params['energy_threshold'] and time_since_stress > self.params['autophagy_delay']:
            autophagy_target = 1 - self.state['energy'] / self.params['energy_threshold']
            self.state['autophagy'] += (autophagy_target - self.state['autophagy']) * 0.05
        else:
            self.state['autophagy'] = max(0, self.state['autophagy'] - 0.025)
        
        # Trehalose synthesis (stress protectant)
        if self.state['msn24_nuclear'] > 0.2:
            trehalose_target = self.state['msn24_nuclear'] * 0.8
            self.state['trehalose'] += (trehalose_target - self.state['trehalose']) * 0.02
        else:
            self.state['trehalose'] = max(0, self.state['trehalose'] - 0.01)
        
        # Recovery processes
        self.state['stress_level'] = max(0, self.state['stress_level'] - 0.002)
        self.state['cell_size'] += (1.0 - self.state['cell_size']) * 0.01
        
        # Energy recovery (when stress is low)
        if self.state['stress_level'] < 0.2:
            self.state['energy'] = min(1, self.state['energy'] + self.params['recovery_rate'])
        
        # Reset stress timer if recovered
        if self.state['stress_level'] < 0.05:
            self.state['stress_start_time'] = None
        
        # Store history
        self.history.append(self.state.copy())
        
        return self.state.copy()

def create_interactive_cell_visualization(model_state):
    """Create an interactive cell visualization using Plotly"""
    
    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cell State Overview', 'Signaling Network', 'Time Course', 'Stress Response'),
        specs=[[{"type": "scatterpolar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Radar chart for overall cell state
    categories = ['Energy', 'SNF1 Active', 'Msn2/4 Nuclear', 'Condensates', 'Autophagy', 'Trehalose']
    values = [
        model_state['energy'],
        model_state['snf1_active'],
        model_state['msn24_nuclear'],
        model_state['condensate_level'],
        model_state['autophagy'],
        model_state['trehalose']
    ]
    
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Cell State',
            line_color='cyan'
        ),
        row=1, col=1
    )
    
    # 2. Network diagram (simplified)
    network_x = [0, 1, 2, 1, 1.5, 0.5]
    network_y = [1, 2, 1, 0, -1, -1]
    network_labels = ['Energy', 'SNF1', 'Msn2/4', 'Condensates', 'Autophagy', 'Trehalose']
    network_sizes = [20, 15 + model_state['snf1_active']*20, 15 + model_state['msn24_nuclear']*20,
                    15 + model_state['condensate_level']*20, 15 + model_state['autophagy']*20,
                    15 + model_state['trehalose']*20]
    
    fig.add_trace(
        go.Scatter(
            x=network_x, y=network_y,
            mode='markers+text',
            marker=dict(size=network_sizes, color='lightblue'),
            text=network_labels,
            textposition="middle center",
            name='Network'
        ),
        row=1, col=2
    )
    
    return fig

def simulate_stress_response(model: YeastStressModel, stress_type: str, duration: int = 100):
    """Simulate stress response over time"""
    
    # Apply stress at the beginning
    model.apply_stress(stress_type, intensity=0.7)
    
    time_points = []
    energy_data = []
    stress_data = []
    snf1_data = []
    condensate_data = []
    
    for i in range(duration):
        state = model.update()
        time_points.append(i)
        energy_data.append(state['energy'])
        stress_data.append(state['stress_level'])
        snf1_data.append(state['snf1_active'])
        condensate_data.append(state['condensate_level'])
    
    return pd.DataFrame({
        'Time': time_points,
        'Energy': energy_data,
        'Stress Level': stress_data,
        'SNF1 Active': snf1_data,
        'Condensates': condensate_data
    })

# Streamlit App
def main():
    st.set_page_config(page_title="Cellular Stress Response Model", layout="wide")
    
    st.title("🔬 Cellular Stress Response: Interactive Model")
    st.markdown("*Based on yeast stress response research - Exploring how cells adapt to crisis*")
    
    # Initialize model in session state
    if 'model' not in st.session_state:
        st.session_state.model = YeastStressModel()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Experimental Controls")
    
    # Stress application
    st.sidebar.subheader("Apply Stress")
    stress_type = st.sidebar.selectbox(
        "Stress Type",
        ['heat', 'osmotic', 'starvation', 'oxidative'],
        help="Different stress types affect the cell differently"
    )
    
    stress_intensity = st.sidebar.slider(
        "Stress Intensity",
        0.1, 1.0, 0.5,
        help="Higher intensity = more severe stress"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔥 Apply Stress"):
            st.session_state.model.apply_stress(stress_type, stress_intensity)
            st.sidebar.success(f"Applied {stress_type} stress!")
    
    with col2:
        if st.button("🔄 Reset Cell"):
            st.session_state.model.reset_state()
            st.sidebar.success("Cell reset to healthy state!")
    
    # Auto-update toggle
    auto_update = st.sidebar.checkbox("⚡ Auto-update (real-time)", value=False)
    
    if auto_update:
        # Auto-refresh every second
        time.sleep(0.1)
        st.session_state.model.update()
        st.rerun()
    else:
        # Manual update button
        if st.sidebar.button("⏭️ Update Step"):
            st.session_state.model.update()
    
    # Main display
    current_state = st.session_state.model.state
    
    # Current state metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Energy (ATP/ADP)",
            f"{current_state['energy']:.2f}",
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        stress_color = "inverse" if current_state['stress_level'] > 0.5 else "normal"
        st.metric(
            "Stress Level",
            f"{current_state['stress_level']:.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "SNF1 Activation",
            f"{current_state['snf1_active']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Protein Condensates",
            f"{current_state['condensate_level']:.2f}",
            delta=None
        )
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Current State", "📈 Time Course", "🧪 Simulation"])
    
    with tab1:
        if len(st.session_state.model.history) > 1:
            # Create DataFrame from history
            history_df = pd.DataFrame(st.session_state.model.history)
            
            # Plot time course of key variables
            fig = px.line(
                history_df.iloc[-50:],  # Last 50 time points
                x='time_step',
                y=['energy', 'stress_level', 'snf1_active', 'condensate_level'],
                title="Cellular State Over Time",
                labels={'value': 'Level (0-1)', 'time_step': 'Time Steps'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Complete Time Course Analysis")
        
        if len(st.session_state.model.history) > 10:
            history_df = pd.DataFrame(st.session_state.model.history)
            
            # Multi-panel time course
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Energy & Stress', 'Signaling Molecules', 'Cellular Responses'),
                shared_xaxes=True
            )
            
            # Panel 1: Energy and stress
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['energy'], 
                                   name='Energy', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['stress_level'], 
                                   name='Stress', line=dict(color='red')), row=1, col=1)
            
            # Panel 2: Signaling
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['snf1_active'], 
                                   name='SNF1', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['msn24_nuclear'], 
                                   name='Msn2/4', line=dict(color='purple')), row=2, col=1)
            
            # Panel 3: Responses
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['condensate_level'], 
                                   name='Condensates', line=dict(color='magenta')), row=3, col=1)
            fig.add_trace(go.Scatter(x=history_df['time_step'], y=history_df['autophagy'], 
                                   name='Autophagy', line=dict(color='orange')), row=3, col=1)
            
            fig.update_layout(height=600, title="Complete Cellular Response Timeline")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🧪 Stress Response Simulation")
        
        sim_stress = st.selectbox("Simulation Stress Type", ['heat', 'osmotic', 'starvation', 'oxidative'])
        sim_duration = st.slider("Simulation Duration", 50, 500, 200)
        
        if st.button("🚀 Run Simulation"):
            # Create fresh model for simulation
            sim_model = YeastStressModel()
            
            with st.spinner("Running simulation..."):
                sim_data = simulate_stress_response(sim_model, sim_stress, sim_duration)
            
            # Plot simulation results
            fig = px.line(
                sim_data,
                x='Time',
                y=['Energy', 'Stress Level', 'SNF1 Active', 'Condensates'],
                title=f"Simulated {sim_stress.title()} Stress Response",
                labels={'value': 'Level (0-1)', 'Time': 'Time Steps'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow data download
            csv = sim_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Data",
                data=csv,
                file_name=f"{sim_stress}_stress_simulation.csv",
                mime="text/csv"
            )
    
    # Scientific information
    with st.expander("📚 About This Model"):
        st.markdown("""
        **Scientific Basis**: This model is based on research into yeast stress response mechanisms,
        particularly the SNF1 pathway and stress-responsive transcription factors.
        
        **Key Components**:
        - **SNF1**: Energy-sensing kinase that activates under low ATP conditions
        - **Msn2/4**: Stress-responsive transcription factors that translocate to nucleus
        - **Protein Condensates**: Stress granules and P-bodies that form under stress
        - **Autophagy**: Cellular degradation process for energy recovery
        
        **Model Parameters**: Based on experimental kinetic data from yeast stress response studies.
        """)

if __name__ == "__main__":
    main()
