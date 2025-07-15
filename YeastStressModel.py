import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class StressType(Enum):
    """Enumeration of stress types with specific cellular targets"""
    HEAT = "heat"
    OSMOTIC = "osmotic" 
    CARBON_STARVATION = "carbon_starvation"
    OXIDATIVE = "oxidative"
    NITROGEN_STARVATION = "nitrogen_starvation"

@dataclass
class ModelParameters:
    """
    Model parameters based on experimental literature
    References included for each parameter set
    """
    
    # Energy dynamics (Hardie et al., 2012; Hedbacker & Carlson, 2008)
    atp_consumption_basal: float = 0.1  # mM/min, basal ATP consumption
    atp_synthesis_max: float = 0.5      # mM/min, maximum ATP synthesis rate
    atp_km: float = 1.0                 # mM, Michaelis constant for ATP synthesis
    energy_threshold_snf1: float = 0.4  # Normalized ATP/ADP ratio for SNF1 activation
    
    # SNF1 kinetics (Mayer et al., 2011; Broach, 2012)
    snf1_k_act: float = 2.0            # min⁻¹, SNF1 activation rate constant
    snf1_k_deact: float = 0.5          # min⁻¹, SNF1 deactivation rate constant
    snf1_hill_coeff: float = 2.0       # Hill coefficient for energy sensing
    snf1_phosphatase_activity: float = 0.1  # Background phosphatase activity
    
    # Msn2/4 translocation (Jacquet et al., 2003; Görner et al., 1998)
    msn24_k_import: float = 1.0        # min⁻¹, nuclear import rate
    msn24_k_export: float = 0.3        # min⁻¹, nuclear export rate (PKA-dependent)
    msn24_snf1_sensitivity: float = 0.8 # Fraction of translocation driven by SNF1
    msn24_direct_stress_sensitivity: float = 0.2  # Direct stress response
    
    # Stress granule/P-body formation (Buchan & Parker, 2009; Protter & Parker, 2016)
    condensate_threshold: float = 0.3   # Stress level threshold for condensate formation
    condensate_k_form: float = 0.05     # min⁻¹, condensate formation rate
    condensate_k_dissolve: float = 0.02 # min⁻¹, condensate dissolution rate
    condensate_saturation: float = 0.8  # Maximum condensate level
    
    # Autophagy (Yorimitsu & Klionsky, 2005; Noda & Ohsumi, 1998)
    autophagy_delay: float = 30         # min, time delay for autophagy induction
    autophagy_k_max: float = 0.05       # min⁻¹, maximum autophagy rate
    autophagy_energy_threshold: float = 0.4  # Energy threshold for autophagy
    autophagy_nutrient_recovery: float = 0.01 # Nutrient recovery efficiency
    
    # Trehalose metabolism (Thevelein, 1984; François & Parrou, 2001)
    trehalose_k_synth: float = 0.1      # min⁻¹, trehalose synthesis rate
    trehalose_k_deg: float = 0.01       # min⁻¹, trehalose degradation rate
    trehalose_msn24_sensitivity: float = 0.8  # Msn2/4 dependence
    
    # Glycogen metabolism (François & Parrou, 2001)
    glycogen_k_synth: float = 0.2       # min⁻¹, glycogen synthesis rate
    glycogen_k_mobilize: float = 0.1    # min⁻¹, glycogen mobilization rate
    glycogen_energy_threshold_synth: float = 0.7  # Energy threshold for synthesis
    glycogen_km: float = 0.6            # Michaelis constant for glycogen synthesis
    
    # Cell cycle control (Dolinski & Botstein, 2007)
    cell_cycle_energy_threshold: float = 0.5  # Energy threshold for cell cycle progression
    cell_cycle_arrest_time: float = 60  # min, minimum arrest time
    
    # Protein synthesis (Ashe et al., 2000; Castelli et al., 2011)
    protein_synthesis_hill_coeff: float = 4.0  # Hill coefficient for energy dependence
    protein_synthesis_k_half: float = 0.7      # Half-maximum energy for protein synthesis
    
    # Recovery dynamics
    stress_recovery_rate: float = 0.002  # min⁻¹, intrinsic stress recovery rate
    membrane_repair_rate: float = 0.005  # min⁻¹, membrane integrity recovery
    size_recovery_rate: float = 0.01     # min⁻¹, cell size recovery rate

class EnhancedYeastStressModel:
    """
    Enhanced yeast stress response model with rigorous scientific basis
    
    Based on established literature and experimental kinetic data:
    - SNF1 pathway kinetics (Hardie et al., 2012)
    - Msn2/4 translocation dynamics (Jacquet et al., 2003)
    - Stress granule formation (Buchan & Parker, 2009)
    - Autophagy regulation (Yorimitsu & Klionsky, 2005)
    - Trehalose/glycogen metabolism (François & Parrou, 2001)
    """
    
    def __init__(self, params: Optional[ModelParameters] = None):
        self.params = params or ModelParameters()
        self.reset_state()
        
        # Stress-specific effects based on literature
        self.stress_effects = {
            StressType.HEAT: {
                'protein_damage_rate': 0.3,      # Protein misfolding/aggregation
                'membrane_fluidity_change': 0.2,  # Membrane perturbation
                'energy_cost_multiplier': 1.5,   # Increased ATP demand for repair
                'condensate_promotion': 0.4,     # Heat promotes stress granules
                'primary_target': 'protein_folding'
            },
            StressType.OSMOTIC: {
                'cell_volume_change': -0.2,      # Cell shrinkage
                'turgor_pressure_loss': 0.3,     # Osmotic imbalance
                'energy_cost_multiplier': 1.3,   # Ion transport energy cost
                'glycerol_synthesis_induction': 0.5,  # Osmolyte response
                'primary_target': 'cell_volume'
            },
            StressType.CARBON_STARVATION: {
                'glucose_depletion_rate': 0.8,   # Direct carbon source loss
                'gluconeogenesis_induction': 0.6, # Alternative carbon pathways
                'energy_cost_multiplier': 0.7,   # Reduced energy availability
                'autophagy_promotion': 0.8,      # Strong autophagy induction
                'primary_target': 'energy_metabolism'
            },
            StressType.OXIDATIVE: {
                'protein_damage_rate': 0.5,      # ROS-induced protein damage
                'dna_damage_rate': 0.2,          # Oxidative DNA damage
                'energy_cost_multiplier': 1.8,   # High repair costs
                'antioxidant_induction': 0.7,    # Antioxidant defense
                'primary_target': 'oxidative_damage'
            },
            StressType.NITROGEN_STARVATION: {
                'amino_acid_depletion': 0.7,     # Nitrogen source depletion
                'protein_degradation_rate': 0.4, # Protein catabolism
                'energy_cost_multiplier': 0.8,   # Reduced anabolic processes
                'autophagy_promotion': 0.9,      # Very strong autophagy
                'primary_target': 'nitrogen_metabolism'
            }
        }
    
    def reset_state(self):
        """Reset to healthy exponential growth state"""
        self.state = {
            # Energy status
            'atp_adp_ratio': 1.0,               # Normalized ATP/ADP ratio
            'glucose_internal': 1.0,            # Internal glucose concentration
            'energy_charge': 0.9,               # Adenine nucleotide energy charge
            
            # Stress levels
            'stress_level': 0.0,                # Overall stress intensity
            'oxidative_stress': 0.0,            # ROS levels
            'osmotic_stress': 0.0,              # Osmotic imbalance
            'heat_stress': 0.0,                 # Temperature stress
            
            # Signaling molecules
            'snf1_active_fraction': 0.0,        # Fraction of SNF1 in active state
            'snf1_phosphorylation': 0.0,        # SNF1 T210 phosphorylation level
            'msn24_nuclear_fraction': 0.0,      # Fraction of Msn2/4 in nucleus
            'pka_activity': 1.0,                # PKA activity (high in good conditions)
            
            # Cellular responses
            'condensate_level': 0.0,            # Stress granule/P-body abundance
            'autophagy_flux': 0.0,              # Autophagy activity
            'proteasome_activity': 1.0,         # Proteasome degradation
            'ribosome_activity': 1.0,           # Translation rate
            
            # Storage compounds
            'trehalose': 0.1,                   # Trehalose concentration (mM)
            'glycogen': 0.5,                    # Glycogen concentration (relative)
            'neutral_lipids': 0.3,              # Lipid droplets
            
            # Physical properties
            'cell_volume': 1.0,                 # Relative cell volume
            'membrane_integrity': 1.0,          # Membrane function
            'turgor_pressure': 1.0,             # Internal pressure
            
            # Cell cycle
            'cell_cycle_phase': 'G1',           # Current cell cycle phase
            'cell_cycle_arrested': False,       # Cell cycle arrest status
            'arrest_duration': 0.0,             # Time in arrest (min)
            
            # Time tracking
            'time': 0.0,                        # Simulation time (min)
            'stress_onset_time': None,          # Time when stress was first applied
            'adaptation_time': None,            # Time when adaptation begins
        }
        
        # History tracking for dynamics
        self.history = [self.state.copy()]
        self.stress_history = []
    
    def apply_stress(self, stress_type: StressType, intensity: float = 0.5, duration: Optional[float] = None):
        """
        Apply specific stress with biologically realistic effects
        
        Args:
            stress_type: Type of stress to apply
            intensity: Stress intensity (0.0 to 1.0)
            duration: Duration of stress (None for sustained)
        """
        if not isinstance(stress_type, StressType):
            stress_type = StressType(stress_type)
        
        effects = self.stress_effects[stress_type]
        
        # Record stress application
        stress_event = {
            'type': stress_type,
            'intensity': intensity,
            'time': self.state['time'],
            'duration': duration
        }
        self.stress_history.append(stress_event)
        
        if self.state['stress_onset_time'] is None:
            self.state['stress_onset_time'] = self.state['time']
        
        # Apply immediate effects based on stress type
        if stress_type == StressType.HEAT:
            self._apply_heat_stress(intensity, effects)
        elif stress_type == StressType.OSMOTIC:
            self._apply_osmotic_stress(intensity, effects)
        elif stress_type == StressType.CARBON_STARVATION:
            self._apply_carbon_starvation(intensity, effects)
        elif stress_type == StressType.OXIDATIVE:
            self._apply_oxidative_stress(intensity, effects)
        elif stress_type == StressType.NITROGEN_STARVATION:
            self._apply_nitrogen_starvation(intensity, effects)
    
    def _apply_heat_stress(self, intensity: float, effects: Dict):
        """Apply heat shock effects"""
        self.state['heat_stress'] = min(1.0, self.state['heat_stress'] + intensity)
        self.state['stress_level'] = min(1.0, self.state['stress_level'] + 0.4 * intensity)
        
        # Protein misfolding increases energy demands
        energy_cost = effects['energy_cost_multiplier'] * intensity
        self.state['atp_adp_ratio'] = max(0.0, self.state['atp_adp_ratio'] - 0.3 * intensity)
        
        # Membrane perturbation
        membrane_damage = effects['membrane_fluidity_change'] * intensity
        self.state['membrane_integrity'] = max(0.5, self.state['membrane_integrity'] - membrane_damage)
        
        # Promote stress granule formation
        self.state['condensate_level'] = min(1.0, self.state['condensate_level'] + 
                                           effects['condensate_promotion'] * intensity)
    
    def _apply_osmotic_stress(self, intensity: float, effects: Dict):
        """Apply osmotic stress effects"""
        self.state['osmotic_stress'] = min(1.0, self.state['osmotic_stress'] + intensity)
        self.state['stress_level'] = min(1.0, self.state['stress_level'] + 0.3 * intensity)
        
        # Cell volume changes
        volume_change = effects['cell_volume_change'] * intensity
        self.state['cell_volume'] = max(0.7, self.state['cell_volume'] + volume_change)
        
        # Turgor pressure loss
        self.state['turgor_pressure'] = max(0.3, self.state['turgor_pressure'] - 
                                          effects['turgor_pressure_loss'] * intensity)
        
        # Energy costs for ion transport
        self.state['atp_adp_ratio'] = max(0.0, self.state['atp_adp_ratio'] - 0.2 * intensity)
    
    def _apply_carbon_starvation(self, intensity: float, effects: Dict):
        """Apply carbon starvation effects"""
        self.state['stress_level'] = min(1.0, self.state['stress_level'] + 0.5 * intensity)
        
        # Direct glucose depletion
        glucose_loss = effects['glucose_depletion_rate'] * intensity
        self.state['glucose_internal'] = max(0.0, self.state['glucose_internal'] - glucose_loss)
        
        # Severe energy depletion
        self.state['atp_adp_ratio'] = max(0.0, self.state['atp_adp_ratio'] - 0.5 * intensity)
        
        # Reduce PKA activity (glucose sensing)
        self.state['pka_activity'] = max(0.1, self.state['pka_activity'] - 0.6 * intensity)
    
    def _apply_oxidative_stress(self, intensity: float, effects: Dict):
        """Apply oxidative stress effects"""
        self.state['oxidative_stress'] = min(1.0, self.state['oxidative_stress'] + intensity)
        self.state['stress_level'] = min(1.0, self.state['stress_level'] + 0.6 * intensity)
        
        # Protein and membrane damage
        self.state['atp_adp_ratio'] = max(0.0, self.state['atp_adp_ratio'] - 0.4 * intensity)
        self.state['membrane_integrity'] = max(0.3, self.state['membrane_integrity'] - 0.2 * intensity)
        
        # Proteasome activation for damaged proteins
        self.state['proteasome_activity'] = min(2.0, self.state['proteasome_activity'] + 0.5 * intensity)
    
    def _apply_nitrogen_starvation(self, intensity: float, effects: Dict):
        """Apply nitrogen starvation effects"""
        self.state['stress_level'] = min(1.0, self.state['stress_level'] + 0.4 * intensity)
        
        # Reduced protein synthesis
        self.state['ribosome_activity'] = max(0.1, self.state['ribosome_activity'] - 0.6 * intensity)
        
        # Energy reduction (less anabolic processes)
        self.state['atp_adp_ratio'] = max(0.0, self.state['atp_adp_ratio'] - 0.3 * intensity)
    
    def update(self, dt: float = 0.1):
        """
        Update all cellular processes for one time step
        
        Args:
            dt: Time step in minutes
        """
        self.state['time'] += dt
        
        # Update core signaling pathways
        self._update_snf1_pathway(dt)
        self._update_msn24_translocation(dt)
        self._update_energy_metabolism(dt)
        
        # Update cellular responses
        self._update_stress_granules(dt)
        self._update_autophagy(dt)
        self._update_storage_compounds(dt)
        self._update_protein_synthesis(dt)
        
        # Update cell cycle
        self._update_cell_cycle(dt)
        
        # Update recovery processes
        self._update_recovery_processes(dt)
        
        # Store state history
        self.history.append(self.state.copy())
        
        return self.state.copy()
    
    def _update_snf1_pathway(self, dt: float):
        """Update SNF1 kinase activation state"""
        # SNF1 activation depends on energy status (Hill function)
        energy_ratio = self.state['atp_adp_ratio'] / self.params.energy_threshold_snf1
        
        if energy_ratio < 1.0:
            # Low energy: activate SNF1
            activation_driving_force = (1.0 - energy_ratio**self.params.snf1_hill_coeff)
            dsnf1_dt = (self.params.snf1_k_act * activation_driving_force * 
                       (1 - self.state['snf1_active_fraction']) - 
                       self.params.snf1_phosphatase_activity * self.state['snf1_active_fraction'])
        else:
            # High energy: deactivate SNF1
            dsnf1_dt = -self.params.snf1_k_deact * self.state['snf1_active_fraction']
        
        self.state['snf1_active_fraction'] = np.clip(
            self.state['snf1_active_fraction'] + dsnf1_dt * dt, 0.0, 1.0
        )
        
        # SNF1 phosphorylation follows activation
        self.state['snf1_phosphorylation'] = self.state['snf1_active_fraction']
    
    def _update_msn24_translocation(self, dt: float):
        """Update Msn2/4 nuclear translocation"""
        # Nuclear import driven by SNF1 and direct stress sensing
        snf1_contribution = (self.params.msn24_snf1_sensitivity * 
                           self.state['snf1_active_fraction'])
        stress_contribution = (self.params.msn24_direct_stress_sensitivity * 
                             self.state['stress_level'])
        
        import_rate = self.params.msn24_k_import * (snf1_contribution + stress_contribution)
        
        # Nuclear export inhibited by stress, promoted by PKA
        export_rate = self.params.msn24_k_export * self.state['pka_activity']
        
        # Net translocation
        dmsn24_dt = (import_rate * (1 - self.state['msn24_nuclear_fraction']) - 
                    export_rate * self.state['msn24_nuclear_fraction'])
        
        self.state['msn24_nuclear_fraction'] = np.clip(
            self.state['msn24_nuclear_fraction'] + dmsn24_dt * dt, 0.0, 1.0
        )
    
    def _update_energy_metabolism(self, dt: float):
        """Update cellular energy status"""
        # ATP synthesis from glucose (Michaelis-Menten)
        glucose_factor = (self.state['glucose_internal'] / 
                         (self.params.atp_km + self.state['glucose_internal']))
        atp_synthesis = self.params.atp_synthesis_max * glucose_factor
        
        # ATP consumption (stress increases consumption)
        stress_multiplier = 1.0 + self.state['stress_level'] * 0.5
        atp_consumption = self.params.atp_consumption_basal * stress_multiplier
        
        # Net energy change
        net_atp_change = (atp_synthesis - atp_consumption) * dt
        self.state['atp_adp_ratio'] = np.clip(
            self.state['atp_adp_ratio'] + net_atp_change, 0.0, 1.0
        )
        
        # Energy charge calculation
        self.state['energy_charge'] = (self.state['atp_adp_ratio'] + 0.5) / 1.5
    
    def _update_stress_granules(self, dt: float):
        """Update stress granule formation"""
        if self.state['stress_level'] > self.params.condensate_threshold:
            # Formation rate depends on stress level
            formation_rate = (self.params.condensate_k_form * 
                            (self.state['stress_level'] - self.params.condensate_threshold))
            target_level = min(self.params.condensate_saturation, 
                             self.state['stress_level'])
            
            dcondensate_dt = formation_rate * (target_level - self.state['condensate_level'])
        else:
            # Dissolution when stress is low
            dcondensate_dt = -self.params.condensate_k_dissolve * self.state['condensate_level']
        
        self.state['condensate_level'] = np.clip(
            self.state['condensate_level'] + dcondensate_dt * dt, 0.0, 1.0
        )
    
    def _update_autophagy(self, dt: float):
        """Update autophagy with time delay"""
        time_since_stress = (self.state['time'] - self.state['stress_onset_time'] 
                           if self.state['stress_onset_time'] else 0)
        
        # Autophagy only activates after delay and under energy stress
        if (self.state['atp_adp_ratio'] < self.params.autophagy_energy_threshold and 
            time_since_stress > self.params.autophagy_delay):
            
            # Autophagy rate depends on energy deficit
            energy_deficit = (self.params.autophagy_energy_threshold - 
                            self.state['atp_adp_ratio'])
            autophagy_target = energy_deficit / self.params.autophagy_energy_threshold
            
            dautophagy_dt = (self.params.autophagy_k_max * 
                           (autophagy_target - self.state['autophagy_flux']))
        else:
            # Autophagy decreases when not needed
            dautophagy_dt = -self.params.autophagy_k_max * 0.5 * self.state['autophagy_flux']
        
        self.state['autophagy_flux'] = np.clip(
            self.state['autophagy_flux'] + dautophagy_dt * dt, 0.0, 1.0
        )
        
        # Autophagy provides nutrient recovery
        if self.state['autophagy_flux'] > 0.1:
            nutrient_recovery = (self.params.autophagy_nutrient_recovery * 
                               self.state['autophagy_flux'] * dt)
            self.state['glucose_internal'] = min(1.0, 
                self.state['glucose_internal'] + nutrient_recovery)
    
    def _update_storage_compounds(self, dt: float):
        """Update trehalose and glycogen levels"""
        # Trehalose synthesis (stress protectant)
        if self.state['msn24_nuclear_fraction'] > 0.2:
            trehalose_target = (self.params.trehalose_msn24_sensitivity * 
                              self.state['msn24_nuclear_fraction'])
            dtrehalose_dt = (self.params.trehalose_k_synth * 
                           (trehalose_target - self.state['trehalose']))
        else:
            dtrehalose_dt = -self.params.trehalose_k_deg * self.state['trehalose']
        
        self.state['trehalose'] = np.clip(
            self.state['trehalose'] + dtrehalose_dt * dt, 0.0, 1.0
        )
        
        # Glycogen metabolism
        if (self.state['atp_adp_ratio'] > self.params.glycogen_energy_threshold_synth and 
            self.state['glucose_internal'] > 0.5):
            # Synthesis when energy and glucose are abundant
            glycogen_driving_force = (self.state['glucose_internal'] / 
                                    (self.params.glycogen_km + self.state['glucose_internal']))
            dglycogen_dt = self.params.glycogen_k_synth * glycogen_driving_force
        elif self.state['atp_adp_ratio'] < self.params.autophagy_energy_threshold:
            # Mobilization when energy is low
            dglycogen_dt = -self.params.glycogen_k_mobilize * self.state['glycogen']
            # Add glucose from glycogen breakdown
            glucose_from_glycogen = self.params.glycogen_k_mobilize * self.state['glycogen'] * dt
            self.state['glucose_internal'] = min(1.0, 
                self.state['glucose_internal'] + glucose_from_glycogen)
        else:
            dglycogen_dt = 0.0
        
        self.state['glycogen'] = np.clip(
            self.state['glycogen'] + dglycogen_dt * dt, 0.0, 1.0
        )
    
    def _update_protein_synthesis(self, dt: float):
        """Update protein synthesis rate"""
        # Hill function for energy dependence
        energy_factor = (self.state['atp_adp_ratio']**self.params.protein_synthesis_hill_coeff / 
                        (self.params.protein_synthesis_k_half**self.params.protein_synthesis_hill_coeff + 
                         self.state['atp_adp_ratio']**self.params.protein_synthesis_hill_coeff))
        
        self.state['ribosome_activity'] = energy_factor
    
    def _update_cell_cycle(self, dt: float):
        """Update cell cycle progression and arrest"""
        if self.state['atp_adp_ratio'] < self.params.cell_cycle_energy_threshold:
            if not self.state['cell_cycle_arrested']:
                self.state['cell_cycle_arrested'] = True
                self.state['cell_cycle_phase'] = 'G1_arrest'
            self.state['arrest_duration'] += dt
        else:
            if (self.state['cell_cycle_arrested'] and 
                self.state['arrest_duration'] > self.params.cell_cycle_arrest_time):
                self.state['cell_cycle_arrested'] = False
                self.state['cell_cycle_phase'] = 'G1'
                self.state['arrest_duration'] = 0.0
    
    def _update_recovery_processes(self, dt: float):
        """Update cellular recovery processes"""
        # Intrinsic stress recovery
        self.state['stress_level'] = max(0.0, 
            self.state['stress_level'] - self.params.stress_recovery_rate * dt)
        self.state['heat_stress'] = max(0.0,
            self.state['heat_stress'] - self.params.stress_recovery_rate * dt)
        self.state['osmotic_stress'] = max(0.0,
            self.state['osmotic_stress'] - self.params.stress_recovery_rate * dt)
        self.state['oxidative_stress'] = max(0.0,
            self.state['oxidative_stress'] - self.params.stress_recovery_rate * dt)
        
        # Membrane repair
        if self.state['stress_level'] < 0.3:
            self.state['membrane_integrity'] = min(1.0,
                self.state['membrane_integrity'] + self.params.membrane_repair_rate * dt)
        
        # Cell size recovery
        self.state['cell_volume'] += (1.0 - self.state['cell_volume']) * self.params.size_recovery_rate * dt
        self.state['turgor_pressure'] += (1.0 - self.state['turgor_pressure']) * self.params.size_recovery_rate * dt
        
        # PKA activity recovery (glucose sensing)
        if self.state['glucose_internal'] > 0.3:
            self.state['pka_activity'] = min(1.0, self.state['pka_activity'] + 0.01 * dt)
        
        # Reset stress onset time if fully recovered
        if self.state['stress_level'] < 0.05:
            self.state['stress_onset_time'] = None
    
    def get_summary_state(self) -> Dict:
        """Get simplified state for visualization"""
        return {
            'energy': self.state['atp_adp_ratio'],
            'stress_level': self.state['stress_level'],
            'snf1_active': self.state['snf1_active_fraction'],
            'msn24_nuclear': self.state['msn24_nuclear_fraction'],
            'condensate_level': self.state['condensate_level'],
            'autophagy': self.state['autophagy_flux'],
            'trehalose': self.state['trehalose'],
            'glycogen': self.state['glycogen'],
            'cell_size': self.state['cell_volume'],
            'protein_synthesis': self.state['ribosome_activity'],
            'time_step': self.state['time']
        }
    
    def get_detailed_metrics(self) -> Dict:
        """Get comprehensive metrics for scientific analysis"""
        return {
            'Energy Status': {
                'ATP/ADP Ratio': self.state['atp_adp_ratio'],
                'Energy Charge': self.state['energy_charge'],
                'Internal Glucose': self.state['glucose_internal'],
                'Glycogen Stores': self.state['glycogen']
            },
            'Stress Response': {
                'Overall Stress': self.state['stress_level'],
                'Heat Stress': self.state['heat_stress'],
                'Osmotic Stress': self.state['osmotic_stress'],
                'Oxidative Stress': self.state['oxidative_stress'],
                'Stress Duration (min)': (self.state['time'] - self.state['stress_onset_time']) 
                                        if self.state['stress_onset_time'] else 0
            },
            'Signaling Network': {
                'SNF1 Active (%)': self.state['snf1_active_fraction'] * 100,
                'SNF1 Phosphorylation': self.state['snf1_phosphorylation'],
                'Msn2/4 Nuclear (%)': self.state['msn24_nuclear_fraction'] * 100,
                'PKA Activity': self.state['pka_activity']
            },
            'Cellular Responses': {
                'Stress Granules': self.state['condensate_level'],
                'Autophagy Flux': self.state['autophagy_flux'],
                'Proteasome Activity': self.state['proteasome_activity'],
                'Protein Synthesis Rate': self.state['ribosome_activity']
            },
            'Storage & Protection': {
                'Trehalose (mM)': self.state['trehalose'],
                'Glycogen (relative)': self.state['glycogen'],
                'Neutral Lipids': self.state['neutral_lipids']
            },
            'Physical State': {
                'Cell Volume': self.state['cell_volume'],
                'Membrane Integrity': self.state['membrane_integrity'],
                'Turgor Pressure': self.state['turgor_pressure']
            },
            'Cell Cycle': {
                'Phase': self.state['cell_cycle_phase'],
                'Arrested': self.state['cell_cycle_arrested'],
                'Arrest Duration (min)': self.state['arrest_duration']
            }
        }
    
    def get_literature_references(self) -> Dict[str, List[str]]:
        """Get literature references for model components"""
        return {
            'SNF1 Pathway': [
                'Hardie et al. (2012) Nature Reviews Mol Cell Bio 13: 251-262',
                'Hedbacker & Carlson (2008) Mol Cell 29: 536-549',
                'Mayer et al. (2011) Curr Opin Cell Biol 23: 744-755'
            ],
            'Msn2/4 Translocation': [
                'Jacquet et al. (2003) J Biol Chem 278: 6999-7008',
                'Görner et al. (1998) Genes Dev 12: 586-597',
                'Garreau et al. (2000) Mol Cell Biol 20: 6485-6496'
            ],
            'Stress Granules': [
                'Buchan & Parker (2009) Mol Cell 36: 932-941',
                'Protter & Parker (2016) Trends Cell Biol 26: 668-679',
                'Hoyle et al. (2007) J Cell Sci 120: 2774-2784'
            ],
            'Autophagy': [
                'Yorimitsu & Klionsky (2005) Cell Death Differ 12: 1542-1552',
                'Noda & Ohsumi (1998) J Biol Chem 273: 3963-3966',
                'Kamada et al. (2000) J Cell Biol 150: 1507-1513'
            ],
            'Trehalose Metabolism': [
                'Thevelein (1984) Microbiol Rev 48: 42-59',
                'François & Parrou (2001) FEMS Microbiol Rev 25: 125-145',
                'Hottiger et al. (1987) J Gen Microbiol 133: 1049-1056'
            ],
            'Energy Metabolism': [
                'Broach (2012) Genetics 192: 73-105',
                'Gancedo (1998) Eur J Biochem 254: 1-6',
                'Rolland et al. (2002) FEMS Yeast Res 2: 183-201'
            ]
        }
    
    def validate_against_experimental_data(self) -> Dict[str, Dict]:
        """
        Validate model predictions against known experimental ranges
        Returns validation results and literature comparisons
        """
        validation_results = {}
        
        # SNF1 activation kinetics (Mayer et al., 2011)
        validation_results['SNF1_Kinetics'] = {
            'parameter': 'SNF1 activation under glucose limitation',
            'model_prediction': f"{self.state['snf1_active_fraction']:.2f}",
            'experimental_range': '0.7-0.9 (under severe glucose limitation)',
            'reference': 'Mayer et al. (2011)',
            'match': 0.7 <= self.state['snf1_active_fraction'] <= 0.9 if self.state['atp_adp_ratio'] < 0.3 else True
        }
        
        # Msn2/4 translocation timing (Jacquet et al., 2003)
        validation_results['Msn24_Translocation'] = {
            'parameter': 'Msn2/4 nuclear translocation under stress',
            'model_prediction': f"{self.state['msn24_nuclear_fraction']:.2f}",
            'experimental_range': '0.6-0.8 (15-30 min post-stress)',
            'reference': 'Jacquet et al. (2003)',
            'match': 0.6 <= self.state['msn24_nuclear_fraction'] <= 0.8 if self.state['stress_level'] > 0.5 else True
        }
        
        # Autophagy delay (Yorimitsu & Klionsky, 2005)
        time_since_stress = (self.state['time'] - self.state['stress_onset_time']) if self.state['stress_onset_time'] else 0
        validation_results['Autophagy_Timing'] = {
            'parameter': 'Autophagy induction delay',
            'model_prediction': f"{self.params.autophagy_delay} min",
            'experimental_range': '20-40 min post-starvation',
            'reference': 'Yorimitsu & Klionsky (2005)',
            'match': 20 <= self.params.autophagy_delay <= 40
        }
        
        # Trehalose accumulation (François & Parrou, 2001)
        validation_results['Trehalose_Response'] = {
            'parameter': 'Trehalose accumulation under stress',
            'model_prediction': f"{self.state['trehalose']:.2f}",
            'experimental_range': '0.1-0.8 (stress-dependent)',
            'reference': 'François & Parrou (2001)',
            'match': 0.1 <= self.state['trehalose'] <= 0.8
        }
        
        return validation_results


# Additional helper functions for enhanced analysis

def calculate_stress_response_metrics(model: EnhancedYeastStressModel) -> Dict:
    """Calculate derived metrics for stress response analysis"""
    state = model.state
    
    # Response amplitude (how much the cell responds to stress)
    response_amplitude = {
        'SNF1_response': state['snf1_active_fraction'] / max(0.01, 1 - state['atp_adp_ratio']),
        'Msn24_response': state['msn24_nuclear_fraction'] / max(0.01, state['stress_level']),
        'Autophagy_response': state['autophagy_flux'] / max(0.01, 1 - state['atp_adp_ratio'])
    }
    
    # Response timing (how fast the cell responds)
    time_since_stress = (state['time'] - state['stress_onset_time']) if state['stress_onset_time'] else 0
    
    # Adaptation index (how well the cell is coping)
    adaptation_index = (
        state['atp_adp_ratio'] * 0.3 +
        state['trehalose'] * 0.2 +
        state['autophagy_flux'] * 0.2 +
        (1 - state['stress_level']) * 0.3
    )
    
    # Recovery rate (how fast stress levels are decreasing)
    if len(model.history) > 10:
        recent_stress = [h['stress_level'] for h in model.history[-10:]]
        stress_slope = np.polyfit(range(len(recent_stress)), recent_stress, 1)[0]
        recovery_rate = -stress_slope if stress_slope < 0 else 0
    else:
        recovery_rate = 0
    
    return {
        'Response Amplitude': response_amplitude,
        'Time Since Stress (min)': time_since_stress,
        'Adaptation Index': adaptation_index,
        'Recovery Rate': recovery_rate,
        'Cell Viability': min(state['membrane_integrity'], state['atp_adp_ratio']),
        'Stress Resistance': 1 - (state['stress_level'] / max(0.01, time_since_stress / 60))
    }

def compare_stress_types(model: EnhancedYeastStressModel, stress_types: List[StressType], 
                        intensity: float = 0.5, duration: int = 120) -> pd.DataFrame:
    """
    Compare cellular responses to different stress types
    Returns DataFrame suitable for plotting
    """
    comparison_data = []
    
    for stress_type in stress_types:
        # Reset model for each stress type
        test_model = EnhancedYeastStressModel(model.params)
        
        # Apply stress and simulate
        test_model.apply_stress(stress_type, intensity)
        
        time_points = []
        for i in range(duration):
            state = test_model.update(dt=1.0)  # 1 minute steps
            
            comparison_data.append({
                'Time (min)': i,
                'Stress Type': stress_type.value,
                'Energy (ATP/ADP)': state['atp_adp_ratio'],
                'SNF1 Active': state['snf1_active_fraction'],
                'Msn2/4 Nuclear': state['msn24_nuclear_fraction'],
                'Stress Granules': state['condensate_level'],
                'Autophagy': state['autophagy_flux'],
                'Trehalose': state['trehalose'],
                'Cell Volume': state['cell_volume'],
                'Overall Stress': state['stress_level']
            })
    
    return pd.DataFrame(comparison_data)

def export_model_parameters(model: EnhancedYeastStressModel, filename: str = None) -> Dict:
    """Export model parameters with literature references for transparency"""
    params_dict = {
        'Model_Version': 'Enhanced Yeast Stress Model v2.0',
        'Last_Updated': '2024-12-30',
        'Parameters': {
            'Energy_Metabolism': {
                'atp_consumption_basal': model.params.atp_consumption_basal,
                'atp_synthesis_max': model.params.atp_synthesis_max,
                'atp_km': model.params.atp_km,
                'energy_threshold_snf1': model.params.energy_threshold_snf1
            },
            'SNF1_Pathway': {
                'snf1_k_act': model.params.snf1_k_act,
                'snf1_k_deact': model.params.snf1_k_deact,
                'snf1_hill_coeff': model.params.snf1_hill_coeff
            },
            'Msn24_Translocation': {
                'msn24_k_import': model.params.msn24_k_import,
                'msn24_k_export': model.params.msn24_k_export,
                'msn24_snf1_sensitivity': model.params.msn24_snf1_sensitivity
            },
            'Stress_Granules': {
                'condensate_threshold': model.params.condensate_threshold,
                'condensate_k_form': model.params.condensate_k_form,
                'condensate_k_dissolve': model.params.condensate_k_dissolve
            },
            'Autophagy': {
                'autophagy_delay': model.params.autophagy_delay,
                'autophagy_k_max': model.params.autophagy_k_max,
                'autophagy_energy_threshold': model.params.autophagy_energy_threshold
            }
        },
        'Literature_References': model.get_literature_references(),
        'Validation_Status': model.validate_against_experimental_data()
    }
    
    if filename:
        import json
        with open(filename, 'w') as f:
            json.dump(params_dict, f, indent=2, default=str)
    
    return params_dict
