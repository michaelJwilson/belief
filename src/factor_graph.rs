use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Emission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Emission,
    Transition,
    Prior,
    Custom,
}

#[derive(Clone)]
pub struct Variable {
    pub id: usize,
    pub var_type: VariableType,
    pub priority: Option<usize>,
}

pub struct Factor {
    pub id: usize,
    pub variables: Vec<usize>,
    pub table: Vec<f64>, // flattened table, row-major, log-probabilities for all clique assignments.
    pub factor_type: FactorType,
    pub priority: Option<usize>,
}

/// NB Forney factor graph is with variables and factors connected by edges.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>, // map: variable id -> factor ids.
    pub factor_to_vars: HashMap<usize, Vec<usize>>, // map: factor id -> variable ids.
    pub domain_size: usize,
}

impl FactorGraph {
    pub fn sort_by_priority(&mut self) {
        if self.variables.iter().any(|v| v.priority.is_some()) {
            self.variables.sort_by_key(|v| v.priority.unwrap_or(usize::MAX));
        }
        
        // Derive factor priority from connected variables.
        // Rule: Factor priority = max(variable_priority) among connected variables.
        // Meaning the factor is processed after its latest variable (in a forward pass logic).
        // If variable priorities are missing, we treat them as 0 for this calculation or ignore.
        
        // First, build a map of variable id -> priority for quick lookup
        let var_priorities: HashMap<usize, usize> = self.variables.iter()
            .map(|v| (v.id, v.priority.unwrap_or(usize::MAX)))
            .collect();

        for factor in &mut self.factors {
            // Calculate priority based on connected variables
            let max_p = factor.variables.iter()
                .map(|vid| var_priorities.get(vid).copied().unwrap_or(usize::MAX))
                .max();
            
            // If all variables have priority (not MAX), we assign the max.
            // If any is MAX (missing), the factor priority effectively becomes MAX (sorted last).
            if let Some(p) = max_p {
                factor.priority = Some(p);
            }
        }

        if self.factors.iter().any(|f| f.priority.is_some()) {
            self.factors.sort_by_key(|f| f.priority.unwrap_or(usize::MAX));
        }
    }
}
