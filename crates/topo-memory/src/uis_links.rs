//! U/I/S Links with Stability Formula
//! 
//! Implementation of the Stability-based link system for meaning induction.

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkState { 
    U, // Unstable - новые связи
    I, // Intermediate - частично стабильные
    S  // Stable - стабильные связи
}

#[derive(Debug, Clone)]
pub struct Link {
    pub id: u64,
    pub a: u64, 
    pub b: u64,     // узлы (RVQ-коды/патчи)
    pub state: LinkState,
    pub stability: f32,
    pub r: f32,     // repetition
    pub e: f32,     // energy
    pub c: f32,     // connectivity
    pub phi: f32,   // phase
    pub last_seen_step: usize,
}

impl Link {
    pub fn new(id: u64, a: u64, b: u64) -> Self {
        Self {
            id,
            a,
            b,
            state: LinkState::U,
            stability: 0.0,
            r: 0.0,
            e: 0.0,
            c: 0.0,
            phi: 0.0,
            last_seen_step: 0,
        }
    }

    pub fn update_stability(&mut self, theta_low: f32, theta_high: f32, decay: f32, replay_boost: bool) {
        // Формула Stability: S = (R + C).ln_1p() + E + Φ - decay
        let base = (self.r.ln_1p() + self.c.ln_1p()) + self.e + self.phi;
        let boost = if replay_boost { 0.1 } else { 0.0 };
        
        // Обновляем стабильность с затуханием
        self.stability = (self.stability * (1.0 - decay)) + base + boost;
        
        // Обновляем состояние на основе стабильности
        self.state = if self.stability >= theta_high { 
            LinkState::S 
        } else if self.stability >= theta_low { 
            LinkState::I 
        } else { 
            LinkState::U 
        };
    }

    pub fn update_signals(&mut self, r: f32, e: f32, c: f32, phi: f32) {
        // Обновляем сигналы с экспоненциальным сглаживанием
        let alpha = 0.1; // коэффициент сглаживания
        
        self.r = (1.0 - alpha) * self.r + alpha * r;
        self.e = (1.0 - alpha) * self.e + alpha * e;
        self.c = (1.0 - alpha) * self.c + alpha * c;
        self.phi = (1.0 - alpha) * self.phi + alpha * phi;
    }
}

#[derive(Debug, Clone)]
pub struct StabilityParams {
    pub theta_low: f32,
    pub theta_high: f32,
    pub decay: f32,
}

impl StabilityParams {
    pub fn new(theta_low: f32, theta_high: f32, decay: f32) -> Self {
        Self {
            theta_low,
            theta_high,
            decay,
        }
    }
}

#[derive(Debug, Default)]
pub struct LinkManager {
    pub links: RwLock<HashMap<u64, Link>>,
    pub u_links: RwLock<Vec<u64>>,
    pub i_links: RwLock<Vec<u64>>,
    pub s_links: RwLock<Vec<u64>>,
    pub step_counter: RwLock<usize>,
}

impl LinkManager {
    pub fn new() -> Self {
        Self {
            links: RwLock::new(HashMap::new()),
            u_links: RwLock::new(Vec::new()),
            i_links: RwLock::new(Vec::new()),
            s_links: RwLock::new(Vec::new()),
            step_counter: RwLock::new(0),
        }
    }

    pub fn add_u_link(&self, link: Link) -> Result<()> {
        let link_id = link.id;
        
        // Добавляем в общую карту связей
        self.links.write().unwrap().insert(link_id, link);
        
        // Добавляем в список U-связей
        self.u_links.write().unwrap().push(link_id);
        
        Ok(())
    }

    pub fn get_link(&self, link_id: u64) -> Result<Option<Link>> {
        Ok(self.links.read().unwrap().get(&link_id).cloned())
    }

    pub fn get_link_mut(&self, link_id: u64) -> Result<Option<Link>> {
        Ok(self.links.read().unwrap().get(&link_id).cloned())
    }

    pub fn update_link(&self, mut link: Link) -> Result<()> {
        let link_id = link.id;
        let old_state = self.links.read().unwrap().get(&link_id).map(|l| l.state);
        
        // Обновляем связь
        self.links.write().unwrap().insert(link_id, link.clone());
        
        // Обновляем списки состояний если нужно
        if let Some(old_state) = old_state {
            if old_state != link.state {
                self.update_link_lists(link_id, old_state, link.state)?;
            }
        }
        
        Ok(())
    }

    pub fn update_link_lists(&self, link_id: u64, old_state: LinkState, new_state: LinkState) -> Result<()> {
        // Удаляем из старого списка
        match old_state {
            LinkState::U => {
                self.u_links.write().unwrap().retain(|&id| id != link_id);
            },
            LinkState::I => {
                self.i_links.write().unwrap().retain(|&id| id != link_id);
            },
            LinkState::S => {
                self.s_links.write().unwrap().retain(|&id| id != link_id);
            }
        }

        // Добавляем в новый список
        match new_state {
            LinkState::U => {
                self.u_links.write().unwrap().push(link_id);
            },
            LinkState::I => {
                self.i_links.write().unwrap().push(link_id);
            },
            LinkState::S => {
                self.s_links.write().unwrap().push(link_id);
            }
        }

        Ok(())
    }

    pub fn sample_recent_u(&self, k: usize) -> Result<Vec<u64>> {
        let u_links = self.u_links.read().unwrap();
        let count = std::cmp::min(k, u_links.len());
        Ok(u_links.iter().rev().take(count).cloned().collect())
    }

    pub fn get_all_u_links(&self) -> Result<Vec<u64>> {
        Ok(self.u_links.read().unwrap().clone())
    }

    pub fn update_links_with_signals(&self, r: f32, e: f32, c: f32, phi: f32, sp: &StabilityParams) -> Result<()> {
        let mut step_counter = self.step_counter.write().unwrap();
        *step_counter += 1;
        let current_step = *step_counter;
        drop(step_counter);

        // Собираем изменения состояния
        let mut state_changes = Vec::new();
        
        {
            let mut links = self.links.write().unwrap();
            for (_, link) in links.iter_mut() {
                // Обновляем сигналы
                let alpha = 0.1; // коэффициент сглаживания
                link.r = (1.0 - alpha) * link.r + alpha * r;
                link.e = (1.0 - alpha) * link.e + alpha * e;
                link.c = (1.0 - alpha) * link.c + alpha * c;
                link.phi = (1.0 - alpha) * link.phi + alpha * phi;
                link.last_seen_step = current_step;

                // Обновляем стабильность
                let old_state = link.state;
                link.update_stability(sp.theta_low, sp.theta_high, sp.decay, false);
                
                // Запоминаем изменения состояния
                if old_state != link.state {
                    state_changes.push((link.id, old_state, link.state));
                }
            }
        }

        // Применяем изменения состояния
        for (link_id, old_state, new_state) in state_changes {
            self.update_link_lists(link_id, old_state, new_state)?;
        }

        Ok(())
    }

    pub fn sweep_and_consolidate(&self, sp: &StabilityParams, replay_boost: bool) -> Result<()> {
        let mut step_counter = self.step_counter.write().unwrap();
        *step_counter += 1;
        drop(step_counter);

        // Собираем изменения состояния
        let mut state_changes = Vec::new();
        
        {
            let mut links = self.links.write().unwrap();
            for (_, link) in links.iter_mut() {
                let old_state = link.state;
                link.update_stability(sp.theta_low, sp.theta_high, sp.decay, replay_boost);
                
                if old_state != link.state {
                    state_changes.push((link.id, old_state, link.state));
                }
            }
        }

        // Применяем изменения состояния
        for (link_id, old_state, new_state) in state_changes {
            self.update_link_lists(link_id, old_state, new_state)?;
        }

        Ok(())
    }

    pub fn get_stats(&self) -> LinkStats {
        let links = self.links.read().unwrap();
        let u_count = self.u_links.read().unwrap().len();
        let i_count = self.i_links.read().unwrap().len();
        let s_count = self.s_links.read().unwrap().len();

        let mut total_stability = 0.0;
        let mut count = 0;

        for link in links.values() {
            total_stability += link.stability;
            count += 1;
        }

        let avg_stability = if count > 0 { total_stability / count as f32 } else { 0.0 };

        LinkStats {
            total_links: count,
            u_links: u_count,
            i_links: i_count,
            s_links: s_count,
            avg_stability,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinkStats {
    pub total_links: usize,
    pub u_links: usize,
    pub i_links: usize,
    pub s_links: usize,
    pub avg_stability: f32,
}

// Trait для интеграции с TopologicalMemory
pub trait Topo {
    fn add_u_link(&mut self, link: Link) -> Result<()>;
    fn get_link(&self, link_id: u64) -> Result<Option<Link>>;
    fn get_link_mut(&self, link_id: u64) -> Result<Option<Link>>;
    fn update_link(&self, link: Link) -> Result<()>;
    fn sample_recent_u(&self, k: usize) -> Result<Vec<u64>>;
    fn get_all_u_links(&self) -> Result<Vec<u64>>;
    fn update_links_with_signals(&self, r: f32, e: f32, c: f32, phi: f32, sp: &StabilityParams) -> Result<()>;
    fn sweep_and_consolidate(&self, sp: &StabilityParams, replay_boost: bool) -> Result<()>;
}
