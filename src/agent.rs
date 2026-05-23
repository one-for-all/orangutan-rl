use burn::prelude::Backend;

use crate::ppo::{PPOModel, react_with_model};

pub struct Agent<B: Backend> {
    pub q_table: [[f32; 2]; 5],

    pub model: Option<PPOModel<B>>,
}

impl<B: Backend> Agent<B> {
    pub fn new() -> Self {
        Self {
            q_table: [[0.; 2]; 5], // 5 states, 2 actions
            model: None,
        }
    }

    pub fn get_action_and_value(&self, _obs: isize) -> (isize, f32, f32) {
        (1, 0., 0.)
    }

    /// 0 -> move left, 1 -> move left
    /// For now, fix the action to be moving left
    pub fn react(&mut self, state: &isize) -> usize {
        if let Some(model) = &self.model {
            react_with_model(state, model)
        } else {
            panic!()
        }
    }
}
