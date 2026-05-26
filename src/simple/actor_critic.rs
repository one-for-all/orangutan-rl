use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Tanh},
    prelude::Backend,
    tensor::{ElementConversion, Int, activation},
};

use crate::simple::Categorical;

pub struct MLPActorCritic<B: Backend> {
    pub pi: MLPCategoricalActor<B>,
    pub v: MLPCritic<B>,
}

impl<B: Backend> MLPActorCritic<B> {
    pub fn new(obs_dim: usize, n_acts: usize, hidden_sizes: &Vec<usize>) -> Self {
        let pi = MLPCategoricalActor::new(obs_dim, n_acts, hidden_sizes);
        let v = MLPCritic::new(obs_dim, hidden_sizes);
        Self { pi, v }
    }

    /// Take a step, and return (action, value estimate, log probability of the action)
    pub fn step(&self, obs: Tensor<B, 2>) -> (i32, f32, f32) {
        assert_eq!(obs.shape()[0], 1);
        let pi = self.pi.distribution(obs.clone());
        let a = pi.sample().into_scalar().elem();
        let logp_a = self
            .pi
            .log_prob_from_distribution(
                &pi,
                Tensor::<B, 1, Int>::from_data([a], &Default::default()),
            )
            .into_scalar()
            .elem();
        let v = self.v.forward(obs).into_scalar().elem();
        (a, v, logp_a)
    }

    pub fn act(&self, obs: Tensor<B, 2>) -> i32 {
        assert_eq!(obs.shape()[0], 1);
        let pi = self.pi.distribution(obs.clone());
        let a = pi.sample().into_scalar().elem();
        a
    }
}

#[derive(Module, Debug)]
pub struct MLPCategoricalActor<B: Backend> {
    logits_net: MLP<B>,
}

impl<B: Backend> MLPCategoricalActor<B> {
    pub fn new(obs_dim: usize, n_acts: usize, hidden_sizes: &Vec<usize>) -> Self {
        let mut sizes = vec![obs_dim];
        sizes.extend(hidden_sizes);
        sizes.push(n_acts);

        let logits_net = MLP::new(&sizes);
        Self { logits_net }
    }

    pub fn distribution(&self, obs: Tensor<B, 2>) -> Categorical<B> {
        let logits = self.logits_net.forward(obs);
        Categorical::new(logits)
    }

    pub fn log_prob_from_distribution(
        &self,
        pi: &Categorical<B>,
        action: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        pi.log_prob(action)
    }
}

#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    v_net: MLP<B>,
}

impl<B: Backend> MLPCritic<B> {
    pub fn new(obs_dim: usize, hidden_sizes: &Vec<usize>) -> Self {
        let mut sizes = vec![obs_dim];
        sizes.extend(hidden_sizes);
        sizes.push(1);

        let v_net = MLP::new(&sizes);
        Self { v_net }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        self.v_net.forward(obs).squeeze_dim(1)
    }
}

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    linear_layers: Vec<Linear<B>>,
    activation: Tanh,
}

impl<B: Backend> MLP<B> {
    pub fn new(sizes: &Vec<usize>) -> Self {
        let mut linear_layers = vec![];
        for i in 0..sizes.len() - 1 {
            linear_layers
                .push(LinearConfig::new(sizes[i], sizes[i + 1]).init::<B>(&Default::default()));
        }
        let activation = Tanh::new();
        Self {
            linear_layers,
            activation,
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = obs;
        for i in 0..self.linear_layers.len() - 1 {
            x = self.linear_layers[i].forward(x);
            x = self.activation.forward(x);
        }
        self.linear_layers.last().unwrap().forward(x)
    }
}
