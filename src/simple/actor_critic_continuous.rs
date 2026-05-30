use burn::{Tensor, module::Module, prelude::Backend, tensor::ElementConversion};

use crate::simple::{
    Normal,
    actor_critic::{MLP, MLPCritic},
};

pub struct MLPActorCriticContinuous<B: Backend> {
    pub pi: MLPGaussianActor<B>,
    pub v: MLPCritic<B>,
}

impl<B: Backend> MLPActorCriticContinuous<B> {
    pub fn new(obs_dim: usize, act_dim: usize, hidden_sizes: &Vec<usize>) -> Self {
        let pi = MLPGaussianActor::new(obs_dim, act_dim, hidden_sizes);
        let v = MLPCritic::new(obs_dim, hidden_sizes);
        Self { pi, v }
    }

    /// Take a step, and return (action, value estimate, log probability of the action)
    pub fn step(&self, obs: Tensor<B, 2>) -> (f32, f32, f32) {
        assert_eq!(obs.shape()[0], 1);
        let pi = self.pi.distribution(obs.clone());
        let a: f32 = pi.sample().into_scalar().elem();
        let logp_a = self
            .pi
            .log_prob_from_distribution(&pi, Tensor::<B, 1>::from_data([a], &Default::default()))
            .into_scalar()
            .elem();
        let v = self.v.forward(obs).into_scalar().elem();
        (a, v, logp_a)
    }

    pub fn act(&self, obs: Tensor<B, 2>) -> f32 {
        assert_eq!(obs.shape()[0], 1);
        let pi = self.pi.distribution(obs.clone());
        let a = pi.sample().into_scalar().elem();
        a
    }
}

#[derive(Module, Debug)]
pub struct MLPGaussianActor<B: Backend> {
    mu_net: MLP<B>,
    log_std: f32,
}

impl<B: Backend> MLPGaussianActor<B> {
    pub fn new(obs_dim: usize, act_dim: usize, hidden_sizes: &Vec<usize>) -> Self {
        let mut sizes = vec![obs_dim];
        sizes.extend(hidden_sizes);
        assert_eq!(act_dim, 1);
        sizes.push(act_dim);

        let mu_net = MLP::new(&sizes);
        Self {
            mu_net,
            log_std: -0.5,
        }
    }

    pub fn distribution(&self, obs: Tensor<B, 2>) -> Normal<B> {
        let mu = self.mu_net.forward(obs).squeeze_dim(1);
        let std = self.log_std.exp();

        Normal::new(mu, std)
    }

    pub fn log_prob_from_distribution(&self, pi: &Normal<B>, action: Tensor<B, 1>) -> Tensor<B, 1> {
        pi.log_prob(action)
    }
}
