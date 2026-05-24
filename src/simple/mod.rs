use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, Tanh},
    prelude::Backend,
    tensor::{Distribution, activation::log_softmax},
};

#[derive(Module, Debug)]
pub struct SimpleLogitsNet<B: Backend> {
    obs_layer: Linear<B>,
    activation: Tanh,
    action_layer: Linear<B>,
}

impl<B: Backend> SimpleLogitsNet<B> {
    pub fn new(sizes: [usize; 3]) -> Self {
        let [obs_dim, hidden_dim, n_acts] = sizes;
        let obs_layer = LinearConfig::new(obs_dim, hidden_dim).init::<B>(&Default::default());
        let activation = Tanh::new();
        let action_layer = LinearConfig::new(hidden_dim, n_acts).init(&Default::default());

        Self {
            obs_layer,
            activation,
            action_layer,
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.obs_layer.forward(obs);
        let x = self.activation.forward(x);
        self.action_layer.forward(x)
    }
}

pub struct Categorical<B: Backend> {
    log_probs: Tensor<B, 2>, // log probabilities of each action. shape [batch_size, n_acts]
}

impl<B: Backend> Categorical<B> {
    pub fn new(logits: Tensor<B, 2>) -> Self {
        let log_probs = log_softmax(logits, 1);
        Self { log_probs }
    }

    /// Sample one action per batch
    /// Returns a tensor of shape [batch_size]
    pub fn sample(&self) -> Tensor<B, 1, burn::tensor::Int> {
        let gumbel = -(-self
            .log_probs
            .random_like(Distribution::Uniform(1e-20, 1.))
            .log())
        .log();
        (self.log_probs.clone() + gumbel).argmax(1).squeeze_dim(1)
    }

    pub fn log_prob(&self, actions: Tensor<B, 1, burn::tensor::Int>) -> Tensor<B, 1> {
        let batch_size = self.log_probs.dims()[0];
        let actions_2d = actions.reshape([batch_size, 1]);
        let gathered = self.log_probs.clone().gather(1, actions_2d);
        gathered.reshape([batch_size])
    }

    pub fn probs(&self) -> Tensor<B, 2> {
        self.log_probs.clone().exp()
    }
}
