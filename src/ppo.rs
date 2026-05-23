use burn::{
    Tensor,
    module::Module,
    nn::{
        Initializer, Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::Optimizer,
    prelude::Backend,
    tensor::{
        activation::{relu, softmax},
        backend::AutodiffBackend,
    },
};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rng,
};

use crate::{
    memory::{Memory, get_batch, sample_indices},
    util::{
        elementwise_min, get_elem, ref_to_action_tensor, ref_to_not_done_tensor,
        ref_to_reward_tensor, ref_to_state_tensor, update_parameters,
    },
};

#[derive(Module, Debug)]
pub struct PPOModel<B: Backend> {
    linear: Linear<B>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> PPOModel<B> {
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        let initializer = Initializer::XavierUniform { gain: 1.0 };
        Self {
            linear: LinearConfig::new(input_size, dense_size)
                .with_initializer(initializer.clone())
                .init(&Default::default()),
            linear_actor: LinearConfig::new(dense_size, output_size)
                .with_initializer(initializer.clone())
                .init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1)
                .with_initializer(initializer)
                .init(&Default::default()),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let layer0_output = relu(self.linear.forward(input));
        let policies = softmax(self.linear_actor.forward(layer0_output.clone()), 1);
        let values = self.linear_critic.forward(layer0_output);

        (policies, values)
    }
}

pub fn react_with_model<B: Backend>(state: &isize, model: &PPOModel<B>) -> usize {
    let state_tensor = Tensor::<B, 1>::from_floats([(*state as f32)], &Default::default());
    let (policies, _values) = model.forward(state_tensor.clone().unsqueeze());

    let prob = policies.to_data().to_vec::<f32>().unwrap();
    println!("probs: {:?}", prob);

    let Some(dist) = WeightedIndex::new(prob.clone()).ok() else {
        println!("state_tensor: {}", state_tensor);
        println!("policies: {}", policies);
        println!("probs: {:?}", prob);
        panic!()
    };

    let mut rng = rng();
    dist.sample(&mut rng) as usize
}

pub fn train<B: AutodiffBackend, const CAP: usize>(
    mut ppo_model: PPOModel<B>,
    memory: &Memory<CAP>,
    optimizer: &mut (impl Optimizer<PPOModel<B>, B> + Sized),
) -> PPOModel<B> {
    let memory_indices = (0..memory.len()).collect::<Vec<usize>>();
    let (old_policies, old_values) = ppo_model.forward(get_batch(
        &memory.state,
        &memory_indices,
        ref_to_state_tensor,
    ));

    let (expected_returns, advantages) = get_gae(
        old_values,
        get_batch(&memory.reward, &memory_indices, ref_to_reward_tensor),
        get_batch(&memory.done, &memory_indices, ref_to_not_done_tensor),
        0.99,
        0.95,
    );

    for _ in 0..8 {
        for _ in 0..(memory.len() / 1) {
            let sample_indices = sample_indices(memory_indices.clone(), 1);

            let sample_indices_tensor = Tensor::from_ints(
                sample_indices
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &Default::default(),
            );

            let state_batch = get_batch(&memory.state, &sample_indices, ref_to_state_tensor);
            let action_batch = get_batch(&memory.action, &sample_indices, ref_to_action_tensor);
            let old_policy_batch = old_policies
                .clone()
                .select(0, sample_indices_tensor.clone());
            let advantage_batch = advantages.clone().select(0, sample_indices_tensor.clone());
            let expected_return_batch = expected_returns
                .clone()
                .select(0, sample_indices_tensor)
                .detach();

            let (policy_batch, value_batch) = ppo_model.forward(state_batch);

            let ratios = policy_batch
                .clone()
                .div(old_policy_batch)
                .gather(1, action_batch);
            let clipped_ratios = ratios.clone().clamp(1.0 - 0.2, 1.0 + 0.2);

            let actor_loss = -elementwise_min(
                ratios * advantage_batch.clone(),
                clipped_ratios * advantage_batch,
            )
            .sum();
            let critic_loss = MseLoss.forward(expected_return_batch, value_batch, Reduction::Sum);
            let policy_negative_entropy = -(policy_batch.clone().log() * policy_batch)
                .sum_dim(1)
                .mean();

            let loss =
                actor_loss + critic_loss.mul_scalar(0.5) + policy_negative_entropy.mul_scalar(0.01);

            ppo_model = update_parameters(loss, ppo_model, optimizer, 0.001);
        }
    }
    ppo_model
}

pub fn get_gae<B: Backend>(
    values: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    not_dones: Tensor<B, 2>,
    gamma: f32,
    lambda: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut returns = vec![0.; rewards.shape().num_elements()];
    let mut advantages = returns.clone();

    let mut running_return = 0.;
    let mut running_advantage = 0.;
    for i in (0..rewards.shape().num_elements()).rev() {
        let reward = get_elem(i, &rewards).unwrap();
        let not_done = get_elem(i, &not_dones).unwrap();

        running_return = reward + gamma * running_return * not_done;
        running_advantage = reward - get_elem(i, &values).unwrap()
            + gamma
                * not_done
                * (get_elem(i + 1, &values).unwrap_or(0.0) + lambda * running_advantage);

        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    (
        Tensor::<B, 1>::from_floats(returns.as_slice(), &Default::default())
            .reshape([returns.len(), 1]),
        Tensor::<B, 1>::from_floats(advantages.as_slice(), &Default::default())
            .reshape([advantages.len(), 1]),
    )
}
