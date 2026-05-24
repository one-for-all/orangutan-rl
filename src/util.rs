use burn::{
    Tensor,
    optim::{GradientsParams, LearningRate, Optimizer},
    prelude::Backend,
    tensor::{Int, backend::AutodiffBackend},
};

use crate::ppo::PPOModel;

pub(crate) fn to_state_tensor<B: Backend>(state: isize) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([state as f32], &Default::default())
}

pub(crate) fn ref_to_state_tensor<B: Backend>(state: &isize) -> Tensor<B, 1> {
    to_state_tensor(*state)
}

pub(crate) fn get_elem<B: Backend, const D: usize>(i: usize, tensor: &Tensor<B, D>) -> Option<f32> {
    tensor.to_data().as_slice().ok()?.get(i).copied()
}

pub(crate) fn to_action_tensor<B: Backend>(action: usize) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_ints([action as i32], &Default::default())
}

pub(crate) fn ref_to_action_tensor<B: Backend>(action: &usize) -> Tensor<B, 1, Int> {
    to_action_tensor(*action)
}

pub(crate) fn to_reward_tensor<B: Backend>(reward: impl Into<f32> + Clone) -> Tensor<B, 1> {
    Tensor::from_floats([reward.into()], &Default::default())
}

pub(crate) fn ref_to_reward_tensor<B: Backend>(reward: &(impl Into<f32> + Clone)) -> Tensor<B, 1> {
    to_reward_tensor(reward.clone())
}
pub(crate) fn to_not_done_tensor<B: Backend>(done: bool) -> Tensor<B, 1> {
    Tensor::from_floats([if done { 0.0 } else { 1.0 }], &Default::default())
}

pub(crate) fn ref_to_not_done_tensor<B: Backend>(done: &bool) -> Tensor<B, 1> {
    to_not_done_tensor(*done)
}

pub(crate) fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}

pub(crate) fn update_parameters<B: AutodiffBackend>(
    loss: Tensor<B, 1>,
    module: PPOModel<B>,
    optimizer: &mut impl Optimizer<PPOModel<B>, B>,
    learning_rate: LearningRate,
) -> PPOModel<B> {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}

pub fn vec2d_to_tensor<B: Backend>(data: Vec<Vec<f32>>, device: &B::Device) -> Tensor<B, 2> {
    let rows = data.len();
    let cols = data[0].len();

    // Validate all rows have the same length
    assert!(
        data.iter().all(|r| r.len() == cols),
        "All rows must have equal length"
    );

    let flat: Vec<f32> = data.into_iter().flatten().collect();

    Tensor::<B, 1>::from_data(flat.as_slice(), device).reshape([rows, cols])
}
