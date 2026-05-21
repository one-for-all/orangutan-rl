use burn::{
    Tensor,
    backend::{Autodiff, Wgpu},
    optim::{AdamConfig, AdamWConfig},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use orangutan_rl::{
    agent::Agent,
    env::Env,
    model::ModelConfig,
    training::{self, TrainingConfig},
};

fn computation<B: Backend>() {
    let device = Default::default();
    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);

    println!("{:}", tensor1 + tensor2);
}

fn main() {
    let mut env = Env::new();
    let mut agent = Agent::new();
    // let mut optimizer = AdamWConfig::new().init();

    let next_obs = env.reset();

    // for episode in 0..5 {
    //     let mut episode_done = false;
    //     let mut episode_reward = 0.0;
    //     let mut episode_duration = 0_usize;

    //     env.reset();
    //     while !episode_done {
    //         let state = env.state();
    //     }
    // }

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
