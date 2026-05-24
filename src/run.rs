use burn::{
    grad_clipping::GradientClippingConfig, optim::AdamWConfig, tensor::backend::AutodiffBackend,
};

use crate::{
    agent::Agent,
    env::toy::ToyEnv,
    memory::Memory,
    ppo::{PPOModel, react_with_model, train},
};

pub fn run<B: AutodiffBackend>() -> Agent<B> {
    let mut env = ToyEnv::new();

    let mut ppo_model = PPOModel::<B>::new(1, 4, 2);
    let mut agent = Agent::new();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(100.)))
        .init();
    let mut memory = Memory::<512>::default();
    for episode in 0..128 {
        let mut episode_done = false;
        let mut episode_reward = 0.0;
        let mut episode_duration = 0_usize;

        env.reset();
        while !episode_done {
            let state = env.state();
            let action = react_with_model(&state, &ppo_model);
            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;

            memory.push(state, next_state, action, reward, done);
            episode_duration += 1;
            episode_done = done || episode_duration >= 100;
        }

        println!(
            "{{\"episode\": {episode}, \"reward\": {episode_reward:.4}, \"duration\": {episode_duration}}}",
        );

        ppo_model = train::<B, 512>(ppo_model, &memory, &mut optimizer);
        memory.clear();
    }

    agent.model = Some(ppo_model);
    agent
}
