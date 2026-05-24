use burn::{
    Tensor,
    backend::{Autodiff, NdArray, Wgpu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{ElementConversion, backend::AutodiffBackend},
};
use orangutan_rl::{
    env::{dynamic::DoubleIntegratorEnv, grid::OneDimGridEnv},
    plot::plot,
    simple::{Categorical, SimpleLogitsNet},
    util::vec2d_to_tensor,
};

const BATCH_SIZE: usize = 20;
const EPOCHS: usize = 1000;

// type MyBackend = Autodiff<Wgpu>;
type MyBackend = Autodiff<NdArray>;

fn main() {
    MyBackend::seed(&Default::default(), 0);

    // let mut env = OneDimGridEnv::new();
    let mut env = DoubleIntegratorEnv::new();

    let obs_dim = env.obs_dim();
    let n_acts = env.n_acts();

    // policy network
    let mut logits_net = SimpleLogitsNet::<MyBackend>::new([obs_dim, 32, n_acts]);

    let mut optimizer = AdamConfig::new().init();

    let mut data = vec![];

    for epoch in 0..EPOCHS {
        println!("====== epoch: {epoch}");
        let ret;
        (logits_net, ret) = train_one_epoch(&mut env, logits_net, &mut optimizer);
        data.push(ret);

        let policy = get_policy(
            &logits_net,
            Tensor::from_data([[0., 0.], [1., 0.]], &Default::default()),
        );
        let action_probs_formatted: Vec<String> = policy
            .probs()
            .into_data()
            .iter()
            .map(|f: f32| format!("{:.2}", f))
            .collect();
        println!("action probs: {:?}", action_probs_formatted);
    }

    // Roll out a policy
    println!("======= Policy Rollout");
    let mut data2 = vec![];
    let mut obs = env.reset(true);
    data2.push(obs[0]);
    let mut ret = 0.;
    while env.t < 10. {
        let obs_tensor = vec2d_to_tensor(vec![obs.clone()], &Default::default());
        let act = get_action(&logits_net, obs_tensor);
        let (next_obs, rew, _done) = env.step(act);

        ret += rew;
        let obs_formatted: Vec<String> = obs.iter().map(|f| format!("{:.2}", f)).collect();
        println!("obs: {:?}, action: {}, reward: {}", obs_formatted, act, rew);

        obs = next_obs;
        data2.push(obs[0]);
    }
    println!("total return: {ret}");

    plot(&data, 1.0, "simple RL");
    plot(&data2, env.dt, "double integrator trajectory");

    // println!("======= Summary");
    // let policy = get_policy(&logits_net, Tensor::from_data([[0.]], &Default::default()));
    // println!("action probs: {}", policy.probs().into_data());
}

fn get_policy<B: Backend>(logits_net: &SimpleLogitsNet<B>, obs: Tensor<B, 2>) -> Categorical<B> {
    let logits = logits_net.forward(obs);
    Categorical::new(logits)
}

/// Given a single observation of shape [1, obs_dim], return a single action
fn get_action<B: Backend>(logits_net: &SimpleLogitsNet<B>, obs: Tensor<B, 2>) -> i32 {
    assert_eq!(obs.shape()[0], 1);
    let dist = get_policy(logits_net, obs);
    dist.sample().into_scalar().elem()
}

fn compute_loss<B: Backend>(
    logits_net: &SimpleLogitsNet<B>,
    obs: Tensor<B, 2>,
    acts: Tensor<B, 1, burn::tensor::Int>,
    weights: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let logp = get_policy(logits_net, obs).log_prob(acts);
    -(logp * weights).mean()
}

fn train_one_epoch<B: AutodiffBackend>(
    env: &mut DoubleIntegratorEnv,
    logits_net: SimpleLogitsNet<B>,
    optimizer: &mut impl Optimizer<SimpleLogitsNet<B>, B>,
) -> (SimpleLogitsNet<B>, f32) {
    let mut batch_obs = vec![];
    let mut batch_acts = vec![];
    let mut batch_weights = vec![];
    let mut batch_rets = vec![];
    let mut batch_lens = vec![];

    let mut obs = env.reset(true);
    let mut done;
    let mut ep_rews = vec![];

    loop {
        batch_obs.push(obs.clone());

        let obs_tensor = vec2d_to_tensor(vec![obs], &Default::default());
        let act = get_action(&logits_net, obs_tensor);
        let (next_obs, rew, next_done) = env.step(act);
        done = next_done;

        batch_acts.push(act);
        ep_rews.push(rew);

        // println!("obs: {obs}, action: {act}, reward: {rew}, done: {done}");
        obs = next_obs;

        if done {
            let ep_ret: f32 = ep_rews.iter().sum();
            let ep_len = ep_rews.len();
            batch_rets.push(ep_ret);
            batch_lens.push(ep_len);

            batch_weights.extend(reward_to_go(&ep_rews));

            obs = env.reset(true);
            ep_rews.clear();

            if batch_obs.len() >= BATCH_SIZE {
                break;
            }
        }
    }

    let obs_tensor = vec2d_to_tensor(batch_obs, &Default::default());
    let acts_tensor = Tensor::from_data(batch_acts.as_slice(), &Default::default());
    let weights_tensor = Tensor::<B, 1>::from_data(batch_weights.as_slice(), &Default::default());

    let loss = compute_loss(&logits_net, obs_tensor, acts_tensor, weights_tensor);

    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &logits_net);
    let optimized_logits_net = optimizer.step(1e-2, logits_net, gradient_params);

    (
        optimized_logits_net,
        batch_rets.iter().sum::<f32>() / batch_rets.len() as f32,
    )
}

/// Compute the reward to go, i.e. the sum of rewards after the action, for each action
/// rewards is the list of rewards after each action
fn reward_to_go(rewards: &Vec<f32>) -> Vec<f32> {
    let n = rewards.len();
    let mut rtgs = vec![0.; n];

    let gamma = 0.99; // discount factor

    rtgs[n - 1] = rewards[n - 1];
    for i in (0..n - 1).rev() {
        rtgs[i] = rewards[i] + gamma * rtgs[i + 1];
    }
    rtgs
}
