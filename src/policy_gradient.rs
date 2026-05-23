use burn::{
    Tensor,
    backend::{Autodiff, Wgpu},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Distribution, ElementConversion, backend::AutodiffBackend},
};
use orangutan_rl::{
    plot::plot,
    simple::{Categorical, SimpleLogitsNet},
};

const BATCH_SIZE: usize = 20;
const EPOCHS: usize = 50;

pub struct SimpleEnv {
    x: f32,

    i_step: usize, // max 5 steps
}

impl SimpleEnv {
    pub fn new() -> Self {
        Self { x: 0., i_step: 0 }
    }

    /// 1-dim continous observation
    pub fn obs_dim(&self) -> usize {
        1
    }

    /// 2 discrete actions
    pub fn n_acts(&self) -> usize {
        2
    }

    pub fn reset(&mut self) -> f32 {
        self.x = 0.;
        self.i_step = 0;
        self.x
    }

    pub fn step(&mut self, action: i32) -> (f32, f32, bool) {
        let mut reward = -1.;
        let mut done = false;
        match action {
            0 => {}
            1 => {
                self.x += 1.;
                if self.x == 3. {
                    reward = 10.;
                    done = true;
                }
            }
            _ => panic!("unknown action: {action}"),
        }

        let obs = self.x;
        self.i_step += 1;
        if self.i_step >= 5 {
            done = true;
        }
        return (obs, reward, done);
    }
}

type MyBackend = Autodiff<Wgpu>;

fn main() {
    MyBackend::seed(&Default::default(), 0);

    let mut env = SimpleEnv::new();

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
            Tensor::from_data([[0.], [1.], [2.]], &Default::default()),
        );
        println!("action probs: {}", policy.probs().into_data());
    }

    plot(&data, 1.0, "simple RL");

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
    env: &mut SimpleEnv,
    logits_net: SimpleLogitsNet<B>,
    optimizer: &mut impl Optimizer<SimpleLogitsNet<B>, B>,
) -> (SimpleLogitsNet<B>, f32) {
    let mut batch_obs = vec![];
    let mut batch_acts = vec![];
    let mut batch_weights = vec![];
    let mut batch_rets = vec![];
    let mut batch_lens = vec![];

    let mut obs = env.reset();
    let mut done;
    let mut ep_rews = vec![];

    loop {
        batch_obs.push(obs);

        let act = get_action(&logits_net, Tensor::from_data([[obs]], &Default::default()));
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

            batch_weights.extend(vec![ep_ret; ep_len]);

            obs = env.reset();
            ep_rews.clear();

            if batch_obs.len() >= BATCH_SIZE {
                break;
            }
        }
    }

    let obs_tensor = Tensor::<B, 1>::from_data(batch_obs.as_slice(), &Default::default());
    let acts_tensor = Tensor::from_data(batch_acts.as_slice(), &Default::default());
    let weights_tensor = Tensor::<B, 1>::from_data(batch_weights.as_slice(), &Default::default());

    let loss = compute_loss(
        &logits_net,
        obs_tensor.reshape([batch_obs.len(), 1]),
        acts_tensor,
        weights_tensor,
    );

    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &logits_net);
    let optimized_logits_net = optimizer.step(1e-2, logits_net, gradient_params);

    (
        optimized_logits_net,
        batch_rets.iter().sum::<f32>() / batch_rets.len() as f32,
    )
}
