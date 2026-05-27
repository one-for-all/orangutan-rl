use burn::{
    Tensor,
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::ElementConversion,
};
use orangutan_rl::{
    buffer::{VPGBuffer, compute_loss_pi, compute_loss_v},
    env::dynamic::DoubleIntegratorEnv,
    plot::plot,
    simple::actor_critic::MLPActorCritic,
    util::vec2d_to_tensor,
};

const EPOCHS: usize = 2000;

// type MyBackend = Autodiff<Wgpu>;
type MyBackend = Autodiff<NdArray>;

const STEPS_PER_EPOCH: usize = 50; // 4000
const MAX_EP_LEN: usize = 25; // 1000
const GAMMA: f32 = 0.99; // Discount factor
const LAM: f32 = 0.97; // Lambda for GAE-Lambda

const PI_LR: f64 = 3e-3; // Policy learning rate
const VF_LR: f64 = 1e-3; // Value function learning rate

const TRAIN_V_ITERS: usize = 80;

fn main() {
    MyBackend::seed(&Default::default(), 0);

    // let mut env = OneDimGridEnv::new();
    let mut env = DoubleIntegratorEnv::new();

    let obs_dim = env.obs_dim();
    let n_acts = env.n_acts();

    // actor-critic module
    let mut ac = MLPActorCritic::<MyBackend>::new(obs_dim, n_acts, &vec![32, 32]);

    // Set up experience buffer
    let mut buf = VPGBuffer::new(obs_dim, n_acts, STEPS_PER_EPOCH, GAMMA, LAM);

    let mut pi_optimizer = AdamConfig::new().init();
    let mut vf_optimizer = AdamConfig::new().init();

    let mut data = vec![];

    // Prepare for interaction with environment
    let mut o = env.reset(true);
    let mut ep_ret = 0.;
    let mut ep_len = 0;

    for epoch in 0..EPOCHS {
        println!("====== epoch: {epoch}");
        for t in 0..STEPS_PER_EPOCH {
            let (a, v, logp) = ac.step(vec2d_to_tensor(vec![o.clone()], &Default::default()));

            let (next_o, r, d) = env.step(a);
            ep_ret += r;
            ep_len += 1;

            // save to buffer
            buf.store(o.clone(), a, r, v, logp);

            // Update obs
            o = next_o;

            let timeout = ep_len == MAX_EP_LEN;
            let terminal = d || timeout;
            let epoch_ended = t == STEPS_PER_EPOCH - 1;

            if terminal || epoch_ended {
                if epoch_ended && !terminal {
                    println!("Warning: trajectory cut off by epoch at {} steps.", ep_len);
                }
                // if trajectory didn't reach terminal state, bootstrap value target
                let last_v;
                if timeout || epoch_ended {
                    last_v =
                        ac.v.forward(vec2d_to_tensor(vec![o.clone()], &Default::default()))
                            .into_scalar()
                            .elem();
                } else {
                    last_v = 0.;
                }
                buf.finish_path(last_v);
                if terminal {
                    // println!("episode return: {}", ep_ret);
                    data.push(ep_ret);
                }
                o = env.reset(false);
                ep_ret = 0.;
                ep_len = 0;
            }
        }

        // Perform VPG update
        buf.get();

        // Train policy with a single step of gradient descent
        let loss_pi = compute_loss_pi(&buf, &ac.pi);
        let gradients = loss_pi.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &ac.pi);
        ac.pi = pi_optimizer.step(PI_LR, ac.pi, gradient_params);

        for _ in 0..TRAIN_V_ITERS {
            let loss_v = compute_loss_v(&buf, &ac.v);
            let gradients = loss_v.backward();
            let gradient_params = GradientsParams::from_grads(gradients, &ac.v);
            ac.v = vf_optimizer.step(VF_LR, ac.v, gradient_params);
        }

        // Print action probabilities
        let policy = ac
            .pi
            .distribution(Tensor::from_data([[0., 0.], [1., 0.]], &Default::default()));
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
        let act = ac.act(obs_tensor);
        let (next_obs, rew, _done) = env.step(act);

        ret += rew;
        let obs_formatted: Vec<String> = obs.iter().map(|f| format!("{:.2}", f)).collect();
        println!("obs: {:?}, action: {}, reward: {}", obs_formatted, act, rew);

        obs = next_obs;
        data2.push(obs[0]);
    }
    println!("total return: {ret}");

    plot(&data, 1.0, "VPG");
    plot(&data2, env.dt, "double integrator trajectory");
}
