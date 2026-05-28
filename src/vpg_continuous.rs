use burn::{
    Tensor,
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::ElementConversion,
};
use orangutan_rl::{
    buffer::{VPGBuffer, compute_loss_pi_continuous, compute_loss_v},
    env::double_integrator::continuous::ContinuousDoubleIntegrator,
    plot::plot,
    simple::actor_critic_continuous::MLPActorCriticContinuous,
    util::vec2d_to_tensor,
};

type MyBackend = Autodiff<NdArray>;

const EPOCHS: usize = 1000;

const STEPS_PER_EPOCH: usize = 50; // 4000
const MAX_EP_LEN: usize = 25; // 1000
const GAMMA: f32 = 0.99; // Discount factor
const LAM: f32 = 0.97; // Lambda for GAE-Lambda

const PI_LR: f64 = 3e-3; // Policy learning rate
const VF_LR: f64 = 1e-3; // Value function learning rate

const TRAIN_V_ITERS: usize = 80;

fn main() {
    MyBackend::seed(&Default::default(), 0);

    let mut env = ContinuousDoubleIntegrator::new();

    let obs_dim = env.obs_dim();
    let act_dim = env.act_dim();

    // actor-critic module
    let mut ac = MLPActorCriticContinuous::<MyBackend>::new(obs_dim, act_dim, &vec![32, 32]);

    // Set up experience buffer
    let mut buf = VPGBuffer::new(obs_dim, act_dim, STEPS_PER_EPOCH, GAMMA, LAM);

    let mut pi_optimizer = AdamConfig::new().init();
    let mut vf_optimizer = AdamConfig::new().init();

    let mut data = vec![];

    // Prepare for interaction with environment
    let mut o = env.reset(0.);
    let mut ep_ret = 0.;
    let mut ep_len = 0;

    for epoch in 0..EPOCHS {
        println!("====== epoch: {epoch}");
        for t in 0..STEPS_PER_EPOCH {
            let (a, v, logp) = ac.step(vec2d_to_tensor(vec![o.clone()], &Default::default()));

            let (next_o, r) = env.step(a);
            ep_ret += r;
            ep_len += 1;

            // save to buffer
            buf.store(o.clone(), a, r, v, logp);

            // Update obs
            o = next_o;

            let timeout = ep_len == MAX_EP_LEN;
            let epoch_ended = t == STEPS_PER_EPOCH - 1;
            if timeout || epoch_ended {
                if epoch_ended && !timeout {
                    println!("Warning: trajectory cut off by epoch at {} steps.", ep_len);
                }
                // if trajectory didn't reach terminal state, bootstrap value target
                let last_v =
                    ac.v.forward(vec2d_to_tensor(vec![o.clone()], &Default::default()))
                        .into_scalar()
                        .elem();
                buf.finish_path(last_v);

                if timeout {
                    data.push(ep_ret);
                }

                o = env.reset(0.); // reset x to 0
                ep_ret = 0.;
                ep_len = 0;
            }
        }

        // Perform VPG update
        buf.get();

        // Train policy with a single step of gradient descent
        let loss_pi = compute_loss_pi_continuous(&buf, &ac.pi);
        let gradients = loss_pi.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &ac.pi);
        ac.pi = pi_optimizer.step(PI_LR, ac.pi, gradient_params);

        for _ in 0..TRAIN_V_ITERS {
            let loss_v = compute_loss_v(&buf, &ac.v);
            let gradients = loss_v.backward();
            let gradient_params = GradientsParams::from_grads(gradients, &ac.v);
            ac.v = vf_optimizer.step(VF_LR, ac.v, gradient_params);
        }

        // Print mean action
        let policy = ac
            .pi
            .distribution(Tensor::from_data([[0., 0.], [1., 0.]], &Default::default()));

        let action_mean_formatted: Vec<String> = policy
            .mu
            .into_data()
            .iter()
            .map(|f: f32| format!("{:.2}", f))
            .collect();

        println!(
            "action mean: {:?}, std: {:.2}",
            action_mean_formatted, policy.std
        );
    }

    // Roll out a policy
    println!("======= Policy Rollout");
    let mut data2 = vec![];
    let mut obs = env.reset(0.);
    data2.push(obs[0]);
    let mut ret = 0.;
    while env.t < 10. {
        let obs_tensor = vec2d_to_tensor(vec![obs.clone()], &Default::default());
        let act = ac.act(obs_tensor);
        let (next_obs, rew) = env.step(act);

        ret += rew;
        let obs_formatted: Vec<String> = obs.iter().map(|f| format!("{:.2}", f)).collect();
        println!("obs: {:?}, action: {}, reward: {}", obs_formatted, act, rew);

        obs = next_obs;
        data2.push(obs[0]);
    }
    println!("total return: {ret}");

    plot(&data, 1.0, "VPG continuous");
    plot(&data2, env.dt, "double integrator trajectory");
}
