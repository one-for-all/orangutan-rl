use std::f32::consts::PI;

use burn::{
    backend::{Autodiff, NdArray},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{ElementConversion, backend::Backend},
};
use gorilla_physics::{
    hybrid::{articulated::Articulated, control::ArticulatedController},
    types::Float,
    util::console_log,
};
use nalgebra::{DVector, dvector};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    buffer::{VPGBuffer, compute_loss_pi_ppo, compute_loss_v},
    env::pendulum::PendulumEnv,
    simple::{
        actor_critic::MLPCritic,
        actor_critic_continuous::{MLPActorCriticContinuous, MLPGaussianActor},
    },
    util::vec2d_to_tensor,
};

type MyBackend = Autodiff<NdArray>;

const EPOCHS: usize = 500;

const MAX_EP_LEN: usize = 60;
const STEPS_PER_EPOCH: usize = MAX_EP_LEN * 2;
const GAMMA: f32 = 0.99; // Discount factor
const LAM: f32 = 0.97; // Lambda for GAE-Lambda

const PI_LR: f64 = 3e-4; // Policy learning rate
const VF_LR: f64 = 1e-3; // Value function learning rate

const TRAIN_PI_ITERS: usize = 80;
const TRAIN_V_ITERS: usize = 80;

const CLIP_RATIO: f32 = 0.2;
const TARGET_KL: f32 = 0.01;

const SWINGUP: bool = true;

const CONTROL_HZ_DIVIDER: usize = 6;

const DETERMINISTIC: bool = false;

pub struct PPOPendulumController {
    ac: MLPActorCriticContinuous<MyBackend>,
    env: PendulumEnv,
    buf: VPGBuffer,

    pi_optimizer: OptimizerAdaptor<Adam, MLPGaussianActor<MyBackend>, MyBackend>,
    vf_optimizer: OptimizerAdaptor<Adam, MLPCritic<MyBackend>, MyBackend>,

    num_epochs: usize,

    rng: StdRng,
    last_start_top: bool,

    k: usize,
    last_act: f64,

    o: Vec<f32>,
    ep_ret: f32,
    ep_len: usize,
    init_q: f32,
    t: usize,
}

impl PPOPendulumController {
    pub fn new() -> Self {
        let HIDDEN_SIZES: Vec<usize> = vec![32, 32];

        let rng;
        if DETERMINISTIC {
            MyBackend::seed(&Default::default(), 0);
            rng = StdRng::seed_from_u64(1);
        } else {
            rng = StdRng::from_rng(&mut rand::rng());
        }

        let dt = 1. / 60. * CONTROL_HZ_DIVIDER as f32;
        let mut env = PendulumEnv::new(dt);

        let obs_dim = env.obs_dim();
        let act_dim = env.act_dim();
        assert_eq!(act_dim, 1);

        // actor-critic module
        let ac = MLPActorCriticContinuous::<MyBackend>::new(obs_dim, act_dim, &HIDDEN_SIZES);

        // Set up experience buffer
        let buf = VPGBuffer::new(obs_dim, act_dim, STEPS_PER_EPOCH, GAMMA, LAM);

        let pi_optimizer = AdamConfig::new().init();
        let vf_optimizer = AdamConfig::new().init();

        let last_start_top = !SWINGUP;

        // Prepare for interaction with environment
        let init_q = if SWINGUP { PI } else { 0. };
        let o = env.reset(init_q);
        let ep_ret = 0.;
        let ep_len = 0;

        Self {
            ac,
            env,
            buf,
            pi_optimizer,
            vf_optimizer,
            num_epochs: 0,
            rng,
            last_start_top,
            k: 0,
            last_act: 0.,

            o,
            ep_ret,
            ep_len,
            init_q,
            t: 0,
        }
    }
}

impl PPOPendulumController {
    pub fn run_step(&mut self) {
        let (a, v, logp) = self
            .ac
            .step(vec2d_to_tensor(vec![self.o.clone()], &Default::default()));

        let (next_o, r) = self.env.step(a);
        self.ep_ret += r;
        self.ep_len += 1;

        // save to buffer
        self.buf.store(self.o.clone(), a, r, v, logp);

        // Update obs
        self.o = next_o;

        let timeout = self.ep_len == MAX_EP_LEN;
        let epoch_ended = self.t == STEPS_PER_EPOCH - 1;
        if timeout || epoch_ended {
            if epoch_ended && !timeout {
                println!(
                    "Warning: trajectory cut off by epoch at {} steps.",
                    self.ep_len
                );
            }
            // if trajectory didn't reach terminal state, bootstrap value target
            let last_v = self
                .ac
                .v
                .forward(vec2d_to_tensor(vec![self.o.clone()], &Default::default()))
                .into_scalar()
                .elem();
            self.buf.finish_path(last_v);

            if timeout {
                if (self.last_start_top && !SWINGUP) || (!self.last_start_top && SWINGUP) {
                    console_log(&format!("episode return: {:.5}", self.ep_ret));
                }
            }

            self.o = if self.rng.random_bool(0.5) {
                self.last_start_top = false;
                self.env.reset(self.init_q) // reset pendulum to bottom
            } else {
                self.last_start_top = true;
                self.env.reset(0.) // reset pendulum to top
            };

            self.ep_ret = 0.;
            self.ep_len = 0;
        }
    }
}

impl ArticulatedController for PPOPendulumController {
    fn control(&mut self, articulated: &Articulated, input: &Vec<Float>) -> DVector<Float> {
        for _ in 0..MAX_EP_LEN / 2 {
            self.run_step();
            self.t += 1;
        }

        if self.t >= STEPS_PER_EPOCH {
            self.t = 0;
            // Perform VPG update
            self.buf.get();

            // Train policy with a single step of gradient descent
            for i in 0..TRAIN_PI_ITERS {
                let (loss_pi, kl) = compute_loss_pi_ppo(&self.buf, &self.ac.pi, CLIP_RATIO);
                if kl > 1.5 * TARGET_KL {
                    println!("Early stopping at step {} due to reaching max kl", i);
                    break;
                }
                let gradients = loss_pi.backward();
                let gradient_params = GradientsParams::from_grads(gradients, &self.ac.pi);

                // TODO: avoid the clone
                self.ac.pi = self
                    .pi_optimizer
                    .step(PI_LR, self.ac.pi.clone(), gradient_params);
            }

            for _ in 0..TRAIN_V_ITERS {
                let loss_v = compute_loss_v(&self.buf, &self.ac.v);
                let gradients = loss_v.backward();
                let gradient_params = GradientsParams::from_grads(gradients, &self.ac.v);
                self.ac.v = self
                    .vf_optimizer
                    .step(VF_LR, self.ac.v.clone(), gradient_params);
            }
        }

        self.num_epochs += 1;
        console_log(&format!("epoch: {}", self.num_epochs));

        let act;
        if self.k == 0 {
            // Perform action
            let q = articulated.q()[0] as f32;
            let v = articulated.v()[0] as f32;
            let obs = vec![q, v];
            let obs_tensor = vec2d_to_tensor(vec![obs.clone()], &Default::default());

            // Critical: clamp the same way as in pendulum env
            act = self.ac.act(obs_tensor).clamp(-2., 2.) as Float;
            self.last_act = act;
        } else {
            act = self.last_act;
        }
        self.k = (self.k + 1) % CONTROL_HZ_DIVIDER;

        dvector![act]
    }
}
