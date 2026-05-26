use std::{any::Any, vec};

use burn::{Tensor, prelude::Backend, tensor::Int};
use itertools::izip;

use crate::{
    simple::actor_critic::{MLPActorCritic, MLPCategoricalActor, MLPCritic},
    util::{discount_cumsum, mean_and_std, vec2d_to_tensor},
};

pub struct VPGBuffer {
    ptr: usize,
    max_size: usize,
    path_start_idx: usize,

    obs_buf: Vec<Vec<f32>>,
    act_buf: Vec<i32>,
    rew_buf: Vec<f32>,
    val_buf: Vec<f32>,
    logp_buf: Vec<f32>,

    adv_buf: Vec<f32>,
    ret_buf: Vec<f32>,

    gamma: f32,
    lam: f32,
}

impl VPGBuffer {
    pub fn new(obs_dim: usize, n_act: usize, size: usize, gamma: f32, lam: f32) -> Self {
        Self {
            ptr: 0,
            max_size: size,
            path_start_idx: 0,
            obs_buf: vec![vec![]; size],
            act_buf: vec![0; size],
            rew_buf: vec![0.; size],
            val_buf: vec![0.; size],
            logp_buf: vec![0.; size],
            adv_buf: vec![0.; size],
            ret_buf: vec![0.; size],
            gamma,
            lam,
        }
    }

    pub fn store(&mut self, obs: Vec<f32>, act: i32, rew: f32, val: f32, logp: f32) {
        assert!(self.ptr < self.max_size);
        self.obs_buf[self.ptr] = obs;
        self.act_buf[self.ptr] = act;
        self.rew_buf[self.ptr] = rew;
        self.val_buf[self.ptr] = val;
        self.logp_buf[self.ptr] = logp;
        self.ptr += 1;
    }

    pub fn finish_path(&mut self, last_val: f32) {
        let path_slice = self.path_start_idx..self.ptr;
        let mut rews = self.rew_buf[path_slice.clone()].to_vec();
        rews.push(last_val);
        let mut vals = self.val_buf[path_slice.clone()].to_vec();
        vals.push(last_val);

        // GAE-Lambda advantage calculation
        let mut deltas = vec![];
        for (a, b, c) in izip!(
            rews[0..rews.len() - 1].iter(),
            vals[1..vals.len()].iter(),
            vals[0..vals.len() - 1].iter()
        ) {
            deltas.push(a + b * self.gamma - c);
        }
        self.adv_buf[path_slice.clone()]
            .copy_from_slice(&discount_cumsum(&deltas, self.gamma * self.lam));

        // Computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice.clone()]
            .copy_from_slice(&discount_cumsum(&rews, self.gamma)[0..rews.len() - 1]);

        self.path_start_idx = self.ptr;
    }

    pub fn get(&mut self) {
        assert_eq!(self.ptr, self.max_size);
        self.ptr = 0;
        self.path_start_idx = 0;

        // Advantage normalization trick
        let (adv_mean, adv_std) = mean_and_std(&self.adv_buf);
        for x in self.adv_buf.iter_mut() {
            *x = (*x - adv_mean) / adv_std;
        }
    }
}

pub fn compute_loss_pi<B: Backend>(buf: &VPGBuffer, pi: &MLPCategoricalActor<B>) -> Tensor<B, 1> {
    let obs = vec2d_to_tensor::<B>(buf.obs_buf.clone(), &Default::default());
    let act = Tensor::<B, 1, Int>::from_data(buf.act_buf.clone().as_slice(), &Default::default());
    let adv = Tensor::<B, 1>::from_data(buf.adv_buf.clone().as_slice(), &Default::default());

    // Policy loss
    let policy = pi.distribution(obs);
    let logp = pi.log_prob_from_distribution(&policy, act);
    let loss_pi = -(logp * adv).mean();

    loss_pi
}

pub fn compute_loss_v<B: Backend>(buf: &VPGBuffer, v: &MLPCritic<B>) -> Tensor<B, 1> {
    let obs = vec2d_to_tensor::<B>(buf.obs_buf.clone(), &Default::default());
    let ret = Tensor::<B, 1>::from_data(buf.ret_buf.clone().as_slice(), &Default::default());
    (v.forward(obs) - ret).square().mean()
}
