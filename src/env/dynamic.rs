/// A point mass on a frictionless surface.
/// Control input is the acceleration, and goal is to reach and stay at position 1.
/// mathematically: d^2q/dt = u; |u| <= 1
/// For simplicity, action is assumed to be either +-1 or 0
/// Ref: https://underactuated.csail.mit.edu/dp.html#example1
pub struct DoubleIntegratorEnv {
    x: f32,
    v: f32,
    pub t: f32, // current time

    pub dt: f32,
}

impl DoubleIntegratorEnv {
    pub fn new() -> Self {
        Self {
            x: 0.,
            v: 0.,
            dt: 0.1,
            t: 0.,
        }
    }

    /// Observation is (x, v, t)
    pub fn obs_dim(&self) -> usize {
        2
    }

    /// 3 discrete actions:
    /// 0 -> u = 0;
    /// 1 -> u = 1;
    /// 2 -> u = -1;
    pub fn n_acts(&self) -> usize {
        3
    }

    pub fn reset(&mut self, x: f32) -> Vec<f32> {
        self.x = x;
        self.v = 0.;

        self.t = 0.;

        let obs = vec![self.x, self.v];
        assert_eq!(obs.len(), self.obs_dim());

        obs
    }

    pub fn step(&mut self, action: i32) -> (Vec<f32>, f32, bool) {
        let u = match action {
            0 => 0.,
            1 => 1.,
            2 => -1.,
            _ => panic!("unknown action: {action}"),
        };

        // Semi-implicit integration
        self.v += u * self.dt;
        self.x += self.v * self.dt;
        self.t += self.dt;

        // Compute the reward
        let x_goal = 1.;
        let cost = (self.x - x_goal).powi(2);
        let reward = -cost * self.dt;

        let obs = vec![self.x, self.v];
        assert_eq!(obs.len(), self.obs_dim());
        let done = false;

        // Add the infinite discounted sum of future rewards assuming constant velocity
        // Note: not good, because this means any non-zero velocity will cause large cost. So the policy learns to prefer zero action at the beginning.
        // if done {
        //     reward = -infinite_sum(self.v, self.x - x_goal, 0.99, self.dt);
        // }

        return (obs, reward, done);
    }
}

/// Sum over t from 0 to infinity of (alpha * t + beta)^2 * gamma^t * dt
fn infinite_sum(alpha: f32, beta: f32, gamma: f32, dt: f32) -> f32 {
    let numerator = alpha * alpha * gamma * (1. + gamma)
        + 2. * alpha * beta * gamma * (1. - gamma)
        + beta * beta * (1. - gamma).powi(2);
    let denominator = (1. - gamma).powi(3);
    numerator / denominator * dt
}
