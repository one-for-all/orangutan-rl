/// A point mass on a frictionless surface.
/// Control input is the acceleration, and goal is to reach and stay at position 1.
/// mathematically: d^2q/dt = u; |u| <= 1
/// Ref: https://underactuated.csail.mit.edu/dp.html#example1
pub struct ContinuousDoubleIntegrator {
    x: f32,
    v: f32,

    pub t: f32, // current time
    pub dt: f32,
}

impl ContinuousDoubleIntegrator {
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

    /// Action is u, a single number
    pub fn act_dim(&self) -> usize {
        1
    }

    pub fn reset(&mut self, x: f32) -> Vec<f32> {
        self.x = x;
        self.v = 0.;

        self.t = 0.;

        let obs = vec![self.x, self.v];
        assert_eq!(obs.len(), self.obs_dim());

        obs
    }

    /// Returns (obs, reward)
    pub fn step(&mut self, u: f32) -> (Vec<f32>, f32) {
        let u = u.clamp(-1., 1.);

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

        return (obs, reward);
    }
}
