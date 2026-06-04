use gorilla_physics::{
    PI, WORLD_FRAME,
    hybrid::{Hybrid, Rigid, articulated::Articulated, control::ArticulatedController},
    joint::{Joint, JointPosition, JointVelocity},
    na::vector,
    spatial::transform::Transform3D,
    types::Float,
};
use nalgebra::{DVector, Vector3, dvector};

pub struct PendulumEnv {
    pub state: Hybrid,

    pub t: f32,
    pub dt: f32,
}

impl PendulumEnv {
    pub fn new(dt: f32) -> Self {
        let mut state = Hybrid::empty();

        let m = 1.0;
        let w = 0.1;
        let d = 0.1;
        let h = 1.0;
        let com = vector![0., 0., h / 2.];
        let pendulum_frame = "pendulum";
        let pendulum = Rigid::new_cuboid_at(&com, m, w, d, h, pendulum_frame);

        let pendulum_joint = Joint::new_revolute(
            Transform3D::identity(pendulum_frame, WORLD_FRAME),
            Vector3::y_axis(),
        );

        let articulated = Articulated::new(vec![pendulum], vec![pendulum_joint]);

        state.add_articulated(articulated);

        let controller = PendulumController {};
        state.set_controller(0, controller);

        let mut pendulum = Self {
            state,
            t: 0.,
            dt: dt,
        };

        pendulum.reset(0.);
        pendulum
    }

    /// Observation is joint angle and velocity (q, v)
    pub fn obs_dim(&self) -> usize {
        2
    }

    /// Action is joint torque
    pub fn act_dim(&self) -> usize {
        1
    }

    pub fn reset(&mut self, q: f32) -> Vec<f32> {
        self.state.articulated[0].reset();
        self.state.articulated[0].set_joint_q(0, JointPosition::Float(q as f64));
        // self.state.articulated[0].set_joint_q(0, JointPosition::Float(0.1));
        self.t = 0.;

        let q = self.state.articulated[0].q()[0] as f32;

        // let obs = vec![q.sin(), q.cos(), self.state.articulated[0].v()[0] as f32];
        let obs = vec![q, self.state.articulated[0].v()[0] as f32];
        assert_eq!(obs.len(), self.obs_dim());

        obs
    }

    /// Returns (obs, reward)
    pub fn step(&mut self, u: f32) -> (Vec<f32>, f32) {
        let u = u.clamp(-2., 2.);

        self.state.step(self.dt as Float, &vec![u as Float]);
        let v = self.state.articulated[0].v()[0] as f32;
        // self.state.articulated[0].set_joint_v(0, JointVelocity::Float(v.clamp(-16., 16.) as f64));

        self.t += self.dt;

        // Compute reward
        let q_goal = 0.; // std::f32::consts::PI;
        let q = self.state.articulated[0].q()[0] as f32;

        let v = self.state.articulated[0].v()[0] as f32;
        let cost = angle_difference(q, q_goal).powi(2) + 0.1 * v * v;
        // let cost = (q-q_goal).powi(2);
        let reward = -cost * self.dt;

        // let obs = vec![q.sin(), q.cos(), v];
        let obs = vec![q, v];
        assert_eq!(obs.len(), self.obs_dim());

        return (obs, reward);
    }

    pub fn get_state(&self) -> Vec<f32> {
        let q = self.state.articulated[0].q()[0] as f32;
        let v = self.state.articulated[0].v()[0] as f32;
        vec![q, v]
    }
}

pub fn pendulum_obs(q: f32, v: f32) -> [f32; 2] {
    // [q.sin(), q.cos(), v]
    [q, v]
}

struct PendulumController {}

impl ArticulatedController for PendulumController {
    fn control(
        &mut self,
        articulated: &Articulated,
        input: &Vec<gorilla_physics::types::Float>,
    ) -> nalgebra::DVector<gorilla_physics::types::Float> {
        DVector::from_iterator(input.len(), input.iter().cloned())
    }
}

fn angle_difference(a: f32, b: f32) -> f32 {
    // Formula: atan2(sin(a - b), cos(a - b))
    (a - b).sin().atan2((a - b).cos())
}
