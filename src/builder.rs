#[cfg(any(target_arch = "wasm32", rust_analyzer))]
use {
    crate::ppo_controller::PPOPendulumController,
    gorilla_physics::{
        WORLD_FRAME,
        hybrid::{Hybrid, Rigid, articulated::Articulated, control::NullArticulatedController},
        interface::hybrid::InterfaceHybrid,
        joint::{Joint, JointPosition},
        spatial::transform::Transform3D,
    },
    nalgebra::{Vector3, vector},
    std::f64::consts::PI,
    wasm_bindgen::prelude::wasm_bindgen,
};

#[cfg(any(target_arch = "wasm32", rust_analyzer))]
#[allow(non_snake_case)]
#[wasm_bindgen]
pub async fn createPendulumSwingup() -> InterfaceHybrid {
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

    let mut articulated = Articulated::new(vec![pendulum], vec![pendulum_joint]);
    articulated.set_joint_q(0, JointPosition::Float(PI));

    state.add_articulated(articulated);

    let controller = PPOPendulumController::new();
    state.set_controller(0, controller);

    InterfaceHybrid::new(state)
}
