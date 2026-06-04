import { getSimulator, initSimulator } from "./sim";

initSimulator();

document.addEventListener("keydown", (event) => {
  if (event.code == "Space") {
    console.log("space pressed");
    let simulator = getSimulator();
    simulator.hybrid.reset();
    simulator.hybrid.set_joint_q(0, Math.PI);
  }
});

setInterval(() => {
  let simulator = getSimulator();
  if (simulator.hybrid.time() > 7) {
    let q = simulator.hybrid.joint_q()[0];
    if (q > Math.PI / 2 || q < -Math.PI / 2) {
      simulator.hybrid.reset();
      simulator.hybrid.set_joint_q(0, Math.PI);
    }
  }
}, 100);
