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
