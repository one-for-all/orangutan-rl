import { createPendulumSwingup } from "orangutan-rl";
import { Simulator } from "gorilla-physics-ui";

let _simulator: Simulator | null = null;

export function initSimulator() {
  let interfaceSimulator = null;
  let showGrid = true;
  let simulator = new Simulator(interfaceSimulator, showGrid);

  createPendulumSwingup().then((state) => {
    simulator.addHybrid(state);
    simulator.updateHybrid();

    let cameraPosition = {
      eye: { x: 0.0, y: -2, z: 1 },
      target: { x: 0.0, y: 0, z: 0 },
    };
    simulator.graphics.lookAt(cameraPosition);

    simulator.run(1, 0); // 10

    setSimulator(simulator);
  });
}

function setSimulator(sim: Simulator) {
  _simulator = sim;
}

export function getSimulator(): Simulator | null {
  return _simulator;
}
