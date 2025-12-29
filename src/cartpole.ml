type state = { x : float; x_dot : float; theta : float; theta_dot : float }

let gravity = 9.81
let masscart = 1.
let masspole = 0.1
let total_mass = masscart +. masspole
let length = 0.5
let polemass_length = masspole *. length
let force_mag = 10.
let tau = 0.02

let initial_state () =
  {
    x = Random.float 0.1 -. 0.05;
    x_dot = Random.float 0.1 -. 0.05;
    theta = Random.float 0.1 -. 0.05;
    theta_dot = Random.float 0.1 -. 0.05;
  }

let step state action =
  let force = if action > 0.5 then force_mag else -.force_mag in

  let costheta = cos state.theta in
  let sintheta = sin state.theta in

  let temp =
    (force +. (polemass_length *. (state.theta ** 2.) *. sintheta))
    /. total_mass
  in

  let thetaacc =
    ((gravity *. sintheta) -. (costheta *. temp))
    /. (length *. ((4. /. 3.) -. (masspole *. (costheta ** 2.) /. total_mass)))
  in

  let xacc = temp -. (polemass_length *. thetaacc *. costheta /. total_mass) in

  let x = state.x +. (tau *. state.x_dot) in
  let x_dot = state.x_dot +. (tau *. xacc) in
  let theta = state.theta +. (tau *. state.theta_dot) in
  let theta_dot = state.theta_dot +. (tau *. thetaacc) in

  { x; x_dot; theta; theta_dot }

let is_failed state =
  state.x < -2.4 || state.x > 2.4 || state.theta < -0.2095
  || state.theta > 0.2095

let get_inputs state = [ state.theta; state.theta_dot; state.x; state.x_dot ]
