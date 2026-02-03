open Neatml
open Neural.Layers
open Neural.Models
open Neural.Training
open Neural.Core

let create_dataset n =
  Array.init n (fun _ ->
      let seed = Random.float (4. *. Float.pi) in
      (seed, sin seed))

let () =
  Random.self_init ();

  (* 1. Enable GPU at the very beginning *)
  Utils.enable_gpu ();
  Printf.printf
    "\027[1;32m[SYSTEM]\027[0m GPU Acceleration Enabled: \027[1;33m%b\027[0m\n"
    !Utils.use_gpu;

  (* 2. Create a larger dataset to saturate the GPU *)
  let n_samples = 10000 in
  let dataset = create_dataset n_samples in

  let x_train =
    let t = Tensor.CPU (Array.map (fun row -> [| fst row |]) dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in
  let y_train =
    let t = Tensor.CPU (Array.map (fun row -> [| snd row |]) dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in

  let test_dataset = create_dataset 200 in
  let x_test =
    let t = Tensor.CPU (Array.map (fun row -> [| fst row |]) test_dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in
  let y_test =
    let t = Tensor.CPU (Array.map (fun row -> [| snd row |]) test_dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in

  (* 3. Use a larger model (Hidden layer: 128 neurons) *)
  let hidden_dim = 256 in
  let batch_size = 256 in

  let hidden_layer = Linear.create 1 hidden_dim batch_size Activations.ReLU in
  let output_layer = Linear.create hidden_dim 1 batch_size Activations.Tanh in

  let model =
    {
      Sequential.layers =
        [ Layer.Linear hidden_layer; Layer.Linear output_layer ];
    }
  in

  let lr = 0.001 in
  let optimizers = Optimizer.create lr Optimizer.Adam model in

  Printf.printf
    "\027[1;34m[INFO]\027[0m Training SINE network (Samples: %d, Hidden: %d, \
     Batch: %d)...\n"
    n_samples hidden_dim batch_size;
  flush stdout;

  (* Optimizer.fit now reports timing per epoch *)
  Optimizer.fit model x_train y_train x_test y_test batch_size 500 optimizers
    Errors.MSE;

  Printf.printf "\027[1;32m[SUCCESS]\027[0m Training complete. \n";

  (* 4. Visualization *)
  let open Graphics in
  open_graph " 800x600";
  set_window_title "SINE Network: Reality vs Prediction";

  let width = 800 in
  let height = 600 in
  let x_scale = float width /. (2. *. Float.pi) in
  let y_scale = float height /. 2.5 in
  let y_offset = height / 2 in

  (* Draw axes *)
  set_color (rgb 150 150 150);
  moveto 0 y_offset;
  lineto width y_offset;
  moveto 0 0;
  lineto 0 height;

  (* Plot Real Sine (Blue) *)
  set_color blue;
  set_line_width 2;
  for x_px = 0 to width - 1 do
    let x = float x_px /. x_scale in
    let y = sin x in
    let y_px = int_of_float (y *. y_scale) + y_offset in
    if x_px = 0 then moveto x_px y_px else lineto x_px y_px
  done;

  (* Plot Predicted Sine (Red) *)
  set_color red;
  set_line_width 2;
  let sample_points = 200 in
  for i = 0 to sample_points - 1 do
    let x = float i /. float sample_points *. (2. *. Float.pi) in
    let x_tensor = Tensor.CPU [| [| x |] |] in
    let x_gpu = if !Utils.use_gpu then Utils.to_gpu x_tensor else x_tensor in
    let pred_tensor = Sequential.forward_seq model x_gpu in
    let pred = (Utils.to_cpu pred_tensor).(0).(0) in

    let x_px = int_of_float (x *. x_scale) in
    let y_px = int_of_float (pred *. y_scale) + y_offset in

    if i = 0 then moveto x_px y_px else lineto x_px y_px;

    (* Cleanup GPU prediction if needed *)
    (match pred_tensor with
    | Tensor.GPU g -> Gpu.release g
    | _ -> ());
    match x_gpu with Tensor.GPU g -> Gpu.release g | _ -> ()
  done;

  (* Legend *)
  set_color blue;
  moveto (width - 150) (height - 30);
  draw_string "Reality (sin x)";
  set_color red;
  moveto (width - 150) (height - 50);
  draw_string "Prediction";

  Printf.printf "Press any key to close the window...\n";
  ignore (wait_next_event [ Key_pressed ]);
  close_graph ()
