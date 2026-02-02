open Tensor

let use_gpu = ref false
let enable_gpu () = use_gpu := true
let disable_gpu () = use_gpu := false

let rows (t : Tensor.t) = 
  match t with
  | CPU m -> Array.length m
  | GPU g -> (Gpu.to_cpu g |> Array.length) (* Helper access, slow but correct *)

let cols (t : Tensor.t) = 
  match t with
  | CPU m -> if Array.length m = 0 then 0 else Array.length m.(0)
  | GPU g -> (Gpu.to_cpu g |> fun m -> if Array.length m = 0 then 0 else Array.length m.(0))

(* Constructor always creates CPU tensor initially, can be moved later *)
let zeros r c = CPU (Array.init r (fun _ -> Array.make c 0.0))

let copy_mat (t : Tensor.t) = 
  match t with
  | CPU m -> CPU (Array.map Array.copy m)
  | GPU g -> 
      (* Deep copy on GPU or download-upload? 
         For now, assume download-upload or just alias if immutable logic, 
         but our library relies on mutability (e.g. optimizer).
         Let's do CPU copy for safety. *)
      let m = Gpu.to_cpu g in
      if !use_gpu then GPU (Gpu.of_cpu m) else CPU m

let copy_mat_inplace (src : Tensor.t) (dst : Tensor.t) =
  match src, dst with
  | CPU s, CPU d -> 
      Array.iteri (fun i a -> Array.iteri (fun j v -> d.(i).(j) <- v) a) s
  | GPU s, GPU d -> Gpu.copy_inplace s d
  | CPU s, GPU d -> 
      (* Upload to existing buffer *)
      let tmp = Gpu.of_cpu s in
      Gpu.copy_inplace tmp d
  | GPU s, CPU d -> 
      (* Download to existing array *)
      let s_cpu = Gpu.to_cpu s in
      Array.iteri (fun i a -> Array.iteri (fun j v -> d.(i).(j) <- v) a) s_cpu

let map_mat f (t : Tensor.t) =
  match t with
  | CPU m -> CPU (Array.mapi (fun i row -> Array.mapi (fun j x -> f x i j) row) m)
  | GPU g -> 
      (* Fallback to CPU for arbitrary function f *)
      let m = Gpu.to_cpu g in
      let res = Array.mapi (fun i row -> Array.mapi (fun j x -> f x i j) row) m in
      if !use_gpu then GPU (Gpu.of_cpu res) else CPU res

let map2_mat f t1 t2 =
  match t1, t2 with
  | CPU m1, CPU m2 ->
      if Array.length m1 <> Array.length m2 || Array.length m1.(0) <> Array.length m2.(0)
      then failwith "Error: expected same sized matrices";
      CPU (Array.mapi (fun i row -> Array.mapi (fun j x -> f x m2.(i).(j)) row) m1)
  | GPU g1, GPU g2 ->
      (* Fallback to CPU *)
      let m1 = Gpu.to_cpu g1 in
      let m2 = Gpu.to_cpu g2 in
       if Array.length m1 <> Array.length m2 || Array.length m1.(0) <> Array.length m2.(0)
      then failwith "Error: expected same sized matrices";
      let res = Array.mapi (fun i row -> Array.mapi (fun j x -> f x m2.(i).(j)) row) m1 in
      GPU (Gpu.of_cpu res)
  | _, _ -> failwith "Utils.map2_mat: Mixed types"

let scalar s t = 
  match t with
  | CPU m -> CPU (Array.map (fun row -> Array.map (fun x -> x *. s) row) m)
  | GPU g -> 
     (* TODO: Kernel for scalar mult. Fallback. *)
     let m = Gpu.to_cpu g in
     let res = Array.map (fun row -> Array.map (fun x -> x *. s) row) m in
     GPU (Gpu.of_cpu res)

let sum_column t j =
  match t with
  | CPU m -> 
      let s = ref 0.0 in
      let r = Array.length m in
      if r = 0 then 0.0 else (
        for i = 0 to r - 1 do
          let row = m.(i) in
          if j < Array.length row then s := !s +. row.(j)
        done;
        !s)
  | GPU g -> 
      let m = Gpu.to_cpu g in
      (* Same logic *)
      let s = ref 0.0 in
      let r = Array.length m in
      if r = 0 then 0.0 else (
        for i = 0 to r - 1 do
          let row = m.(i) in
          if j < Array.length row then s := !s +. row.(j)
        done;
        !s)

let rec transpose t =
  match t with
  | CPU m -> 
      if Array.length m = 0 then CPU [||]
      else
        let r = Array.length m in
        let c = Array.length m.(0) in
        let res = Array.make_matrix c r 0.0 in
        for i = 0 to r - 1 do
          let row = m.(i) in
          for j = 0 to c - 1 do
            res.(j).(i) <- row.(j)
          done
        done;
        CPU res
  | GPU g ->
      (* Fallback *)
      let m = Gpu.to_cpu g in
      let res = transpose (CPU m) in (* recurse *)
      match res with CPU r -> GPU (Gpu.of_cpu r) | _ -> res

let iter_matrix f t =
  match t with
  | CPU m -> Array.iteri (fun i row -> Array.iteri (fun j x -> f x i j) row) m
  | GPU g -> 
      let m = Gpu.to_cpu g in
      Array.iteri (fun i row -> Array.iteri (fun j x -> f x i j) row) m

let multiply_matrix t1 t2 =
  match t1, t2 with
  | CPU m, CPU n -> 
      if !use_gpu then 
        GPU (Gpu.matmul (Gpu.of_cpu m) (Gpu.of_cpu n))
      else
        if Array.length m = 0 || Array.length n = 0 then CPU [||]
        else (
          assert (Array.length m.(0) = Array.length n);
          let m_rows = Array.length m in
          let m_cols = Array.length m.(0) in
          let n_cols = Array.length n.(0) in
          let out = Array.make_matrix m_rows n_cols 0.0 in
          for i = 0 to m_rows - 1 do
            let mi = m.(i) in
            for k = 0 to m_cols - 1 do
              let mik = mi.(k) in
              let nk = n.(k) in
              for j = 0 to n_cols - 1 do
                out.(i).(j) <- out.(i).(j) +. (mik *. nk.(j))
              done
            done
          done;
          CPU out)
  | GPU g1, GPU g2 -> GPU (Gpu.matmul g1 g2)
  | CPU m, GPU g -> GPU (Gpu.matmul (Gpu.of_cpu m) g)
  | GPU g, CPU m -> GPU (Gpu.matmul g (Gpu.of_cpu m))

let add_matrices t1 t2 =
  match t1, t2 with
  | CPU m, CPU n ->
      if !use_gpu then GPU (Gpu.add (Gpu.of_cpu m) (Gpu.of_cpu n))
      else (
        assert (Array.length m = Array.length n);
        CPU (Array.init (Array.length m) (fun i ->
          let ra = m.(i) and rb = n.(i) in
          Array.init (Array.length ra) (fun j -> ra.(j) +. rb.(j)))))
  | GPU g1, GPU g2 -> GPU (Gpu.add g1 g2)
  | CPU m, GPU g -> GPU (Gpu.add (Gpu.of_cpu m) g)
  | GPU g, CPU m -> GPU (Gpu.add g (Gpu.of_cpu m))

let rec list_iter3 f a b c =
  match (a, b, c) with
  | [], [], [] -> ()
  | h1 :: t1, h2 :: t2, h3 :: t3 ->
      f h1 h2 h3;
      list_iter3 f t1 t2 t3
  | _, _, _ -> failwith "Error: expected lists of same size"

let conv mode t1 t2 =
  match t1, t2 with
  | CPU m1, CPU m2 ->
      if !use_gpu then GPU (Gpu.conv2d mode (Gpu.of_cpu m1) (Gpu.of_cpu m2))
      else (
         if mode <> "full" && mode <> "valid" then failwith "Utils : convolution method is unknown" ;
          let m1_height = Array.length m1 and m1_width = Array.length m1.(0) in
          let m2_height = Array.length m2 and m2_width = Array.length m2.(0) in
          if mode = "full" then begin
            let r = Array.make_matrix (m1_height + m2_height - 1) (m1_width + m2_width - 1) 0. in
            for i = 0 to m1_height + m2_height - 2 do
              for j = 0 to m1_width + m2_width - 2 do
                let sum = ref 0. in
                for k1 = 0 to m2_height - 1 do
                  for k2 = 0 to m2_width - 1 do
                    let i_idx = i - k1 in
                    let j_idx = j - k2 in
                    if i_idx >= 0 && i_idx < m1_height && j_idx >= 0 && j_idx < m1_width then
                      sum := !sum +. (m1.(i_idx).(j_idx) *. m2.(k1).(k2))
                  done
                done;
                r.(i).(j) <- !sum
              done
            done;
            CPU r
          end else begin
             let out_h = m1_height - m2_height + 1 in
            let out_w = m1_width - m2_width + 1 in
            if out_h <= 0 || out_w <= 0 then failwith "Utils: Input smaller than kernel in valid convolution";
            let r = Array.make_matrix out_h out_w 0. in
            for i = 0 to out_h - 1 do
              for j = 0 to out_w - 1 do
                let sum = ref 0. in
                for k1 = 0 to m2_height - 1 do
                  for k2 = 0 to m2_width - 1 do
                     sum := !sum +. (m1.(i + k1).(j + k2) *. m2.(k1).(k2))
                  done
                done;
                r.(i).(j) <- !sum
              done
            done;
            CPU r
          end
      )
  | GPU g1, GPU g2 -> GPU (Gpu.conv2d mode g1 g2)
  | CPU m, GPU g -> GPU (Gpu.conv2d mode (Gpu.of_cpu m) g)
  | GPU g, CPU m -> GPU (Gpu.conv2d mode g (Gpu.of_cpu m))

(* Helpers to force CPU or GPU if needed *)
let to_cpu t = 
  match t with
  | CPU m -> m
  | GPU g -> Gpu.to_cpu g

let to_gpu t =
  match t with
  | CPU m -> GPU (Gpu.of_cpu m)
  | GPU g -> GPU g
