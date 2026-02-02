(** Tensor type representing 2D arrays (matrices) of floating-point numbers.
    Used throughout the neural network for representing data, weights, and gradients. *)
type t = float array array

(** Global flag for GPU acceleration *)
let use_gpu = ref false

(** [enable_gpu ()] enables GPU acceleration using Metal backend. *)
let enable_gpu () = use_gpu := true

(** [disable_gpu ()] disables GPU acceleration (reverts to CPU). *)
let disable_gpu () = use_gpu := false

(** [rows t] returns the number of rows in a tensor.
    @param t The input tensor
    @return The number of rows *)
let rows (t : Tensor.t) = Array.length t

(** [cols t] returns the number of columns in a tensor.
    Returns 0 if the tensor has no rows.
    @param t The input tensor
    @return The number of columns *)
let cols (t : Tensor.t) = if rows t = 0 then 0 else Array.length t.(0)

(** [zeros r c] creates a zero-initialized tensor.
    @param r Number of rows
    @param c Number of columns
    @return A tensor of shape (r, c) filled with zeros *)
let zeros r c = Array.init r (fun _ -> Array.make c 0.0)

(** [copy_mat m] creates a deep copy of a matrix.
    @param m The matrix to copy
    @return A new matrix with the same values *)
let copy_mat m = Array.map Array.copy m

(** [copy_mat_inplace m dest] copies matrix m into dest in-place.
    @param m The source matrix
    @param dest The destination matrix *)
let copy_mat_inplace m dest =
  Array.iteri (fun i a -> Array.iteri (fun j v -> dest.(i).(j) <- v) a) m

(** [map_mat f m] applies a function element-wise to a matrix.
    The function receives the value and its row and column indices.
    @param f Function (value -> row_index -> col_index -> new_value)
    @param m The input matrix
    @return A new matrix with the function applied *)
let map_mat f m =
  Array.mapi (fun i row -> Array.mapi (fun j x -> f x i j) row) m

(** [map2_mat f m1 m2] applies a function element-wise to two matrices of the same size.
    @param f Function (value1 -> value2 -> result)
    @param m1 The first matrix
    @param m2 The second matrix
    @return A new matrix with the function applied
    @raise Failure if matrices have different dimensions *)
let map2_mat f m1 m2 =
  if
    Array.length m1 <> Array.length m2
    || Array.length m1.(0) <> Array.length m2.(0)
  then failwith "Error: expected same sized matrices";
  Array.mapi (fun i row -> Array.mapi (fun j x -> f x m2.(i).(j)) row) m1

(** [scalar s m] multiplies all elements of a matrix by a scalar.
    @param s The scalar multiplier
    @param m The input matrix
    @return A new matrix with all elements multiplied by s *)
let scalar s m = map_mat (fun x _ _ -> x *. s) m

(** [sum_column m j] computes the sum of all elements in column j.
    @param m The input matrix
    @param j The column index
    @return The sum of column j *)
let sum_column (m : Tensor.t) j =
  let s = ref 0.0 in
  let r = Array.length m in
  if r = 0 then 0.0
  else (
    for i = 0 to r - 1 do
      let row = m.(i) in
      if j < Array.length row then s := !s +. row.(j)
    done;
    !s)

(** [transpose m] transposes a matrix (swaps rows and columns).
    @param m The input matrix of shape (r, c)
    @return The transposed matrix of shape (c, r) *)
let transpose m =
  if Array.length m = 0 then [||]
  else
    let r = Array.length m in
    let c = Array.length m.(0) in
    let t = Array.make_matrix c r 0.0 in
    for i = 0 to r - 1 do
      let row = m.(i) in
      for j = 0 to c - 1 do
        t.(j).(i) <- row.(j)
      done
    done;
    t

(** [iter_matrix f m] iterates over all elements of a matrix.
    @param f Function (value -> row_index -> col_index -> unit)
    @param m The input matrix *)
let iter_matrix f m =
  Array.iteri (fun i row -> Array.iteri (fun j x -> f x i j) row) m

(** [multiply_matrix m n] performs matrix multiplication m @ n.
    @param m The left matrix of shape (m_rows, m_cols)
    @param n The right matrix of shape (m_cols, n_cols)
    @return The product matrix of shape (m_rows, n_cols)
    @raise Assert_failure if dimensions are incompatible *)
let multiply_matrix m n =
  if !use_gpu then Gpu.matmul m n
  else
  if Array.length m = 0 || Array.length n = 0 then [||]
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
    out)

(** [add_matrices m n] performs element-wise matrix addition.
    @param m The first matrix
    @param n The second matrix (must have same shape as m)
    @return The sum matrix
    @raise Assert_failure if dimensions don't match *)
let add_matrices m n =
  if !use_gpu then Gpu.add m n
  else (
  assert (
    Array.length m = Array.length n
    && (Array.length m = 0 || Array.length m.(0) = Array.length n.(0)));
  Array.init (Array.length m) (fun i ->
      let ra = m.(i) and rb = n.(i) in
      Array.init (Array.length ra) (fun j -> ra.(j) +. rb.(j)))
  )

(** [list_iter3 f a b c] iterates simultaneously over three lists of the same length.
    @param f Function (elem_a -> elem_b -> elem_c -> unit)
    @param a First list
    @param b Second list
    @param c Third list
    @raise Failure if lists have different lengths *)
let rec list_iter3 f a b c =
  match (a, b, c) with
  | [], [], [] -> ()
  | h1 :: t1, h2 :: t2, h3 :: t3 ->
      f h1 h2 h3;
      list_iter3 f t1 t2 t3
  | _, _, _ -> failwith "Error: expected lists of same size"

let conv m m1 m2 =
  if !use_gpu then Gpu.conv2d m m1 m2
  else (
  if m <> "full" && m <> "valid" then failwith "Utils : convolution method is unknown" ;
  let m1_height = Array.length m1 and m1_width = Array.length m1.(0) in
  let m2_height = Array.length m2 and m2_width = Array.length m2.(0) in
  if m = "full" then begin
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
    r
  end else begin (* valid *)
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
    r
  end
  )