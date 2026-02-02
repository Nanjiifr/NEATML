(** Global flag for GPU acceleration *)
val use_gpu : bool ref

(** [enable_gpu ()] enables GPU acceleration using Metal backend. *)
val enable_gpu : unit -> unit

(** [disable_gpu ()] disables GPU acceleration (reverts to CPU). *)
val disable_gpu : unit -> unit

(** [rows t] returns the number of rows in a tensor.
    @param t The input tensor
    @return The number of rows *)
val rows : Tensor.t -> int

(** [cols t] returns the number of columns in a tensor.
    Returns 0 if the tensor has no rows.
    @param t The input tensor
    @return The number of columns *)
val cols : Tensor.t -> int

(** [zeros r c] creates a zero-initialized tensor.
    @param r Number of rows
    @param c Number of columns
    @return A tensor of shape (r, c) filled with zeros *)
val zeros : int -> int -> Tensor.t

(** [copy_mat m] creates a deep copy of a matrix.
    @param m The matrix to copy
    @return A new matrix with the same values *)
val copy_mat : 'a array array -> 'a array array

(** [copy_mat_inplace m dest] copies matrix m into dest in-place.
    @param m The source matrix
    @param dest The destination matrix *)
val copy_mat_inplace : 'a array array -> 'a array array -> unit

(** [map_mat f m] applies a function element-wise to a matrix.
    The function receives the value and its row and column indices.
    @param f Function (value -> row_index -> col_index -> new_value)
    @param m The input matrix
    @return A new matrix with the function applied *)
val map_mat : (float -> int -> int -> float) -> Tensor.t -> Tensor.t

(** [sum_column m j] computes the sum of all elements in column j.
    @param m The input matrix
    @param j The column index
    @return The sum of column j *)
val sum_column : Tensor.t -> int -> float

(** [map2_mat f m1 m2] applies a function element-wise to two matrices of the same size.
    @param f Function (value1 -> value2 -> result)
    @param m1 The first matrix
    @param m2 The second matrix
    @return A new matrix with the function applied
    @raise Failure if matrices have different dimensions *)
val map2_mat :
  ('a -> 'b -> 'c) -> 'a array array -> 'b array array -> 'c array array

(** [transpose m] transposes a matrix (swaps rows and columns).
    @param m The input matrix of shape (r, c)
    @return The transposed matrix of shape (c, r) *)
val transpose : Tensor.t -> Tensor.t

(** [scalar s m] multiplies all elements of a matrix by a scalar.
    @param s The scalar multiplier
    @param m The input matrix
    @return A new matrix with all elements multiplied by s *)
val scalar : float -> Tensor.t -> Tensor.t

(** [multiply_matrix m n] performs matrix multiplication m @ n.
    @param m The left matrix of shape (m_rows, m_cols)
    @param n The right matrix of shape (m_cols, n_cols)
    @return The product matrix of shape (m_rows, n_cols)
    @raise Assert_failure if dimensions are incompatible *)
val multiply_matrix : Tensor.t -> Tensor.t -> Tensor.t

(** [add_matrices m n] performs element-wise matrix addition.
    @param m The first matrix
    @param n The second matrix (must have same shape as m)
    @return The sum matrix
    @raise Assert_failure if dimensions don't match *)
val add_matrices : Tensor.t -> Tensor.t -> Tensor.t

(** [iter_matrix f m] iterates over all elements of a matrix.
    @param f Function (value -> row_index -> col_index -> unit)
    @param m The input matrix *)
val iter_matrix : ('a -> int -> int -> unit) -> 'a array array -> unit

(** [list_iter3 f a b c] iterates simultaneously over three lists of the same length.
    @param f Function (elem_a -> elem_b -> elem_c -> unit)
    @param a First list
    @param b Second list
    @param c Third list
    @raise Failure if lists have different lengths *)
val list_iter3 :
  ('a -> 'b -> 'c -> unit) -> 'a list -> 'b list -> 'c list -> unit

val conv : string -> float array array -> float array array -> float array array
