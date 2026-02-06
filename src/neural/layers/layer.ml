(** Layer type - currently only supports Linear layers.
    Future extensions may include Conv2d, BatchNorm, etc. *)
type t =
  | Linear of Linear.t
  | Conv2d of Conv2d.t
  | Dropout of Dropout.t
  | MaxPool2d of Pooling.t

let set_training_mode (l : t) (active : bool) =
  match l with
  | Dropout d -> Dropout.set_training_mode d active
  | _ -> () (* Other layers don't have distinct train/eval modes yet *)
(*| BatchNorm of BatchNorm.t *)
;;
