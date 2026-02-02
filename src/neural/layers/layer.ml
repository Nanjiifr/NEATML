(** Layer type - currently only supports Linear layers.
    Future extensions may include Conv2d, BatchNorm, etc. *)
type t =
  | Linear of Linear.t
  | Conv2d of Conv2d.t
(*| BatchNorm of BatchNorm.t *)
;;
