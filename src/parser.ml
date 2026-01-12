open Types

let save_model (g : genome) (filepath : string) =
  let oc = open_out_bin filepath in
  Marshal.to_channel oc g [];
  close_out oc

let load_model (filepath : string) : genome =
  let ic = open_in_bin filepath in
  let g : genome = Marshal.from_channel ic in
  close_in ic;
  g
