open Types

module InnovationManager = struct
  type t = {
    curr : innovation_id ref;
    mutations : (node_id * node_id * mutation_type, innovation_id) Hashtbl.t;
  }

  let create init_id = { curr = ref init_id; mutations = Hashtbl.create 16 }
  let reset_innovation innov = Hashtbl.reset innov.mutations

  let get_innov_id innov source_id target_id mut =
    match Hashtbl.find_opt innov.mutations (source_id, target_id, mut) with
    | None ->
        incr innov.curr;
        Hashtbl.add innov.mutations (source_id, target_id, mut) !(innov.curr);
        !(innov.curr)
    | Some c -> c
end
