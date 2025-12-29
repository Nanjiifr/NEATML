type direction = Up | Down | Left | Right
type position = int * int (* x, y *)

type state = {
  snake : position list;
  food : position;
  size : int;
  score : int;
  steps : int;
  alive : bool;
}

let size = 20

let init_game size =
  let mid = size / 2 in
  let snake = [ (mid, mid); (mid, mid + 1); (mid, mid + 2) ] in
  let food_temp = ref (Random.int size, Random.int size) in
  while List.mem !food_temp snake do
    food_temp := (Random.int size, Random.int size)
  done;
  let food = !food_temp in
  let score = 0 in
  let steps = 0 in
  let alive = true in

  { snake; food; size; score; steps; alive }

let get_size state = state.size

let spawn_food size snake =
  let food = ref (Random.int size, Random.int size) in
  while List.mem !food snake do
    food := (Random.int size, Random.int size)
  done;
  !food

let next_head snake_head dir =
  let x, y = snake_head in
  match dir with
  | Up -> (x + 1, y)
  | Down -> (x - 1, y)
  | Right -> (x, y + 1)
  | Left -> (x, y - 1)

let step s dir =
  let new_head = next_head (List.hd s.snake) dir in
  let nx, ny = new_head in
  if
    nx = s.size || nx = -1 || ny = s.size || ny = -1 || s.steps >= 100
    || List.mem new_head s.snake
  then { s with alive = false }
  else if new_head = s.food then
    let new_snake = new_head :: s.snake in
    let new_food = spawn_food s.size s.snake in
    {
      s with
      steps = 0;
      snake = new_snake;
      food = new_food;
      score = s.score + 1;
    }
  else
    let new_snake = List.take (s.score + 3) (new_head :: s.snake) in
    { s with snake = new_snake; steps = s.steps + 1 }

let dist x1 y1 x2 y2 = abs (x1 - x2) + abs (y1 - y2) (*Distance de Manhattan*)

let get_wall_dist snake size dir =
  let rec aux pos d =
    let x, y = pos in
    if d >= size then size
    else if x = size || x = -1 || y = size || y = -1 then d
    else aux (next_head pos dir) (d + 1)
  in
  aux (List.hd snake) 0

let get_food_dist snake food size dir =
  let rec aux pos d =
    if d >= size then size
    else if pos = food then d
    else aux (next_head pos dir) (d + 1)
  in
  aux (List.hd snake) 0

let get_tail_dist snake size dir =
  let rec aux pos d =
    if d >= size then size
    else if List.mem pos (List.tl snake) then d
    else aux (next_head pos dir) (d + 1)
  in
  aux (List.hd snake) 0

let get_inputs state =
  let inputs = ref [] in
  let forward_dir =
    let hd_x, hd_y = List.hd state.snake in
    let snd_x, snd_y = List.hd (List.tl state.snake) in
    match hd_x - snd_x with
    | 1 -> Down
    | -1 -> Up
    | _ -> (
        match hd_y - snd_y with
        | 1 -> Right
        | -1 -> Left
        | _ -> failwith "get_inputs: Failed to get snake direction\n")
  in
  let rel_right, rel_left =
    match forward_dir with
    | Up -> (Right, Left)
    | Down -> (Left, Right)
    | Right -> (Down, Up)
    | Left -> (Up, Down)
  in
  let dirs = [ forward_dir; rel_right; rel_left ] in
  List.iter
    (fun dir ->
      let max_dist = state.size in
      let dist_wall = get_wall_dist state.snake state.size dir in
      let dist_food = get_food_dist state.snake state.food state.size dir in
      let dist_tail = get_tail_dist state.snake state.size dir in
      let norm_wall = float (max_dist - dist_wall) /. float max_dist in
      let norm_food = float (max_dist - dist_food) /. float max_dist in
      let norm_tail = float (max_dist - dist_tail) /. float max_dist in
      inputs := [ norm_wall; norm_food; norm_tail ] @ !inputs)
    dirs;
  !inputs
