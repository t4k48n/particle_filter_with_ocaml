let list_init n f =
  let rec loop acc n =
    match n with
      | 0 -> acc
      | n -> loop (f (n-1)::acc) (n - 1)
  in
  loop [] n

let pi = 4.0 *. atan 1.0

type nonrec noise_type_t =
  | Uniform of float * float (* (lower_bound, upper_bound) *)
  | Normal of float * float (* (mean, std. dev.) *)

type nonrec state_t = float

let noise_system = Uniform ((-0.01), 0.01)

let state_add_noise ?(noise_type=noise_system) state =
  let r = match noise_type with
    | Uniform (a, b) -> Random.float (b -. a) +. a
    | Normal (m, s) -> (
        let x = Random.float 1.0 in
        let y = Random.float 1.0 in
        sqrt (-.2. *. log x) *. cos (2. *. pi *. y)
      )
  in
  state +. r

let state_next state =
  let k = 1.0 in
  let dt = 0.01 in
  let state_deriv = -.k *. state**3.0 in
  state +. state_deriv *. dt

let output_of_state state = state

let noise_observation = Uniform ((-0.1), 0.1)

let output_add_noise ?(noise_type=noise_observation) output =
  let r = match noise_type with
    | Uniform (a, b) -> Random.float (b -. a) +. a
    | Normal (m, s) -> (
        let x = Random.float 1.0 in
        let y = Random.float 1.0 in
        sqrt (-.2. *. log x) *. cos (2. *. pi *. y)
      )
  in
  output +. r

let state_likelihood ?(noise_type=noise_observation) state observed_output =
  let estimated_output = output_of_state state in
  match noise_type with
    | Uniform (a, b) -> (
        let lower = estimated_output +. a in
        let upper = estimated_output +. b in
        if lower <= observed_output && observed_output <= upper
          then 1.0 /. (b -. a)
          else 0.0
      )

type nonrec particle_t = { weight: float; state: state_t }

let particle_update { weight=w; state=s } observed_output =
  let s = state_add_noise (state_next s) in
  let lh = state_likelihood s observed_output in
  { weight=w*.lh; state=s }

let string_of_particle p =
  string_of_float p.weight ^ "," ^ string_of_float p.state

let particles_normalize ps =
  let sum = List.fold_left (fun acc p -> acc +. p.weight) 0.0 ps in
  List.map (fun {weight=w; state=s} -> {weight=w/.sum; state=s}) ps

type nonrec resampling_method_t =
  | Random

let particles_resample ?(resampling_method=Random) ps =
  match resampling_method with
    | Random -> (
        (* weightの和と累積和*)
        let sum, acc_sum =
          let acc_f acc { weight=w; state=_ } =
            match acc with
              | []          -> [w]
              | a::_ as acc -> (a+.w)::acc
          in
          let acc_sum_r = List.fold_left acc_f [] ps in
          (List.hd acc_sum_r), (List.rev acc_sum_r)
        in
        let sample ps =
          match ps with
            | [] -> failwith "empty"
            | [p] -> p
            | ps -> (
                let r = Random.float sum in
                let _, p = List.find (fun (a, _) -> r < a) (List.map2 (fun a b -> (a, b)) acc_sum ps) in
                p
              )
        in
        List.map (fun _ -> sample ps) ps
      )

let particles_update ps observed_output =
  let ps = List.map (fun p -> particle_update p observed_output) ps in
  let ps = particles_resample ps in
  particles_normalize ps

let particles_mean ps =
  let w, s = List.fold_left (fun (wsum, ssum) { weight=w; state=s } -> (wsum+.w, ssum+.w*.s)) (0.0, 0.0) ps in
  s /. w

let particles_series particle_num series_num =
  if particle_num <=0 || series_num <= 0 then failwith "invalid number"
  else
  let state_init = 1.0 in
  let observation_init = output_add_noise (output_of_state state_init) in
  let noise_init = Uniform ((-0.1), 0.1) in
  let particles_init =
    particles_normalize
      (
        list_init
        particle_num
        (fun _ -> { weight=1.0 /. float particle_num; state=state_add_noise ~noise_type:noise_init state_init})
      )
  in
  let rec loop n acc =
    match n with
      | 1 -> acc
      | n -> (
          match acc with
            | []                -> failwith "empty"
            | (ps, s, o)::_ as acc -> (
                let ns = state_add_noise (state_next s) in
                loop
                  (n - 1)
                  (((particles_update ps o), (ns), (output_add_noise (output_of_state ns)))::acc)
              )
        )
  in
  loop series_num [(particles_init, state_init, observation_init)]

let () =
  let pss = particles_series 500 1000 in
  let particle_states =
    List.map
      (fun (ps, _, _) ->
        List.map (fun { weight=_; state=s } -> s) ps
      )
      pss
  in
  let true_states = List.map (fun (_, s, _) -> s) pss in
  let observations = List.map (fun (_, _, o) -> o) pss in
  let f = open_out "particle_states.csv" in
  List.iter
    (fun ss ->
      let sep = ref "" in
      List.iter (fun s -> Printf.fprintf f "%s%f" !sep s; sep := ",") ss;
      Printf.fprintf f "\n"
    )
    (List.rev particle_states);
  close_out f;
  let f = open_out "true_states.csv" in
  List.iter (fun m -> Printf.fprintf f "%f\n" m) (List.rev true_states);
  close_out f;
  let f = open_out "observations.csv" in
  List.iter (fun m -> Printf.fprintf f "%f\n" m) (List.rev observations);
  close_out f
