//! odo_replay: the runtime's contact-based odometry (`odometry::Odometry::alpha()`, unchanged) over recorded sim samples.
//! stdin: lines "env tick j0 .. j14 qw qx qy qz [flight]" (JOINT_NAMES order, mouth = 0; one env's ticks in order; flight 1 =
//! both feet up, `Odometry::update_with_flight`, 7-15).
//! stdout: lines "env tick x y z yaw", one per input line, flushed per line (usable online through a pipe).
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use duck_ipc_proto::JOINT_NAMES;
use odometry::Odometry;

fn main() {
    let mut odo: HashMap<u32, Odometry> = HashMap::new();
    let out = io::stdout();
    let mut w = out.lock();
    for line in io::stdin().lock().lines() {
        let line = line.expect("read");
        let v: Vec<f64> = line.split_whitespace().map(|s| s.parse().expect("number")).collect();
        let (env, tick) = (v[0] as u32, v[1] as u64);
        let mut joints = [0.0; JOINT_NAMES.len()];
        joints.copy_from_slice(&v[2..2 + JOINT_NAMES.len()]);
        let q = &v[2 + JOINT_NAMES.len()..];
        let o = odo.entry(env).or_insert_with(Odometry::alpha);
        o.update_with_flight(&joints, [q[0], q[1], q[2], q[3]], q.get(4).is_some_and(|f| *f != 0.0));
        let p = o.position();
        writeln!(w, "{} {} {:.6} {:.6} {:.6} {:.6}", env, tick, p[0], p[1], p[2], o.yaw()).unwrap();
        w.flush().unwrap();
    }
}
