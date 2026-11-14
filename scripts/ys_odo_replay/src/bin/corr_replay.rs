//! corr_replay <model.bin>: the runtime's learned position correction (`odometry::Correction`, 7-28) over one recorded game.
//! stdin: lines "active o0 .. o72" (one per 20 ms tick, in order; active 1 = from J's set-off to its end call; o = what J's policy is given).
//! stdout: lines "cx cy" (the correction in use after that tick, m), one per input line.
use std::io::{self, BufRead, Write};
use odometry::Correction;

fn main() {
    let path = std::env::args().nth(1).expect("usage: corr_replay <model.bin>");
    let mut corr = Correction::load(path.as_ref()).expect("model");
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    for line in io::stdin().lock().lines() {
        let v: Vec<f32> = line.expect("read").split_whitespace().map(|s| s.parse().expect("number")).collect();
        let [x, y] = corr.step(&v[1..], v[0] != 0.0, 0.02);
        writeln!(w, "{x:.9} {y:.9}").unwrap();
    }
}
