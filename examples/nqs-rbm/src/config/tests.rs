use super::*;

fn parse_args(args: &str) -> Result<Config, String> {
    let args: Vec<String> = args.split_whitespace().map(String::from).collect();
    Config::from_args(&args)
}

#[test]
fn defaults_follow_the_selected_system() {
    assert_eq!(
        parse_args("").unwrap(),
        Config::defaults(System::IsingChain)
    );
    let square = parse_args("--system heisenberg-square").unwrap();
    assert_eq!((square.lx(), square.ly(), square.alpha), (10, 10, 16));
    assert_eq!((square.sr_shift, square.max_weight_step), (1e-3, 0.002));
    let chain = parse_args("--system heisenberg-chain").unwrap();
    assert_eq!((chain.lx(), chain.ly(), chain.seed), (80, 1, 43));
}

#[test]
fn overrides_apply_regardless_of_order() {
    let config =
        parse_args("--steps 10 --system heisenberg-chain --size 8 --exact --samples 64").unwrap();
    assert_eq!(config.system, System::HeisenbergChain);
    assert_eq!((config.size, config.steps, config.samples), (8, 10, 64));
    assert!(config.exact);
}

#[test]
fn invalid_arguments_are_rejected_with_a_reason() {
    for (args, message) in [
        ("--epochs 3", "unknown option `--epochs`"),
        ("--steps", "missing value after `--steps`"),
        ("--steps 3 --steps 4", "given more than once"),
        ("--steps three", "cannot parse `three`"),
        ("--system kagome", "unknown system `kagome`"),
        ("extra", "unexpected argument `extra`"),
        ("--samples 100", "multiple of 8"),
        ("--system heisenberg-chain --size 7", "at least 4 and even"),
        ("--system heisenberg-square --size 3", "must be even"),
        ("--sr-shift 0", "--sr-shift must be positive"),
        (
            "--max-weight-step -1",
            "--max-weight-step must be non-negative",
        ),
        ("--validation-sweeps 100", "multiple of 64"),
        ("--exact", "at most 4096 states"),
        (
            "--system heisenberg-square --size 4 --exact",
            "16 sites have more",
        ),
    ] {
        let error = parse_args(args).expect_err(args);
        assert!(error.contains(message), "`{args}` gave `{error}`");
    }
}

#[test]
fn exact_check_accepts_small_lattices() {
    assert!(parse_args("--size 12 --exact").is_ok());
    assert!(parse_args("--system heisenberg-chain --size 14 --exact").is_ok());
    assert!(parse_args("--size 13 --exact").is_err());
}
