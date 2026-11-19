use super::*;

fn parse(args: &[&str]) -> Result<Config, Box<dyn Error>> {
    Config::parse(args.iter().map(|s| (*s).to_owned()))
}

#[test]
fn defaults_and_short_run_overrides() {
    let c = parse(&[]).unwrap();
    assert_eq!(
        (
            c.epochs,
            c.width,
            c.train_samples,
            c.test_samples,
            c.batch,
            c.seed
        ),
        (1000, 16, 256, 64, 16, 42)
    );
    assert!(c.plot.is_none());
    let c = parse(&[
        "--epochs",
        "3",
        "--width",
        "4",
        "--train-samples",
        "8",
        "--test-samples",
        "4",
        "--batch",
        "4",
        "--seed",
        "7",
        "--plot",
        "result.png",
    ])
    .unwrap();
    assert_eq!(
        (
            c.epochs,
            c.width,
            c.train_samples,
            c.test_samples,
            c.batch,
            c.seed
        ),
        (3, 4, 8, 4, 4, 7)
    );
    assert_eq!(c.plot.as_deref(), Some("result.png"));
}

#[test]
fn rejects_invalid_or_missing_values_and_unknown_flags() {
    for args in [
        vec!["--unknown", "1"],
        vec!["--epochs"],
        vec!["--epochs", "0"],
        vec!["--width", "0"],
        vec!["--batch", "0"],
        vec!["--test-samples", "0"],
        vec!["--train-samples", "0"],
        vec!["--epochs", "-1"],
        vec!["--seed", "x"],
        vec!["--plot"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}
