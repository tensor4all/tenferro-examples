use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fno_burgers"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn help_is_recognized_in_flag_position() {
    for args in [vec!["--help"], vec!["--epochs", "5", "--help"]] {
        let output = run(&args);
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).starts_with("fno_burgers "));
    }
}

#[test]
fn help_in_value_position_does_not_bypass_invalid_arguments() {
    for args in [
        vec!["--epochs", "--help"],
        vec!["--plot", "--help", "--epochs", "not-a-number"],
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "accepted {args:?}: {output:?}");
    }
}
