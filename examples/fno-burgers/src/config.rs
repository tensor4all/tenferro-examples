use std::error::Error;

#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) epochs: usize,
    pub(crate) seed: u64,
    pub(crate) width: usize,
    pub(crate) train_samples: usize,
    pub(crate) test_samples: usize,
    pub(crate) batch: usize,
    pub(crate) plot: Option<String>,
}

impl Config {
    pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut config = Self {
            epochs: 1000,
            seed: 42,
            width: 16,
            train_samples: 256,
            test_samples: 64,
            batch: 16,
            plot: None,
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--epochs" => config.epochs = value.parse()?,
                "--seed" => config.seed = value.parse()?,
                "--width" => config.width = value.parse()?,
                "--train-samples" => config.train_samples = value.parse()?,
                "--test-samples" => config.test_samples = value.parse()?,
                "--batch" => config.batch = value.parse()?,
                "--plot" => config.plot = Some(value),
                _ => return Err(format!("unknown option {flag}; use --help").into()),
            }
        }
        if [
            config.epochs,
            config.width,
            config.train_samples,
            config.test_samples,
            config.batch,
        ]
        .contains(&0)
        {
            return Err("epochs, width, sample counts and batch must be positive".into());
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests;
