use std::fmt;

#[derive(Debug)]
pub struct YoloError {
    pub message: String,
}

impl YoloError {
    pub fn new<E: std::error::Error>(err: E) -> Self {
        YoloError {
            message: err.to_string(),
        }
    }

    pub fn from_message<S: Into<String>>(msg: S) -> Self {
        YoloError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for YoloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {}", self.message)
    }
}

impl std::error::Error for YoloError {}
