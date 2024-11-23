use serde::{
    de::Deserialize,
    ser::{Serialize, Serializer},
};

use crate::Kc;

impl Serialize for Kc {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unsafe {
            // SAFETY: ggez::event::KeyCode has #[repr(u32)]
            std::mem::transmute::<ggez::event::KeyCode, u32>(self.0).serialize(serializer)
        }
    }
}

impl Deserialize<'de> for Kc {
    // Required method
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>;
}
