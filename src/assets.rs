use super::*;

pub const BROWN: Color = Color { r: 0.5, g: 0.2, b: 0.2, a: 1. };
pub const GREEN: Color = Color { r: 0.1, g: 0.4, b: 0.1, a: 1. };
pub const RED: Color = Color { r: 1., g: 0., b: 0., a: 1. };
pub const BLUE: Color = Color { r: 0., g: 0., b: 1., a: 1. };
pub const FLETCH_THICKNESS: f32 = 0.07;
pub const FLETCH_LENGTH: f32 = 0.2;
pub const FLETCH_INDENT: f32 = 0.04;
pub const HEAD_THICKNESS: f32 = 0.05;
pub const HEAD_LENGTH: f32 = 0.13;
pub const SHAFT_THICKNESS: f32 = 0.017;
pub const LIMB_WIDTH: f32 = 70.0;
pub const LIMB_DEPTH: f32 = 35.0;
pub const WALK_SPEED: f32 = 4.0;

impl core::ops::Index<Tautness> for AudioAssets {
    type Output = Source;
    fn index(&self, t: Tautness) -> &Self::Output {
        &self.taut[t.level()]
    }
}
impl core::ops::IndexMut<Tautness> for AudioAssets {
    fn index_mut(&mut self, t: Tautness) -> &mut Self::Output {
        &mut self.taut[t.level()]
    }
}
fn linear(mut image: Image) -> Image {
    image.set_filter(FilterMode::Nearest);
    image
}
impl Assets {
    pub fn new(ctx: &mut Context) -> Self {
        let tex = TexAssets {
            doodads: SpriteBatch::new(linear(Image::new(ctx, "/doodads.png").unwrap())),
            archer_back: linear(Image::new(ctx, "/archer_back.png").unwrap()),
            archer: linear(Image::new(ctx, "/archer.png").unwrap()),
            archer_front: linear(Image::new(ctx, "/archer_front.png").unwrap()),
            unit_line: MeshBuilder::new()
                .line(&[[0., 0.], [1., 0.]], 1., graphics::WHITE)
                .unwrap()
                .build(ctx)
                .unwrap(),
            arrow_batch: SpriteBatch::new(Image::new(ctx, "/arrow.png").unwrap()),
            limb: MeshBuilder::new()
                .polygon(
                    DrawMode::fill(),
                    &[
                        [1.0, 1.0],
                        [0.89, 0.8],
                        [0.64, 0.6],
                        [0.4, 0.4],
                        [0.1, 0.2],
                        [0.0, 0.0],
                        [0.1, 0.0],
                        [0.2, 0.2],
                        [0.5, 0.4],
                        [0.71, 0.6],
                        [0.94, 0.8],
                    ],
                    WHITE,
                )
                .unwrap()
                .build(ctx)
                .unwrap(),
        };
        let audio = AudioAssets {
            taut: [
                ggez::audio::Source::new(ctx, "/taut1.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut2.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut3.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut4.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/taut4.wav").unwrap(),
            ],
            twang: [
                ggez::audio::Source::new(ctx, "/twang1.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/twang2.wav").unwrap(),
                ggez::audio::Source::new(ctx, "/twang3.wav").unwrap(),
            ],
            loose: [
                //
                ggez::audio::Source::new(ctx, "/loose1.wav").unwrap(),
            ],
        };
        Assets { audio, tex }
    }
}
