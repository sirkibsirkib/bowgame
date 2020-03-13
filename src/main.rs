use ggez::audio::SoundSource;
use ggez::audio::Source;
use ggez::event::EventHandler;
use ggez::event::KeyCode;
use ggez::event::KeyMods;
use ggez::event::MouseButton;
use ggez::{
    graphics::{
        self, spritebatch::SpriteBatch, Color, DrawMode, DrawParam, FilterMode, Image, Mesh,
        MeshBuilder, Rect, BLACK, WHITE,
    },
    nalgebra::{self as na, geometry::Rotation2},
    *,
};

#[derive(Default)]
struct Pressing {
    right: Option<bool>,
    down: Option<bool>,
}

const ROOT_OF_2: f32 = 1.41421356;

mod assets;
use assets::*;
mod helper;
use helper::*;

fn main() {
    // Make a Context.
    let (mut ctx, mut event_loop) = ContextBuilder::new("my_game", "Cool Game Author")
        .build()
        .expect("aieee, could not create ggez context!");

    let mut my_game = MyGame::new(&mut ctx);
    match event::run(&mut ctx, &mut event_loop, &mut my_game) {
        Ok(_) => println!("Exited cleanly."),
        Err(e) => println!("Error occured: {}", e),
    }
}
#[inline]
fn ortho(n: Pt3) -> Pt2 {
    [n[0], n[1] / ROOT_OF_2 + n[2] / ROOT_OF_2].into()
}

type Pt2 = na::Point2<f32>;
type Pt3 = na::Point3<f32>;

struct Assets {
    audio: AudioAssets,
    tex: TexAssets,
}
struct TexAssets {
    arrow_batch: SpriteBatch,
    // arrowhead: Mesh,
    // arrowshaft: Mesh,
    limb: Mesh,
    unit_line: Mesh,
    archer_back: Image,
    archer: Image,
    archer_front: Image,
}
struct AudioAssets {
    taut: [Source; 4],
    loose: [Source; 1],
    twang: [Source; 3],
}

struct MyGame {
    ticks: usize,
    pressing: Pressing,
    dude: Pt3,
    time: f32,
    arrows: Vec<Arrow>,
    stuck_arrows: Vec<StuckArrow>,
    nocked: Option<Nocked>,
    assets: Assets,
}
struct Nocked {
    start: Pt2,
    aim_right: bool,
    tautness: Tautness,
}
fn pt2_to_pt3(xy: Pt2, z: f32) -> Pt3 {
    Pt3::new(xy[0], xy[1], z)
}
struct Arrow {
    pos: Pt3,
    vel: Pt3,
}
struct StuckArrow {
    pos: Pt3,
    vel: Pt3,
    // vibration_amplitude: f32,
}
impl Arrow {
    fn rot_of_xy(p: Pt2) -> f32 {
        Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &p.coords).angle()
    }
    fn stick(self) -> StuckArrow {
        let Self { mut pos, vel } = self;
        pos[2] = 0.;
        StuckArrow { pos, vel }
    }
    fn vel_to_rot_len(mut vel_xyz: Pt3) -> [f32; 2] {
        vel_xyz.coords = vel_xyz.coords.normalize();
        let xy = ortho(vel_xyz);
        [Self::rot_of_xy(xy), xy.coords.norm()]
    }
    fn vel_to_rot_len_shadow(mut vel_xyz: Pt3) -> [f32; 2] {
        vel_xyz.coords = vel_xyz.coords.normalize();
        vel_xyz[2] = 0.;
        let xy = ortho(vel_xyz);
        [Self::rot_of_xy(xy), xy.coords.norm()]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Tautness {
    None,
    Low,
    Med,
    High,
}
impl Tautness {
    fn level(self) -> usize {
        match self {
            Tautness::None => 0,
            Tautness::Low => 1,
            Tautness::Med => 2,
            Tautness::High => 3,
        }
    }
}

trait Squarable: core::ops::Mul + Copy {
    fn sqr(self) -> <Self as core::ops::Mul>::Output {
        self * self
    }
}
impl<T: core::ops::Mul + Copy> Squarable for T {}

impl EventHandler for MyGame {
    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        match keycode {
            KeyCode::W => self.pressing.down = Some(false),
            KeyCode::A => self.pressing.right = Some(false),
            KeyCode::S => self.pressing.down = Some(true),
            KeyCode::D => self.pressing.right = Some(true),
            KeyCode::Escape => ggez::event::quit(ctx),
            _ => {}
        }
    }
    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods) {
        match keycode {
            KeyCode::W if self.pressing.down == Some(false) => self.pressing.down = None,
            KeyCode::A if self.pressing.right == Some(false) => self.pressing.right = None,
            KeyCode::S if self.pressing.down == Some(true) => self.pressing.down = None,
            KeyCode::D if self.pressing.right == Some(true) => self.pressing.right = None,
            _ => {}
        }
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        self.time += 1.0;
        let mut draining = Draining::new(&mut self.arrows);
        let mut speed = WALK_SPEED;
        if self.pressing.down.is_some() && self.pressing.right.is_some() {
            speed /= ROOT_OF_2
        };
        match (self.pressing.right, &self.nocked) {
            (Some(walk_right), Some(Nocked { aim_right, .. })) => {
                speed *= if walk_right == *aim_right { 0.666 } else { 0.333 };
            }
            _ => {}
        }
        self.dude[0] += match self.pressing.right {
            Some(true) => speed,
            Some(false) => -speed,
            _ => 0.,
        };
        self.dude[1] += match self.pressing.down {
            Some(true) => speed,
            Some(false) => -speed,
            _ => 0.,
        };
        while let Some(mut entry) = draining.next() {
            let arrow: &mut Arrow = entry.get_mut();
            let nsq = arrow.vel.coords.norm_squared().sqr() + 1.;
            arrow.vel += Pt3::new(0., 0., 0.3).coords;
            arrow.vel *= nsq.powf(0.995) / nsq;
            arrow.pos += arrow.vel.coords;
            if arrow.pos[2] >= 0. {
                let index = unsafe { std::mem::transmute::<_, f32>(self.time) } as usize % 3;
                self.assets.audio.twang[index].play().unwrap();
                self.stuck_arrows.push(entry.take().stick())
            }
        }
        self.ticks = self.ticks.wrapping_add(1);
        Ok(())
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        if let MouseButton::Left = button {
            let start = [x, y].into();
            self.nocked = Some(Nocked { start, aim_right: true, tautness: Tautness::None });
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        if let MouseButton::Left = button {
            if let Some(Nocked { start, tautness, .. }) = self.nocked.take() {
                if tautness != Tautness::None {
                    let second: Pt2 = [x, y].into();
                    let diff = (start - second.coords) * 0.08;
                    let vel = [diff[0], diff[1] / ROOT_OF_2, -9.].into();
                    self.assets.audio.loose[0].play().unwrap();
                    self.assets.audio.taut[tautness as usize].stop();
                    self.arrows.push(Arrow { pos: self.dude + Pt3::new(0., 0., 2.).coords, vel });
                }
            }
        } else {
            self.stuck_arrows.extend(self.arrows.drain(..).map(Arrow::stick));
        }
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        if let Some(Nocked { start, tautness, aim_right }) = &mut self.nocked {
            let at: Pt2 = [x, y].into();
            let diff = at - start.coords;
            *aim_right = diff[0] < 0.;
            let new_tautness = match diff.coords.norm_squared() {
                x if x < 50.0f32.sqr() => Tautness::None,
                x if x < 100.0f32.sqr() => Tautness::Low,
                x if x < 150.0f32.sqr() => Tautness::Med,
                _ => Tautness::High,
            };
            if *tautness != new_tautness {
                self.assets.audio[*tautness].stop();
                self.assets.audio[new_tautness].play().unwrap();
                *tautness = new_tautness;
            }
        }
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, GREEN);
        // the dude
        const DUDE_SCALE: [f32; 2] = [2.; 2];
        const ARROW_SCALE: f32 = 1.6;
        let right_facing =
            self.nocked.as_ref().map(|x| x.aim_right).or(self.pressing.right).unwrap_or(true);
        let facing_dude_scale = [
            //
            DUDE_SCALE[0] * if right_facing { 1. } else { -1. },
            DUDE_SCALE[1],
        ];
        let dude_param = DrawParam {
            src: Rect { x: 0., y: 0., h: 0.2, w: 0.2 },
            dest: ortho(self.dude).into(),
            color: WHITE,
            scale: facing_dude_scale.into(),
            offset: [0.5, 0.5].into(),
            ..Default::default()
        };
        fn flatten(p: Pt3) -> Pt3 {
            [p[0], p[1], 0.].into()
        }
        for arrow in &self.arrows {
            // shadow
            let dest = ortho(flatten(arrow.pos));
            let [rotation, len] = Arrow::vel_to_rot_len_shadow(flatten(arrow.vel));
            let p = DrawParam {
                dest: dest.into(),
                scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                rotation,
                color: BLACK,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            //real
            self.assets.tex.arrow_batch.add(p);
            let dest = ortho(arrow.pos);
            let [rotation, len] = Arrow::vel_to_rot_len(arrow.vel);
            let p = DrawParam {
                dest: dest.into(),
                scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                rotation,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            self.assets.tex.arrow_batch.add(p);
        }
        for arrow in &mut self.stuck_arrows {
            // shadow
            let dest = ortho(flatten(arrow.pos));
            let [rotation, len] = Arrow::vel_to_rot_len_shadow(flatten(arrow.vel));
            let p = DrawParam {
                dest: dest.into(),
                scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                rotation,
                color: BLACK,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            //real
            self.assets.tex.arrow_batch.add(p);
            let dest = ortho(arrow.pos);
            let [rotation, len] = Arrow::vel_to_rot_len(arrow.vel);
            let p = DrawParam {
                dest: dest.into(),
                scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                rotation,
                offset: [0.95, 0.5].into(),
                ..Default::default()
            };
            self.assets.tex.arrow_batch.add(p);
        }
        let main_src = {
            let x = if self.pressing.down.is_some() || self.pressing.right.is_some() {
                const FRAME_TICKS: usize = 10;
                match self.ticks % (FRAME_TICKS * 6) {
                    x if x < FRAME_TICKS * 1 => 0.0,
                    x if x < FRAME_TICKS * 2 => 0.2,
                    x if x < FRAME_TICKS * 3 => 0.4,
                    x if x < FRAME_TICKS * 4 => 0.6,
                    x if x < FRAME_TICKS * 5 => 0.4,
                    ________________________ => 0.2,
                }
            } else {
                0.4
            };
            Rect { x, y: 0., h: 0.2, w: 0.2 }
        };
        let end: Pt2 = ggez::input::mouse::position(ctx).into();
        match self.nocked {
            Some(Nocked { start, tautness, .. }) if start != end => {
                let dif = end - start.coords;
                let dif_rotation =
                    Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &dif.coords);
                let dif_rotation_angle = dif_rotation.angle();
                // let dif_rotation_angle_neg = dif_rotation_angle + core::f32::consts::PI;

                let aim_angle = if tautness.level() >= 2 {
                    dif_rotation_angle + core::f32::consts::PI
                } else {
                    0.
                };
                let arm_angle =
                    if right_facing { aim_angle } else { aim_angle + core::f32::consts::PI };

                // draw the dude
                let arm_src = Rect { x: 0.2 * tautness.level() as f32, y: 0., h: 0.2, w: 0.2 };
                graphics::draw(
                    ctx,
                    &self.assets.tex.archer_back,
                    DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
                )?;
                graphics::draw(
                    ctx,
                    &self.assets.tex.archer,
                    DrawParam { src: main_src, scale: facing_dude_scale.into(), ..dude_param },
                )?;
                graphics::draw(
                    ctx,
                    &self.assets.tex.archer_front,
                    DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
                )?;

                // nocked arrow
                let nock_at = self.dude + Pt3::new(0., 0., -3.).coords;
                let p = DrawParam {
                    dest: ortho(nock_at).into(),
                    color: WHITE,
                    scale: [ARROW_SCALE; 2].into(),
                    offset: [0., 0.5].into(),
                    rotation: aim_angle,
                    ..Default::default()
                };
                self.assets.tex.arrow_batch.add(p);

                // white line
                // graphics::draw(
                //     ctx,
                //     &self.assets.tex.unit_line,
                //     DrawParam { src: arm_src, rotation: draw_angle, ..dude_param },
                // )?;
            }
            _ => {
                let src = Rect { x: 0., y: 0., h: 0.2, w: 0.2 };
                graphics::draw(ctx, &self.assets.tex.archer_back, DrawParam { src, ..dude_param })?;
                graphics::draw(
                    ctx,
                    &self.assets.tex.archer,
                    DrawParam { src: main_src, ..dude_param },
                )?;
                graphics::draw(
                    ctx,
                    &self.assets.tex.archer_front,
                    DrawParam { src, ..dude_param },
                )?;
            }
        }
        graphics::draw(ctx, &self.assets.tex.arrow_batch, DrawParam::default())?;
        self.assets.tex.arrow_batch.clear();
        graphics::present(ctx)
    }
}
