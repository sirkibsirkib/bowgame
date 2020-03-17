use core::f32::consts::PI;
use ggez::{
    audio::{SoundSource, Source},
    conf::WindowMode,
    event::{EventHandler, KeyCode, KeyMods, MouseButton},
    graphics::{
        self, spritebatch::SpriteBatch, Color, DrawParam, FilterMode, Image, Mesh, MeshBuilder,
        Rect, BLACK, WHITE,
    },
    nalgebra::{self as na, Rotation2, Rotation3, Transform3, Translation3},
    *,
};

use rand::{
    distributions::{Distribution, Standard},
    rngs::SmallRng,
    Rng, SeedableRng,
};

#[derive(Default)]
struct Pressing {
    right: Option<bool>,
    down: Option<bool>,
    clockwise: Option<bool>,
}

const MIN_VEL: f32 = 9.;
const MAX_VEL: f32 = 35.;

const ROOT_OF_2: f32 = 1.41421356;
const WIN_DIMS: [f32; 2] = [800., 600.];
const ARM_Z: f32 = -29.;
const BADDIE_SPEED: f32 = 0.5;

const SCREEN_TRANS_V2: [f32; 2] = [
    //
    WIN_DIMS[0] * 0.3,
    WIN_DIMS[1] * 0.7,
];
const SCREEN_TRANS_V3: [f32; 3] = [
    //
    WIN_DIMS[0] * 0.3,
    WIN_DIMS[1] * ROOT_OF_2 * 0.7,
    0.,
];
const TO_ARMS: [f32; 3] = [0., 0., -32.];

mod assets;
use assets::*;
mod helper;
use helper::*;

fn main() {
    // Make a Context.
    let (mut ctx, mut event_loop) = ContextBuilder::new("bowdraw", "Christopher Esterhuyse")
        .window_mode(WindowMode { width: WIN_DIMS[0], height: WIN_DIMS[1], ..Default::default() })
        .build()
        .unwrap();
    ggez::input::mouse::set_cursor_grabbed(&mut ctx, true).unwrap();
    let mut my_game = MyGame::new(&mut ctx);
    event::run(&mut ctx, &mut event_loop, &mut my_game).expect("Game Err");
}

type Pt2 = na::Point2<f32>;
type Pt3 = na::Point3<f32>;
type Vec2 = na::Vector2<f32>;
type Vec3 = na::Vector3<f32>;

struct Assets {
    audio: AudioAssets,
    tex: TexAssets,
}
struct TexAssets {
    cross: Mesh,
    arrow_batch: SpriteBatch,
    doodads: SpriteBatch,
    unit_line: Mesh,
    archer_back: Image,
    archer: Image,
    archer_front: Image,
}
struct AudioAssets {
    taut: [Source; 4],
    loose: [Source; 1],
    twang: [Source; 3],
    thud: [Source; 1],
}
enum DoodadKind {
    Bush,
    Rock,
    Pebbles,
    Shrub,
}
struct Doodad {
    kind: DoodadKind,
    pos: Pt3,
}
struct Camera {
    world_pos: Pt3,
    world_rot: f32,
}
struct RclickAnchor {
    anchor_pt: Pt2,
    anchor_angle: f32,
}
struct MyGame {
    faced_right_most_recently: bool,
    aim_assist: u8,
    camera: Camera,
    ticks: usize,
    pressing: Pressing,
    baddies: Vec<Baddie>,
    dude: Entity,
    arrows: Vec<Entity>,
    stuck_arrows: Vec<Entity>,
    nocked: Option<Nocked>,
    rclick_anchor: Option<RclickAnchor>,
    assets: Assets,
    doodads: Vec<Doodad>,
    last_mouse_at: Pt2,
    rng: SmallRng,
}
struct Nocked {
    start: Pt2,
    shot_vel: Vec3,
}
struct Baddie {
    entity: Entity,
    stuck_arrows: Vec<Entity>,
    health: f32,
}
#[derive(Clone)]
struct Entity {
    pos: Pt3,
    vel: Vec3,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum PullLevel {
    TooLittle,
    Low,
    Med,
    High,
    Max,
    TooMuch,
}
/////////////////// IMPL IMPL IMPL IMPL IMPL

impl DoodadKind {
    fn rect(&self) -> Rect {
        let x = match self {
            Self::Bush => 0.0,
            Self::Rock => 0.2,
            Self::Pebbles => 0.4,
            Self::Shrub => 0.6,
        };
        Rect { x, y: 0., w: 0.2, h: 0.2 }
    }
}
impl Entity {
    fn update(&mut self) {
        let nsq = self.vel.norm_squared().sqr() + 1.;
        self.vel += Pt3::new(0., 0., 0.5).coords; // gravity
        self.vel *= nsq.powf(0.996) / nsq;
        self.pos += self.vel;
    }
}
impl Default for Camera {
    fn default() -> Self {
        Self { world_pos: [0.; 3].into(), world_rot: 0. }
    }
}
impl Camera {
    fn vec_3_to_2(&self, v: Vec3) -> Vec2 {
        let v: Vec3 = Rotation3::from_axis_angle(&Vec3::z_axis(), self.world_rot) * v;
        [v[0], v[1] / ROOT_OF_2 + v[2] / ROOT_OF_2].into()
    }
    fn vec_2_to_3(&self, v: Vec2) -> Vec3 {
        let v: Vec3 = [v[0], v[1] * ROOT_OF_2, 0.].into();
        Rotation3::from_axis_angle(&Vec3::z_axis(), -self.world_rot) * v
    }
    fn pt_2_to_3(&self, p: Pt2) -> Pt3 {
        // 1. extrapolate [x,y] into flat [x,y*sqrt(2),0]
        let p: Pt3 = [p[0], p[1] * ROOT_OF_2, 0.].into();
        let m = {
            let mut m = Transform3::identity();

            // 4. translate relative to camera origin
            m *= Translation3::from(self.world_pos.coords);
            // 3. rotate about the z axis
            m *= Rotation3::from_axis_angle(&Vec3::z_axis(), -self.world_rot);
            // 2. translate from center to tl corner
            m *= Translation3::from(-Vec3::from(SCREEN_TRANS_V3));
            m
        };
        m * p
    }
    fn pt_3_to_2(&self, p: Pt3) -> Pt2 {
        let m = {
            let mut m = Transform3::identity();
            // 3. translate from tl corner to center
            m *= Translation3::from(Vec3::from(SCREEN_TRANS_V3));
            // 2. rotate about the z axis
            m *= Rotation3::from_axis_angle(&Vec3::z_axis(), self.world_rot);
            // 1. translate relative to camera origin
            m *= Translation3::from(-self.world_pos.coords);
            m
        };
        let p: Pt3 = m * p;

        // 4. fuse y and z axes
        [p[0], p[1] / ROOT_OF_2 + p[2] / ROOT_OF_2].into()
    }
    fn rot_of_xy(v: Vec2) -> f32 {
        Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &v).angle()
    }
    fn vel_to_rot_len(&self, mut vel_xyz: Vec3) -> [f32; 2] {
        vel_xyz = vel_xyz.normalize();
        let xy = self.vec_3_to_2(vel_xyz);
        [Self::rot_of_xy(xy), xy.norm()]
    }
    fn vel_to_rot_len_shadow(&self, mut vel_xyz: Vec3) -> [f32; 2] {
        vel_xyz = vel_xyz.normalize();
        vel_xyz[2] = 0.;
        let xy = self.vec_3_to_2(vel_xyz);
        [Self::rot_of_xy(xy), xy.norm()]
    }
}

impl PullLevel {
    fn can_shoot(self) -> bool {
        match self {
            Self::TooLittle | Self::TooMuch => false,
            _ => true,
        }
    }
    fn from_shot_vel(shot_vel: Vec3) -> Self {
        const DIFF: f32 = MAX_VEL - MIN_VEL;
        use PullLevel::*;
        match shot_vel.norm() {
            x if x < MIN_VEL => TooLittle,
            x if x < (MIN_VEL + DIFF * 0.25) => Low,
            x if x < (MIN_VEL + DIFF * 0.50) => Med,
            x if x < (MIN_VEL + DIFF * 0.75) => High,
            x if x < (MIN_VEL + DIFF * 1.00) => Max,
            _ => TooMuch,
        }
    }
}

impl MyGame {
    fn body_facing(&self) -> bool {
        self.nocked
            .as_ref()
            .map(|nocked| self.camera.vec_3_to_2(nocked.shot_vel)[0] > 0.)
            .or(self.pressing.right)
            .unwrap_or(self.faced_right_most_recently)
    }
    fn new(ctx: &mut Context) -> Self {
        let mut rng = rand::rngs::SmallRng::from_seed([2; 16]);
        let dude = Entity { pos: [0.; 3].into(), vel: [0.; 3].into() };
        MyGame {
            faced_right_most_recently: true,
            baddies: (0..3)
                .map(|_| Baddie {
                    stuck_arrows: vec![],
                    health: 1.0,
                    entity: Entity {
                        pos: Self::rand_baddie_spot(&mut rng, dude.pos),
                        vel: [0.; 3].into(),
                    },
                })
                .collect(),
            camera: Camera::default(),
            aim_assist: 0,
            ticks: 0,
            dude,
            last_mouse_at: [0.; 2].into(),
            rclick_anchor: None,
            pressing: Pressing::default(),
            arrows: vec![],
            stuck_arrows: vec![],
            nocked: None,
            assets: Assets::new(ctx),
            doodads: starting_doodads(&mut rng),
            rng,
        }
    }
    fn recalculate_dude_vel(&mut self) {
        let mut speed = WALK_SPEED;

        // speed decreased if aiming
        if self.nocked.is_some() {
            speed *= 0.7;
            let facing = self.body_facing();
            if let Some(right) = self.pressing.right {
                // speed decreased if walking backwards
                if facing != right {
                    speed *= 0.5;
                }
            }
        }
        // h and v speed decreased if moving horizontally & vertically
        if self.pressing.down.is_some() && self.pressing.right.is_some() {
            speed /= ROOT_OF_2
        };

        let vel: Vec2 = [
            match self.pressing.right {
                Some(true) => speed,
                Some(false) => -speed,
                _ => 0.,
            },
            match self.pressing.down {
                Some(true) => speed,
                Some(false) => -speed,
                _ => 0.,
            },
        ]
        .into();
        self.dude.vel = self.camera.vec_2_to_3(vel);
    }
    fn recalculate_rotation_wrt(&mut self, wrt: Pt2) {
        if let Some(RclickAnchor { anchor_angle, anchor_pt }) = &mut self.rclick_anchor {
            let diff_is: Pt2 = wrt - self.camera.pt_3_to_2(self.dude.pos).coords;
            let angle_is = Camera::rot_of_xy(diff_is.coords);

            // ggez::input::mouse::set_position(ctx, anchor_pt).unwrap();
            let axisangle = na::Vector3::z() * (angle_is - *anchor_angle);
            let rot = Rotation3::new(axisangle);
            let origin = self.dude.pos.coords;
            let pos_recalc = move |pos| (rot * (pos - origin)) + origin;
            for a in self.arrows.iter_mut().chain(self.stuck_arrows.iter_mut()) {
                a.pos = pos_recalc(a.pos);
                a.vel = rot * a.vel;
            }
            for d in self
                .doodads
                .iter_mut()
                .map(|d| &mut d.pos)
                .chain(self.baddies.iter_mut().map(|d| &mut d.entity.pos))
            {
                *d = pos_recalc(*d);
            }
            *anchor_pt = wrt;
            *anchor_angle = angle_is;
        }
    }

    fn rand_baddie_spot(rng: &mut SmallRng, dude_pos: Pt3) -> Pt3 {
        let mut p = dude_pos;
        while (p - dude_pos).norm() < 500. {
            let offset = Vec3::new(rng.gen_range(-800., 800.), rng.gen_range(-800., 800.), 0.);
            p = dude_pos + offset;
        }
        p
    }
}
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
            KeyCode::Q => self.pressing.clockwise = Some(false),
            //
            KeyCode::S => self.pressing.down = Some(true),
            KeyCode::D => self.pressing.right = Some(true),
            KeyCode::E => self.pressing.clockwise = Some(true),
            KeyCode::Escape => ggez::event::quit(ctx),
            KeyCode::Space => self.aim_assist = (self.aim_assist + 1) % 3,
            _ => return,
        }
        self.recalculate_dude_vel();
    }
    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods) {
        match keycode {
            KeyCode::W if self.pressing.down == Some(false) => self.pressing.down = None,
            KeyCode::A if self.pressing.right == Some(false) => self.pressing.right = None,
            KeyCode::Q if self.pressing.clockwise == Some(false) => self.pressing.clockwise = None,
            //
            KeyCode::S if self.pressing.down == Some(true) => self.pressing.down = None,
            KeyCode::D if self.pressing.right == Some(true) => self.pressing.right = None,
            KeyCode::E if self.pressing.clockwise == Some(true) => self.pressing.clockwise = None,
            _ => return,
        }
        self.recalculate_dude_vel();
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        let mut draining = Draining::new(&mut self.arrows);
        for b in &mut self.baddies {
            b.entity.pos += b.entity.vel;
        }
        self.dude.pos += self.dude.vel;
        self.camera.world_rot += match self.pressing.clockwise {
            None => 0.,
            Some(true) => -0.05,
            Some(false) => 0.05,
        };
        self.camera.world_pos = self.dude.pos;
        'entry_loop: while let Some(mut entry) = draining.next() {
            let arrow: &mut Entity = entry.get_mut();
            arrow.update();
            let mut stuck = arrow.pos[2] >= 0.;
            for b in &mut self.baddies {
                let mut baddie_dist = arrow.pos.coords - b.entity.pos.coords;
                baddie_dist[2] *= 0.25;
                if baddie_dist.norm() < 32. {
                    b.entity.vel = [0.; 3].into();
                    b.health += arrow.pos[2] * 0.01;
                    let mut arrow = entry.take();
                    arrow.pos -= b.entity.pos.coords;
                    if b.health <= 0. {
                        // "killed"
                        b.health = 1.;
                        b.entity.pos = Self::rand_baddie_spot(&mut self.rng, self.dude.pos);
                        b.stuck_arrows.clear();
                    } else {
                        b.stuck_arrows.push(arrow);
                    }
                    self.assets.audio.thud[0].play().unwrap();
                    continue 'entry_loop;
                }
            }
            for d in self.doodads.iter() {
                if stuck {
                    break;
                }
                let mut diff = d.pos - arrow.pos.coords;
                diff[2] *= 0.5;
                if diff.coords.norm_squared() < 400. {
                    stuck = true;
                }
            }
            if stuck {
                let index = self.ticks % 3;
                self.assets.audio.twang[index].play().unwrap();
                arrow.pos[2] = arrow.pos[2].min(0.);
                self.stuck_arrows.push(entry.take())
            }
        }
        match self.rclick_anchor {
            Some(RclickAnchor { anchor_pt, .. })
                if self.pressing.down.is_some() || self.pressing.right.is_some() =>
            {
                self.recalculate_rotation_wrt(anchor_pt);
            }
            _ => {}
        }
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % 32 == 0 {
            // recompute baddie trajectory
            for b in &mut self.baddies {
                b.entity.vel = (self.dude.pos - b.entity.pos).normalize() * BADDIE_SPEED;
            }
        }
        Ok(())
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        match button {
            MouseButton::Left => {
                let start = [x, y].into();
                self.nocked = Some(Nocked { start, shot_vel: [0.; 3].into() });
            }
            MouseButton::Right => {
                let anchor_pt: Pt2 = [x, y].into();
                let anchor_diff: Pt2 = anchor_pt - self.camera.pt_3_to_2(self.dude.pos).coords;
                self.rclick_anchor = Some(RclickAnchor {
                    anchor_pt,
                    anchor_angle: Camera::rot_of_xy(anchor_diff.coords),
                })
            }
            _ => {}
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, _x: f32, _y: f32) {
        match button {
            MouseButton::Left => {
                if let Some(arrow) =
                    self.nocked.take().and_then(|nocked| match nocked.shot_vel.norm_squared() {
                        x if x < MIN_VEL.sqr() => None,
                        x if x > MAX_VEL.sqr() => None,
                        _ => Some(Entity {
                            pos: self.dude.pos + Vec3::from(TO_ARMS),
                            vel: nocked.shot_vel,
                        }),
                    })
                {
                    self.assets.audio.loose[0].play().unwrap();
                    for t in &mut self.assets.audio.taut {
                        t.stop();
                    }
                    self.arrows.push(arrow);
                }
            }
            MouseButton::Right => self.rclick_anchor = None,
            _ => {}
        }
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        let mouse_at: Pt2 = [x, y].into();
        if self.last_mouse_at == mouse_at {
            return;
        }
        self.last_mouse_at = mouse_at;
        if let Some(Nocked { start, shot_vel }) = &mut self.nocked {
            let line_v = start.coords - mouse_at.coords;
            let pull_v = line_v * 0.11;
            let duderel = self.camera.pt_3_to_2(self.dude.pos).coords - mouse_at.coords;
            let proj_scalar = pull_v.dot(&duderel) / pull_v.dot(&pull_v);
            let proj_v = pull_v * proj_scalar;
            let perp_v = proj_v - duderel;
            let z = -(perp_v.norm() + 20.) * 0.07;
            *shot_vel = self.camera.vec_2_to_3(pull_v);
            shot_vel[2] = z;

            // let new_pull_level = match diff.coords.norm_squared() {
            //     x if x < 30.0f32.sqr() => PullLevel::None,
            //     x if x < 60.0f32.sqr() => PullLevel::Low,
            //     x if x < 110.0f32.sqr() => PullLevel::Med,
            //     x if x < 180.0f32.sqr() => PullLevel::High,
            //     _ => PullLevel::Max,
            // };
            // if *pull_level != new_pull_level {
            //     self.assets.audio[*pull_level].stop();
            //     self.assets.audio[new_pull_level].play().unwrap();
            //     *pull_level = new_pull_level;
            // }
        }
        self.recalculate_rotation_wrt(mouse_at);
    }

    ///////////////////////////////////////////////DRAW DRAW DRAW DRAW DRAW DRAW

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, GREEN);
        // the dude
        const DUDE_SCALE: [f32; 2] = [2.; 2];
        const ARROW_SCALE: f32 = 1.6;
        let right_facing = self.body_facing();
        let facing_dude_scale = [
            //
            DUDE_SCALE[0] * if right_facing { 1. } else { -1. },
            DUDE_SCALE[1],
        ];
        let dude_arm_pos = self.camera.pt_3_to_2(self.dude.pos + Vec3::new(0., 0., ARM_Z));
        let dude_feet_pos = self.camera.pt_3_to_2(self.dude.pos);
        let dude_param = DrawParam {
            src: Rect { x: 0., y: 0., h: 0.2, w: 0.2 },
            dest: dude_arm_pos.into(),
            color: WHITE,
            scale: facing_dude_scale.into(),
            offset: [0.5, 0.5].into(),
            ..Default::default()
        };
        fn flatten(p: Pt3) -> Pt3 {
            [p[0], p[1], 0.].into()
        }
        for doodad in self.doodads.iter() {
            self.assets.tex.doodads.add(DrawParam {
                src: doodad.kind.rect(),
                dest: self.camera.pt_3_to_2(doodad.pos).into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            });
        }
        let arrow_draw =
            |y: f32, arrow: &Entity, arrow_batch: &mut SpriteBatch, camera: &Camera| {
                let src = Rect { x: 0., y, w: 1., h: 0.25 };
                let dest = camera.pt_3_to_2(flatten(arrow.pos));
                let [rotation, len] = camera.vel_to_rot_len_shadow(arrow.vel);
                let p = DrawParam {
                    src,
                    dest: dest.into(),
                    scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                    rotation,
                    color: BLACK,
                    offset: [0.95, 0.5].into(),
                    ..Default::default()
                };
                //real
                arrow_batch.add(p);
                let dest = camera.pt_3_to_2(arrow.pos);
                let [rotation, len] = camera.vel_to_rot_len(arrow.vel);
                let p = DrawParam {
                    src,
                    dest: dest.into(),
                    scale: [len * ARROW_SCALE, ARROW_SCALE].into(),
                    rotation,
                    offset: [0.95, 0.5].into(),
                    ..Default::default()
                };
                arrow_batch.add(p);
            };
        // baddie
        for b in &self.baddies {
            self.assets.tex.doodads.add(DrawParam {
                src: DoodadKind::Rock.rect(),
                dest: self.camera.pt_3_to_2(b.entity.pos).into(),
                color: Color { r: 1.0, g: b.health, b: b.health, a: 1. },
                scale: [3.1, 5.0].into(),
                offset: [0.5, 0.75].into(),
                ..Default::default()
            });
            for arrow in &b.stuck_arrows {
                let mut a: Entity = Entity::clone(arrow);
                a.pos += b.entity.pos.coords;
                arrow_draw(0.25, &a, &mut self.assets.tex.arrow_batch, &self.camera);
            }
        }

        for arrow in self.arrows.iter() {
            arrow_draw(0.00, arrow, &mut self.assets.tex.arrow_batch, &self.camera);
        }
        for arrow in self.stuck_arrows.iter() {
            arrow_draw(0.25, arrow, &mut self.assets.tex.arrow_batch, &self.camera);
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
        match self.nocked.as_ref().map(|n| (n, PullLevel::from_shot_vel(n.shot_vel))) {
            Some((nocked, pull_level)) => {
                let line_v = end.coords - nocked.start.coords;
                let dif_rotation = Rotation2::rotation_between(&Pt2::new(1., 0.).coords, &line_v);
                let dif_rotation_angle = dif_rotation.angle();

                let [aim_angle, arm_angle] = if pull_level.can_shoot() {
                    let aim_angle = Camera::rot_of_xy(self.camera.vec_3_to_2(nocked.shot_vel));
                    [aim_angle, aim_angle + if right_facing { 0. } else { PI }]
                } else {
                    [0.; 2]
                };
                const NOCK_0_ANGLE: f32 = 0.45;
                let nock_angle = match (right_facing, pull_level.can_shoot()) {
                    (true, false) => NOCK_0_ANGLE,
                    (false, false) => PI - NOCK_0_ANGLE,
                    _ => aim_angle,
                };

                // draw the dude
                let arm_src = {
                    use PullLevel::*;
                    let x = match pull_level {
                        TooLittle | TooMuch => 0.,
                        Low => 0.2,
                        Med => 0.4,
                        High => 0.6,
                        Max => 0.8,
                    };
                    Rect { x, y: 0., h: 0.2, w: 0.2 }
                };
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
                let nock_at = self.dude.pos + Pt3::new(0., 0., ARM_Z).coords;
                let p = DrawParam {
                    src: Rect { x: 0., y: 0., h: 0.25, w: 1. },
                    dest: self.camera.pt_3_to_2(nock_at).into(),
                    color: WHITE,
                    scale: [ARROW_SCALE; 2].into(),
                    offset: [0., 0.5].into(),
                    rotation: nock_angle,
                    ..Default::default()
                };
                self.assets.tex.arrow_batch.add(p);

                let n = line_v.norm();
                let (color, linelen) = match pull_level.can_shoot() {
                    false => (RED, n),
                    true => {
                        if self.aim_assist >= 1 {
                            let aim_e_v3 =
                                self.dude.pos + Vec3::from(TO_ARMS) + nocked.shot_vel * 7.;
                            let aim_e_v2 = self.camera.pt_3_to_2(aim_e_v3);
                            let rel_v2 = aim_e_v2 - Vec2::from(SCREEN_TRANS_V2);
                            graphics::draw(
                                ctx,
                                &self.assets.tex.unit_line,
                                DrawParam {
                                    scale: [rel_v2.coords.norm(), 1.].into(),
                                    rotation: Camera::rot_of_xy(rel_v2.coords),
                                    offset: [0., 0.5].into(),
                                    color: BLUE,
                                    ..dude_param
                                },
                            )?;
                            let aim_e_v3 = self.dude.pos
                                + Vec3::new(nocked.shot_vel[0], nocked.shot_vel[1], 0.) * 7.;
                            let aim_e_v2 = self.camera.pt_3_to_2(aim_e_v3);
                            let rel_v2 = aim_e_v2 - Vec2::from(SCREEN_TRANS_V2);
                            graphics::draw(
                                ctx,
                                &self.assets.tex.unit_line,
                                DrawParam {
                                    dest: dude_feet_pos.into(),
                                    scale: [rel_v2.coords.norm(), 1.].into(),
                                    rotation: Camera::rot_of_xy(rel_v2.coords),
                                    offset: [0., 0.5].into(),
                                    color: BLACK,
                                    ..dude_param
                                },
                            )?;
                            if self.aim_assist >= 2 {
                                let mut arrow = Entity {
                                    vel: nocked.shot_vel,
                                    pos: self.dude.pos - Vec3::new(0., 0., 32.),
                                };
                                while arrow.pos[2] < 0. {
                                    arrow.update();
                                }
                                arrow.pos[2] = 0.;
                                graphics::draw(
                                    ctx,
                                    &self.assets.tex.cross,
                                    DrawParam {
                                        dest: self.camera.pt_3_to_2(arrow.pos).into(),
                                        scale: [8., 8. / ROOT_OF_2].into(),
                                        ..Default::default()
                                    },
                                )?;
                            }
                        }
                        (WHITE, n)
                    }
                };
                graphics::draw(
                    ctx,
                    &self.assets.tex.unit_line,
                    DrawParam {
                        color,
                        dest: nocked.start.into(),
                        rotation: dif_rotation_angle,
                        scale: [linelen, 1.].into(),
                        ..Default::default()
                    },
                )?;
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
        graphics::draw(ctx, &self.assets.tex.doodads, DrawParam::default())?;
        self.assets.tex.arrow_batch.clear();
        self.assets.tex.doodads.clear();
        graphics::present(ctx)
    }
}
