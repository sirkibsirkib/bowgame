use core::convert::TryInto;
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

mod config {
    use ggez::input::keyboard::KeyCode::{self, *};
    pub(crate) const UP: KeyCode = W;
    pub(crate) const DO: KeyCode = S;
    pub(crate) const LE: KeyCode = A;
    pub(crate) const RI: KeyCode = D;
    pub(crate) const CL: KeyCode = E;
    pub(crate) const AN: KeyCode = Q;
}

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
const ARM_Z: f32 = -42.;
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
    // let mut args = std::env::args();
    // args.next().unwrap();
    // let addr: SocketAddr = args.next().unwrap().parse().unwrap();
    // println!("addr {:?}", addr);
    // let (n, am_server) = match args.next().unwrap().as_ref() {
    //     "S" => (std::net::TcpListener::bind(addr).unwrap().accept().unwrap().0, true),
    //     "C" => (std::net::TcpStream::connect(addr).unwrap(), false),
    //     _ => panic!("NOT OK"),
    // };
    // n.set_nonblocking(true).unwrap();
    let am_server = true;
    // Make a Context.
    let (mut ctx, mut event_loop) = ContextBuilder::new("bowdraw", "Christopher Esterhuyse")
        .window_mode(WindowMode { width: WIN_DIMS[0], height: WIN_DIMS[1], ..Default::default() })
        .build()
        .unwrap();
    ggez::input::mouse::set_cursor_grabbed(&mut ctx, true).unwrap();
    let mut my_game = MyGame::new(&mut ctx, am_server);
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
struct Ui {
    camera: Camera,
    lclick_state: Option<LclickState>,
    rclick_state: Option<RclickState>,
    pressing: Pressing,
    aim_assist: u8,
    last_mouse_at: Pt2,
    controlling: usize,
    config: UiConfig,
}

struct Archer {
    entity: Entity,
    shot_vel: Option<Vec3>,
}

#[derive(serde_derive::Deserialize)]
struct UiConfigSerde {
    up: String,
    down: String,
    left: String,
    right: String,
    clockwise: String,
    anticlockwise: String,
    aim_assist: String,
    quit: String,
}
struct UiConfig {
    up: KeyCode,
    down: KeyCode,
    left: KeyCode,
    right: KeyCode,
    clockwise: KeyCode,
    anticlockwise: KeyCode,
    aim_assist: KeyCode,
    quit: KeyCode,
}
struct MyGame {
    am_server: bool,
    ticks: usize,
    baddies: Vec<Baddie>,
    archers: Vec<Archer>,
    arrows: Vec<Entity>,
    stuck_arrows: Vec<Entity>,
    assets: Assets,
    doodads: Vec<Doodad>,
    rng: SmallRng,
    ui: Ui,
}
struct RclickState {
    anchor_angle: f32,
}
struct LclickState {
    start: Pt2,
    last_pull_level: PullLevel,
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

impl Ui {
    fn recalculate_dude_vel(&self, archer: &mut Archer) {
        let vel2 = Vec2::new(
            match self.pressing.right {
                Some(true) => 1.,
                Some(false) => -1.,
                _ => 0.,
            },
            match self.pressing.down {
                Some(true) => 1.,
                Some(false) => -1.,
                _ => 0.,
            },
        );
        let vel3 = self.camera.vec_2_to_3(vel2);

        let mut speed = if self.pressing.down.is_some() && self.pressing.right.is_some() {
            WALK_SPEED / ROOT_OF_2
        } else {
            WALK_SPEED
        };
        if let Some(shot_vel) = archer.shot_vel {
            speed *= 0.6;
            let backpedaling = shot_vel.dot(&vel3) < 0.;
            if backpedaling {
                speed *= 0.4;
            }
        }
        archer.entity.vel = vel3 * speed;
    }
}
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
    fn archer_facing_right(&self, archer: &Archer) -> bool {
        archer
            .shot_vel
            .map(|v| self.vec_3_to_2(v)[0] >= 0.)
            .unwrap_or(self.vec_3_to_2(archer.entity.vel)[0] >= 0.)
    }
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
        match shot_vel.norm_squared() {
            x if x < MIN_VEL.sqr() => TooLittle,
            x if x < (MIN_VEL + DIFF * 0.25).sqr() => Low,
            x if x < (MIN_VEL + DIFF * 0.50).sqr() => Med,
            x if x < (MIN_VEL + DIFF * 0.75).sqr() => High,
            x if x < (MIN_VEL + DIFF * 1.00).sqr() => Max,
            _ => TooMuch,
        }
    }
}

impl MyGame {
    fn new(ctx: &mut Context, am_server: bool) -> Self {
        let mut rng = rand::rngs::SmallRng::from_seed([2; 16]);
        let config = {
            let s = std::fs::read_to_string("ui_config.toml").unwrap_or_else(|_| {
                panic!(
                    "Couldn't find `ui_config.toml` at current directory: {:?}",
                    std::env::current_dir().ok()
                )
            });
            let x: UiConfigSerde = toml::from_str(&s).expect("Failed to parse config toml!");
            x.try_into().expect("Failed to parse config toml!")
        };
        let archers = vec![
            Archer {
                shot_vel: None,
                entity: Entity { pos: [0., 0., 0.].into(), vel: [0.; 3].into() },
            },
            Archer {
                shot_vel: Some(Vec3::new(8., 12., -12.)),
                entity: Entity { pos: [50., 50., 0.].into(), vel: [0.; 3].into() },
            },
            Archer {
                shot_vel: Some(Vec3::new(-6., 11., -17.)),
                entity: Entity { pos: [120., 50., 0.].into(), vel: [0.; 3].into() },
            },
        ];
        let ui = Ui {
            controlling: 0,
            config,
            camera: Camera::default(),
            last_mouse_at: [0.; 2].into(),
            rclick_state: None,
            lclick_state: None,
            pressing: Default::default(),
            aim_assist: 0,
        };
        MyGame {
            baddies: (0..3)
                .map(|_| Baddie {
                    stuck_arrows: vec![],
                    health: 1.0,
                    entity: Entity {
                        pos: Self::rand_baddie_spot(&mut rng, &archers[ui.controlling], &archers),
                        vel: [0.; 3].into(),
                    },
                })
                .collect(),
            am_server,
            ticks: 0,
            archers,
            arrows: vec![],
            stuck_arrows: vec![],
            assets: Assets::new(ctx),
            doodads: starting_doodads(&mut rng),
            rng,
            ui,
        }
    }

    fn rand_baddie_spot(rng: &mut SmallRng, focus: &Archer, archers: &[Archer]) -> Pt3 {
        let mut p;
        'another: loop {
            let offset = Vec3::new(rng.gen_range(-800., 800.), rng.gen_range(-800., 800.), 0.);
            p = focus.entity.pos + offset;
            for a in archers.iter() {
                if (p - a.entity.pos).norm() < 500. {
                    continue 'another;
                }
            }
            return p;
        }
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
            x if x == self.ui.config.up => self.ui.pressing.down = Some(false),
            x if x == self.ui.config.left => self.ui.pressing.right = Some(false),
            x if x == self.ui.config.anticlockwise => self.ui.pressing.clockwise = Some(false),
            //
            x if x == self.ui.config.down => self.ui.pressing.down = Some(true),
            x if x == self.ui.config.right => self.ui.pressing.right = Some(true),
            x if x == self.ui.config.clockwise => self.ui.pressing.clockwise = Some(true),
            x if x == self.ui.config.quit => ggez::event::quit(ctx),
            x if x == self.ui.config.aim_assist => {
                self.ui.aim_assist = (self.ui.aim_assist + 1) % 3
            }
            _ => return,
        }
        self.ui.recalculate_dude_vel(&mut self.archers[self.ui.controlling]);
    }
    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods) {
        match keycode {
            x if x == self.ui.config.up && self.ui.pressing.down == Some(false) => {
                self.ui.pressing.down = None
            }
            x if x == self.ui.config.left && self.ui.pressing.right == Some(false) => {
                self.ui.pressing.right = None
            }
            x if x == self.ui.config.anticlockwise && self.ui.pressing.clockwise == Some(false) => {
                self.ui.pressing.clockwise = None
            }
            //
            x if x == self.ui.config.down && self.ui.pressing.down == Some(true) => {
                self.ui.pressing.down = None
            }
            x if x == self.ui.config.right && self.ui.pressing.right == Some(true) => {
                self.ui.pressing.right = None
            }
            x if x == self.ui.config.clockwise && self.ui.pressing.clockwise == Some(true) => {
                self.ui.pressing.clockwise = None
            }
            _ => return,
        }
        self.ui.recalculate_dude_vel(&mut self.archers[self.ui.controlling]);
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        // rotate camera
        self.ui.camera.world_rot += match self.ui.pressing.clockwise {
            None => 0.,
            Some(true) => -0.05,
            Some(false) => 0.05,
        };
        // update archers and baddies' positions
        for e in self
            .baddies
            .iter_mut()
            .map(|b| &mut b.entity)
            .chain(self.archers.iter_mut().map(|b| &mut b.entity))
        {
            e.pos += e.vel;
        }
        // update camera position
        self.ui.camera.world_pos = self.archers[self.ui.controlling].entity.pos;
        // update arrows' positions and fire collision events
        let mut draining = Draining::new(&mut self.arrows);
        'entry_loop: while let Some(mut entry) = draining.next() {
            let arrow: &mut Entity = entry.get_mut();
            arrow.update();
            if arrow.pos[2] >= 0. {
                let index = self.ticks % 3;
                self.assets.audio.twang[index].play().unwrap();
                arrow.pos[2] = arrow.pos[2].min(0.);
                self.stuck_arrows.push(entry.take());
                // stuck in the ground
                continue 'entry_loop;
            }
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
                        b.entity.pos = Self::rand_baddie_spot(
                            &mut self.rng,
                            &self.archers[self.ui.controlling],
                            &self.archers,
                        );
                        b.stuck_arrows.clear();
                    } else {
                        b.stuck_arrows.push(arrow);
                    }
                    self.assets.audio.thud[0].play().unwrap();
                    // stuck in a baddie
                    continue 'entry_loop;
                }
            }
        }
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % 32 == 0 {
            // recompute baddie trajectory
            for b in &mut self.baddies {
                b.entity.vel = (self.archers[self.ui.controlling].entity.pos - b.entity.pos)
                    .normalize()
                    * BADDIE_SPEED;
            }
        }
        Ok(())
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        match button {
            MouseButton::Left => {
                let start = [x, y].into();
                self.ui.lclick_state =
                    Some(LclickState { start, last_pull_level: PullLevel::TooLittle });
                self.archers[self.ui.controlling].shot_vel = Some(Vec3::new(0., 0., 0.));
            }
            MouseButton::Right => {
                let anchor_pt: Pt2 = [x, y].into();
                let anchor_diff: Pt2 = anchor_pt
                    - self.ui.camera.pt_3_to_2(self.archers[self.ui.controlling].entity.pos).coords;
                self.ui.rclick_state =
                    Some(RclickState { anchor_angle: Camera::rot_of_xy(anchor_diff.coords) })
            }
            _ => {}
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, _x: f32, _y: f32) {
        match button {
            MouseButton::Left => {
                if let Some(arrow) =
                    self.archers[self.ui.controlling].shot_vel.take().and_then(|shot_vel| {
                        self.ui.lclick_state = None;
                        match shot_vel.norm_squared() {
                            x if x < MIN_VEL.sqr() => None,
                            x if x > MAX_VEL.sqr() => None,
                            _ => Some(Entity {
                                pos: self.archers[self.ui.controlling].entity.pos
                                    + Vec3::from(TO_ARMS),
                                vel: shot_vel,
                            }),
                        }
                    })
                {
                    self.assets.audio.loose[0].play().unwrap();
                    for t in &mut self.assets.audio.taut {
                        t.stop();
                    }
                    self.arrows.push(arrow);
                }
            }
            MouseButton::Right => self.ui.rclick_state = None,
            _ => {}
        }
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        let mouse_at: Pt2 = [x, y].into();
        if self.ui.last_mouse_at == mouse_at {
            return;
        }
        self.ui.last_mouse_at = mouse_at;
        if let (Some(LclickState { start, last_pull_level }), (entity, Some(shot_vel))) =
            (&mut self.ui.lclick_state, {
                let Archer { entity, shot_vel } = &mut self.archers[self.ui.controlling];
                (entity as &Entity, shot_vel)
            })
        {
            let line_v = start.coords - mouse_at.coords;
            let pull_v = line_v * 0.11;
            let duderel = self.ui.camera.pt_3_to_2(entity.pos).coords - mouse_at.coords;
            let proj_scalar = pull_v.dot(&duderel) / pull_v.dot(&pull_v);
            let proj_v = pull_v * proj_scalar;
            let perp_v = proj_v - duderel;
            let z = perp_v.norm() * -0.07;
            *shot_vel = self.ui.camera.vec_2_to_3(pull_v);
            shot_vel[2] = z;

            let new_pull_level = PullLevel::from_shot_vel(*shot_vel);
            if new_pull_level != *last_pull_level {
                fn pl_to_audio_index(pl: PullLevel) -> Option<usize> {
                    Some(match pl {
                        PullLevel::TooMuch | PullLevel::TooLittle => return None,
                        PullLevel::Low => 0,
                        PullLevel::Med => 1,
                        PullLevel::High => 2,
                        PullLevel::Max => 3,
                    })
                }
                if let Some(index) = pl_to_audio_index(*last_pull_level) {
                    self.assets.audio.taut[index].stop();
                }
                if let Some(index) = pl_to_audio_index(new_pull_level) {
                    self.assets.audio.taut[index].play().unwrap();
                }
                *last_pull_level = new_pull_level;
            }
        }
        if let Some(RclickState { anchor_angle }) = &mut self.ui.rclick_state {
            let diff_is: Pt2 = mouse_at
                - self.ui.camera.pt_3_to_2(self.archers[self.ui.controlling].entity.pos).coords;
            let angle_is = Camera::rot_of_xy(diff_is.coords);
            let angle_diff = angle_is - *anchor_angle;
            self.ui.camera.world_rot += angle_diff;
            *anchor_angle = angle_is;
        }
    }

    ///////////////////////////////////////////////DRAW DRAW DRAW DRAW DRAW DRAW

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, GREEN);
        // the dude
        const DUDE_SCALE: [f32; 2] = [2.; 2];
        const ARROW_SCALE: f32 = 1.7;
        const PULL_BAD_ANGLE: f32 = -0.4;
        // const PULL_LOW_ANGLE: f32 = -0.1;

        // dbg!(self.archers[self.ui.controlling].entity.vel);
        for (archer_index, a) in self.archers.iter().enumerate() {
            let right_facing = self.ui.camera.archer_facing_right(a);
            let facing_dude_scale = [
                //
                DUDE_SCALE[0] * if right_facing { 1. } else { -1. },
                DUDE_SCALE[1],
            ];
            let dude_arm_pos = self.ui.camera.pt_3_to_2(a.entity.pos + Vec3::new(0., 0., ARM_Z));
            // let dude_feet_pos = self.ui.camera.pt_3_to_2(a.entity.pos);
            let dude_param = DrawParam {
                src: Rect { x: 0., y: 0., h: 0.2, w: 0.2 },
                dest: dude_arm_pos.into(),
                color: WHITE,
                scale: facing_dude_scale.into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            };
            let main_src = {
                let x = if a.entity.vel != Vec3::new(0., 0., 0.) {
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
            let (arm_src, arm_angle, arrow_angle) = if let Some(shot_vel) = a.shot_vel {
                // drawing the bow
                let pull_level = PullLevel::from_shot_vel(shot_vel);
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
                let (arm_angle, arrow_angle) = if pull_level.can_shoot() {
                    let x = Camera::rot_of_xy(self.ui.camera.vec_3_to_2(shot_vel));
                    (if right_facing { x } else { x + PI }, Some(x))
                } else if right_facing {
                    (PULL_BAD_ANGLE, Some(-PULL_BAD_ANGLE))
                } else {
                    (-PULL_BAD_ANGLE, Some(PULL_BAD_ANGLE + PI))
                };

                if archer_index == self.ui.controlling {
                    let lclick_state = self.ui.lclick_state.as_ref().unwrap();
                    let end: Pt2 = ggez::input::mouse::position(ctx).into();
                    let diff = end - lclick_state.start;
                    let difflen = diff.norm();
                    let diffang = Camera::rot_of_xy(diff);
                    let color = if lclick_state.last_pull_level.can_shoot() { WHITE } else { RED };
                    graphics::draw(
                        ctx,
                        &self.assets.tex.unit_line,
                        DrawParam {
                            color,
                            dest: lclick_state.start.into(),
                            rotation: diffang,
                            scale: [difflen, 1.].into(),
                            ..Default::default()
                        },
                    )?;
                    if self.ui.aim_assist >= 1 {
                        let aim_e_v3 = a.entity.pos + shot_vel * 7.;
                        let aim_e_v2 = self.ui.camera.pt_3_to_2(aim_e_v3);
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
                        let aim_e_v3 = a.entity.pos + Vec3::new(shot_vel[0], shot_vel[1], 0.) * 7.;
                        let aim_e_v2 = self.ui.camera.pt_3_to_2(aim_e_v3);
                        let rel_v2 = aim_e_v2 - Vec2::from(SCREEN_TRANS_V2);
                        let dude_feet_pos = self.ui.camera.pt_3_to_2(a.entity.pos);
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
                        if self.ui.aim_assist >= 2 {
                            let mut arrow = Entity {
                                vel: shot_vel,
                                pos: a.entity.pos - Vec3::new(0., 0., 32.),
                            };
                            while arrow.pos[2] < 0. {
                                arrow.update();
                            }
                            arrow.pos[2] = 0.;
                            graphics::draw(
                                ctx,
                                &self.assets.tex.cross,
                                DrawParam {
                                    dest: self.ui.camera.pt_3_to_2(arrow.pos).into(),
                                    scale: [8., 8. / ROOT_OF_2].into(),
                                    ..Default::default()
                                },
                            )?;
                        }
                    }
                }
                (arm_src, arm_angle, arrow_angle)
            } else {
                let arm_src = Rect { x: 0., y: 0., h: 0.2, w: 0.2 };
                let arm_angle = if right_facing { PULL_BAD_ANGLE } else { -PULL_BAD_ANGLE };
                (arm_src, arm_angle, None)
            };
            graphics::draw(
                ctx,
                &self.assets.tex.archer_back,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            graphics::draw(
                ctx,
                &self.assets.tex.archer,
                DrawParam { src: main_src, ..dude_param },
            )?;
            graphics::draw(
                ctx,
                &self.assets.tex.archer_front,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            if let Some(arrow_angle) = arrow_angle {
                let p = DrawParam {
                    dest: (dude_arm_pos + Vec2::new(0., -5.0)).into(),
                    src: Rect { x: 0., y: 0., w: 1., h: 0.25 },
                    scale: [ARROW_SCALE, ARROW_SCALE].into(),
                    rotation: arrow_angle,
                    offset: [0., 0.5].into(),
                    ..dude_param
                };
                self.assets.tex.arrow_batch.add(p);
            }
        }
        fn flatten(p: Pt3) -> Pt3 {
            [p[0], p[1], 0.].into()
        }
        // doodads
        for doodad in self.doodads.iter() {
            self.assets.tex.doodads.add(DrawParam {
                src: doodad.kind.rect(),
                dest: self.ui.camera.pt_3_to_2(doodad.pos).into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            });
        }
        let arrow_draw =
            |y: f32, arrow: &Entity, arrow_batch: &mut SpriteBatch, camera: &Camera| {
                // shadow
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
                // real
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
        // baddies
        for b in &self.baddies {
            self.assets.tex.doodads.add(DrawParam {
                src: DoodadKind::Rock.rect(),
                dest: self.ui.camera.pt_3_to_2(b.entity.pos).into(),
                color: Color { r: 1.0, g: b.health, b: b.health, a: 1. },
                scale: [3.1, 5.0].into(),
                offset: [0.5, 0.75].into(),
                ..Default::default()
            });
            for arrow in &b.stuck_arrows {
                let mut a: Entity = Entity::clone(arrow);
                a.pos += b.entity.pos.coords;
                arrow_draw(0.25, &a, &mut self.assets.tex.arrow_batch, &self.ui.camera);
            }
        }

        for arrow in self.arrows.iter() {
            arrow_draw(0.00, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        for arrow in self.stuck_arrows.iter() {
            arrow_draw(0.25, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        graphics::draw(ctx, &self.assets.tex.arrow_batch, DrawParam::default())?;
        graphics::draw(ctx, &self.assets.tex.doodads, DrawParam::default())?;
        self.assets.tex.arrow_batch.clear();
        self.assets.tex.doodads.clear();
        graphics::present(ctx)
    }
}
