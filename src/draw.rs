use super::*;

impl MyGame {
    pub(crate) fn my_draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        const DUDE_SCALE: [f32; 2] = [2.; 2];
        const ARROW_SCALE: f32 = 1.7;
        const PULL_LIMP_ANGLE: f32 = -0.4;
        const FRAME_TICKS: usize = 6;

        // reset the screen's pixels to green
        graphics::clear(ctx, GREEN);
        // compute 'x' field of texture rectangle for archers.
        // as a function of "time" (self.ticks) this results in a looping animation.
        let archer_walk_x = match self.ticks % (FRAME_TICKS * 6) {
            x if x < FRAME_TICKS * 1 => 0.0,
            x if x < FRAME_TICKS * 2 => 0.2,
            x if x < FRAME_TICKS * 3 => 0.4,
            x if x < FRAME_TICKS * 4 => 0.6,
            x if x < FRAME_TICKS * 5 => 0.4,
            ________________________ => 0.2,
        };
        for (archer_index, a) in self.archers.iter().enumerate() {
            let right_facing = self.ui.camera.archer_facing_right(a);
            // incremetally compute the DrawParam for drawing this archer's texture.
            let facing_dude_scale = [
                //
                DUDE_SCALE[0] * if right_facing { 1. } else { -1. },
                DUDE_SCALE[1],
            ];
            let dude_arm_pos = self.ui.camera.pt_3_to_2(a.entity.pos + Vec3::from(TO_ARMS));
            let dude_param = DrawParam {
                src: Rect { x: 0., y: 0., h: 0.2, w: 0.2 },
                dest: dude_arm_pos.into(),
                color: WHITE,
                scale: facing_dude_scale.into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            };
            let main_src = {
                let x = if a.entity.vel != Vec3::new(0., 0., 0.) { archer_walk_x } else { 0.4 };
                Rect { x, y: 0., h: 0.2, w: 0.2 }
            };
            let (arm_src, arm_angle, arrow_angle) = if let Some(shot_vel) = a.shot_vel {
                // this archer is drawing their bow
                let pull_level = PullLevel::from_shot_vel(shot_vel);
                // compute texture rectangle (i.e. which sprite in the sheet)
                // ALL archers' arms are drawn in according to the angle they are aiming
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
                    // "PULL_LIMP_ANGLE" is the angle you hold your bow limply (not aiming)
                    // used when player has a BAD pull (too little or too much)
                    (PULL_LIMP_ANGLE, Some(-PULL_LIMP_ANGLE))
                } else {
                    (-PULL_LIMP_ANGLE, Some(PULL_LIMP_ANGLE + PI))
                };

                if archer_index == self.ui.controlling {
                    // the player controls this archer! draw extra UI stuff
                    let lclick_state = self.ui.lclick_state.as_ref().unwrap();
                    let end: Pt2 = ggez::input::mouse::position(ctx).into();
                    let diff = end - lclick_state.start;
                    let difflen = diff.norm();
                    let diffang = Camera::rot_of_xy(diff);
                    let color = if lclick_state.last_pull_level.can_shoot() { WHITE } else { RED };
                    // draw line that player uses to "pull" the bow. White if they COULD loose an arrow
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
                        // minimal aim assist is toggled.
                        // draw the little angle and pitch line overlay
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
                            // aim assist is set to max. predict the arrow's impact location.
                            // simple solution: create an arrow entity and simulate its flight WHILE its not hit the ground.
                            let mut arrow = Entity {
                                vel: shot_vel,
                                pos: a.entity.pos - Vec3::new(0., 0., 32.),
                            };
                            while arrow.pos[2] < 0. {
                                arrow.vel = Entity::grav_and_air_resist(arrow.vel);
                                arrow.pos += arrow.vel;
                            }
                            arrow.pos[2] = 0.;
                            // ... and draw a cross on the floor where it impacted
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
                let arm_angle = if right_facing { PULL_LIMP_ANGLE } else { -PULL_LIMP_ANGLE };
                (arm_src, arm_angle, None)
            };
            // draw archer layer 1/4 (back arm)
            graphics::draw(
                ctx,
                &self.assets.tex.archer_back,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            // draw archer layer 2/4 (body)
            graphics::draw(
                ctx,
                &self.assets.tex.archer,
                DrawParam { src: main_src, ..dude_param },
            )?;
            // draw archer layer 3/4 (front arm + bow)
            graphics::draw(
                ctx,
                &self.assets.tex.archer_front,
                DrawParam { src: arm_src, rotation: arm_angle, ..dude_param },
            )?;
            if let Some(arrow_angle) = arrow_angle {
                // draw archer layer 4/4 (nocked arrow)
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
        // draw grasses rocks etc. using INSTANCED RENDERING
        for doodad in self.doodads.iter() {
            self.assets.tex.doodads.add(DrawParam {
                src: doodad.kind.rect(),
                dest: self.ui.camera.pt_3_to_2(doodad.pos).into(),
                offset: [0.5, 0.5].into(),
                ..Default::default()
            });
        }
        // define a closure for adding an arrow instance
        // to the arrow batch (for instanced rendering on GPU)
        let arrow_draw =
            |y: f32, arrow: &Entity, arrow_batch: &mut SpriteBatch, camera: &Camera| {
                // arrow's shadow on the ground
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
                // arrow in the air
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
        // draw baddies in the world
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
        // invoke the `arrow_draw` closure for all arrows in flight
        for arrow in self.arrows.iter() {
            arrow_draw(0.00, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        // invoke the `arrow_draw` closure for all arrows on thr ground
        // use a different texture rectangle (headless arrows because stuck in the target)
        for arrow in self.stuck_arrows.iter() {
            arrow_draw(0.25, arrow, &mut self.assets.tex.arrow_batch, &self.ui.camera);
        }
        // fire off instanced rendering on GPU (synchronous)
        graphics::draw(ctx, &self.assets.tex.arrow_batch, DrawParam::default())?;
        graphics::draw(ctx, &self.assets.tex.doodads, DrawParam::default())?;
        // ... and clear the batch again for use in the next frame
        self.assets.tex.arrow_batch.clear();
        self.assets.tex.doodads.clear();
        // finalize.
        graphics::present(ctx)
    }
}
