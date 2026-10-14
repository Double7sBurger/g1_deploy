## Policy category

`ftuwther`:
mjwarp rough terrain 

`ufmtmwr9`:
soft contact flat terrain 

`m2g8sl07`
soft contact rough terrain

`evh180n4`
mpm flat terrain 

`5f1xyc21`
mpm rough terrain

## Observation

Policy observation dim is 10x(3 + 3 + 29 + 29 + 29) + 7 = 937

```python 
@configclass
class PolicyCfg(ObsGroup):
    """Observations for the actor."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        mirror=mirror_vec3,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 10


@configclass
class CommandCfg(ObsGroup):
    """Command observations for the actor and critic."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        mirror=symmetry.mirror_velocity_heading,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1
```