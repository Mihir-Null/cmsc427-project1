# Project 1
Written in wgpu and deployed using github actions to a demo page at https://427-p1.null-set.dev/

## Controls

- Move the mouse horizontally to rotate the camera yaw around the player.
- Move the mouse vertically to adjust the camera pitch. Pitch is clamped so the camera stays usable and does not flip (I wasn't going to use quaternions while still figuring out rust).
- `W` and `S` move the player forward and backward relative to the camera's horizontal facing direction.
- `A` and `D` strafe left and right relative to the camera's horizontal facing direction.

## Extra Feature: Third-Person Mouse Camera

This project extends the required keyboard navigation with a modern third-person control scheme.

## Implementation Notes

- The player and camera are separate positions. The player moves on the ground plane, while the camera is placed above and behind the player at a fixed follow distance.
- The camera looks at a target point slightly above the player so the visible player object remains in frame.
- The rendered player mesh follows `Camera::player_position` and rotates around the Y axis using the camera yaw, making the avatar's forward direction match the movement controls.
- Mouse movement updates camera yaw and pitch continuously, and keyboard movement is frame-rate independent through the per-frame `dt` value (fromsoft could never).

Relevant files:

- `src/camera.rs`: third-person camera state, mouse rotation, and camera-relative movement.
- `src/app.rs`: mouse motion and keyboard events.
- `src/state.rs`: visible player object placement and rotation.
