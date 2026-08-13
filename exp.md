# Rover Control Console Layout

The rover control console follows a supervisory human-robot interaction model. The operator primarily monitors the rover while it autonomously interprets and executes missions, with human intervention becoming necessary only when the rover's autonomy reaches a state requiring assistance.

The interface is divided into four persistent regions:

┌─────────────────────────────────────────────────────────────┐
│ TOP STATUS / CURRENT STATUS │
├──────────────────────────────────────┬──────────────────────┤
│ │ MISSION │
│ │ │
│ │ Goal list │
│ CAMERA │ Current goal │
│ │ Goal distance │
│ │ Goal belief │
│ │ Goal status │
│ ├──────────────────────┤
│ │ ROVER STATUS │
│ │ Mode │
│ │ Current uncertainty │
│ │ Home uncertainty │
│ │ Telemetry │
├──────────────────────────────────────┴──────────────────────┤
│ TRAJECTORY / BELIEF │
├─────────────────────────────────────────────────────────────┤
│ ■ EMERGENCY STOP │
└─────────────────────────────────────────────────────────────┘

## 1. Top Status / Current Status

The top bar provides a persistent indication of the rover's current operational state. It communicates what the rover is currently doing and whether autonomous operation is proceeding normally or requires human attention.

Typical states may include:

- IDLE
- PARSING MISSION
- SEARCHING
- NAVIGATING
- AWAITING HUMAN
- RECOVERING
- GOAL REACHED
- MISSION COMPLETE
- MANUAL CONTROL

The status should be immediately visible without requiring the operator to inspect the camera, mission, or telemetry panels.

For example:

AUTONOMOUS
Navigating to ANTENNA

or:

HUMAN ASSISTANCE REQUIRED
Goal localization uncertainty exceeded threshold

## 2. Camera View

The camera occupies the largest portion of the interface and serves as the primary visual representation of the rover's environment.

The camera provides:

- Current rover viewpoint
- Environmental and terrain context
- Current target visibility
- Obstacles and relevant scene elements
- Visual confirmation of detected goals
- Visual context during human intervention

The camera is intended to answer:

"What does the rover currently see?"

The default camera view should provide sufficient environmental context for the operator to understand the rover's surroundings and heading while keeping relevant goal objects at a recognizable scale. Zoom should be available as a secondary interaction rather than being required during normal operation.

The camera may also support point-and-select interaction when manually designating a target.

## 3. Right Sidebar: Mission

The Mission panel communicates the rover's current task and progress through the goal sequence generated from the operator's natural-language command.

### Goal List

The VLM converts the natural-language mission into an ordered sequence of goals.

Example:

✓ 1. ANTENNA
→ 2. HABITAT
○ 3. FLAG
○ 4. HOME

Completed goals, the active goal, and upcoming goals should be visually distinguishable.

### Current Goal

The currently active goal is explicitly displayed.

Example:

CURRENT GOAL
HABITAT

### Goal Distance

The estimated distance between the rover and the current goal is displayed.

Example:

DISTANCE
12.4 m

This provides the operator with a simple indication of navigation progress without requiring interpretation of the trajectory or raw rover pose.

### Goal Belief

The interface communicates the rover's current belief regarding the location of the goal.

The belief consists of an estimated goal position and associated uncertainty. The operator should be given a human-readable representation of the belief rather than being required to interpret raw covariance values.

For example:

GOAL BELIEF
HIGH CONFIDENCE

or:

GOAL BELIEF
LOW CONFIDENCE

The corresponding spatial belief and uncertainty are represented in the bottom trajectory/belief panel.

### Goal Status

The current goal's execution state is displayed explicitly.

Possible states include:

- SEARCHING
- DETECTED
- NAVIGATING
- UNCERTAIN
- REACQUIRING
- REACHED

The Mission panel therefore answers:

"What is the rover trying to accomplish, how far away is the goal, and how confident is it about that goal?"

## 4. Right Sidebar: Rover Status

The Rover Status panel communicates the rover's current operational and navigation state.

### Mode

The current control mode is displayed explicitly.

Examples:

AUTONOMOUS
HUMAN GUIDANCE
MANUAL
IDLE

The operator should never need to infer whether the rover is under autonomous or human control.

### Current Uncertainty

The current localization uncertainty associated with the active goal is displayed.

Example:

CURRENT UNCERTAINTY
0.42

The value may additionally be represented qualitatively:

LOW
MEDIUM
HIGH

The uncertainty threshold that triggers human intervention should be represented clearly when relevant.

### Home Uncertainty

The rover's uncertainty regarding its estimated home/start position is displayed separately from the uncertainty of the current goal.

Example:

HOME UNCERTAINTY
LOW

This distinction allows the operator to understand whether uncertainty concerns the active task or the rover's ability to reliably return to its starting location.

### Telemetry

Only useful operational telemetry should be displayed in the primary interface.

Potential values include:

- Rover position
- Heading
- Velocity
- Yaw rate
- Navigation state
- Other relevant runtime information

Low-level implementation details such as model inference information, internal controller flags, or debugging values should remain in a separate diagnostics view rather than occupying the primary operational interface.

The Rover Status panel therefore answers:

"How is the rover operating, what mode is it in, and how certain is its localization?"

## 5. Bottom Bar: Trajectory / Belief

The bottom panel provides persistent spatial information about the rover's recent movement and its estimated goal location.

It contains:

### Trajectory

The recent rover trajectory is displayed to provide spatial context for the rover's movement and progress toward the current goal.

### Belief

The estimated location of the current goal is represented together with its uncertainty.

The belief representation should allow the operator to visually distinguish between:

- High-confidence localization, where the estimated goal position is relatively well constrained.
- Increasing uncertainty, where the possible goal location becomes more spatially dispersed.
- High uncertainty, where the rover may no longer be sufficiently confident to continue autonomous goal navigation.

This panel therefore answers:

"Where has the rover been, and where does it currently believe the goal is?"

## 6. Emergency Stop

The emergency stop is placed in a persistent bottom bar outside the main content area so that it remains accessible regardless of the current interface state.

It must remain available when:

- The sidebar is scrolled
- A mission is executing
- A contextual interaction is active
- The rover is navigating autonomously
- The rover is receiving human guidance
- The rover is operating in manual mode

The emergency stop is intentionally given the greatest visual prominence among the controls.

Example:

■ EMERGENCY STOP

The emergency stop immediately halts rover movement and provides the operator with a direct safety mechanism independent of the current navigation or mission state.

## Overall Information Hierarchy

The interface is organized around five questions that the operator should be able to answer at any point:

1. What is the rover doing?
   → Top Status / Current Status

2. What is the rover trying to accomplish?
   → Mission / Goal List / Current Goal

3. How far is the rover from its goal and how confident is it?
   → Goal Distance / Goal Belief / Goal Status

4. How is the rover operating?
   → Rover Status / Mode / Uncertainty / Telemetry

5. Where has the rover been and where does it believe the goal is?
   → Trajectory / Belief

6. How can I stop the rover immediately?
   → Emergency Stop

The interface therefore prioritizes mission intent, rover state, visual perception, spatial belief, and safety while keeping low-level implementation details out of the primary operator workflow. The rover remains autonomous during normal operation, while the UI provides the operator with sufficient situational awareness to recognize degradation and intervene when the rover's uncertainty requires human assistance.
