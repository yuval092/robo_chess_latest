# V9 Architecture Plan & Chess Project Integration

**Date:** 2026-04-25
**Context:** Analysis of the V8 training run and empirical evaluation of the physical gripper dimensions in relation to a standard 64x64cm chessboard.

## 1. The Cluttered Board Problem (Wrist Collisions)
**User Insight:** *If part of the robotic arm is at the same level as the board, and outside of the boundary cylinder, this might be a problem, because the arm might collide with other chess pieces.*

**Diagnosis:** You are absolutely correct, and your intuition has highlighted a fatal flaw in using the default Fetch gripper for dense chess manipulation. 
I wrote `scripts/diag_wrist_clearance.py` to measure the gripper wrist (`robot0:gripper_link`).
*   **Chess Cell Size:** 8.0 cm x 8.0 cm (on a 64cm board).
*   **Wrist Width:** 12.3 cm
*   **Wrist Depth:** 8.1 cm

**The Reality:** The Fetch robot's wrist is significantly wider than a single chess cell. If the wrist drops to the level of the board (or below the height of adjacent pieces), it will act like a bulldozer, knocking over everything in a 12.3cm radius. To safely pick up a piece on a crowded board, the bulky wrist **must stay above the tallest piece on the entire board**, while only long, slender fingers descend into the 8cm cell. 

**V9 Training Plan Implication:**
Because we proved earlier that the default fingers are too short (the wrist extends lower than the fingertips), **the XML model MUST be modified before the final chess integration.** We must add a vertical "stalk" or elongated tweezers to the robot so the wrist can stay high. 
*   **Training Constraint:** We should introduce a `WRIST_SAFE_Z` constraint in training. The agent will be heavily penalized or terminated if the wrist drops below the maximum piece height, enforcing that it relies entirely on the (soon to be lengthened) fingers to reach the board.

## 2. Re-evaluating the "Funnel" (Strict Vertical Descent)
**User Request:** *Review the reasoning behind the Funnel change.*

**Original Reasoning vs. New Insight:**
Originally, I suggested a Funnel (shrinking the 3.5cm drift limit to 0.5cm as the arm drops) because I worried the agent might grab empty air. However, the current `DESCEND` success criteria already demands a strict 3D Euclidean distance of < 2.0cm, meaning the agent mathematically *must* center itself to get the +500 reward. 

**The Real Reason We Need the Funnel:**
While the 2.0cm success criteria forces the agent to end up in the right spot, a uniform 3.5cm cylinder allows the agent to get there via a **diagonal swoop**. 
If the agent swoops down diagonally into the 8cm cell, the fingers will collide with adjacent pieces before reaching the target. 
*   **The Funnel enforces a strict, top-down vertical drop.** By mathematically narrowing the allowed horizontal drift as Z decreases, the agent is forced to perfectly center its XY coordinates high in the air, and then execute a pure, straight-down Z-axis movement. This is the only safe way to navigate a cluttered 8cm cell.

## 3. Natural and Logical Transit (The Safety Corridor)
**User Request:** *Make sure the transit changes don't already exist and aren't dangerous.*

**Verification:** I have checked the current `step()` function. The only penalties are a generic action jitter penalty and a distance gradient. There is currently no penalty preventing the arm from altering its Z-height during horizontal transit, which is why it currently executes a wasted, risky "U-shape" swoop.

**The Safest Approach:**
To ensure the arm moves horizontally without dipping (which would sweep pieces off the board), we need a **Safety Corridor Penalty**. 
*   **Implementation Plan:** We will add `reward -= 10.0 * abs(gripper_pos[2] - SAFE_Z)` only during the `TRANSIT` scenario. 
*   **Is it dangerous to training?** No. As long as the penalty scales linearly with deviation, it will smoothly guide the SAC gradient to prefer flat trajectories. It won't trigger the "Suicide Trap" because a slight dip (e.g., 1cm) only incurs a minor -0.1 penalty, which is easily offset by the +500 success bonus. 
*   **Result:** The arm will learn industrial point-to-point sweeping, staying strictly above the `WRIST_SAFE_Z` plane until it is directly over the target.
