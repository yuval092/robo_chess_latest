# RoboChess Documentation

| # | Document | Contents |
|---|---|---|
| 00 | [Overview](00_overview.md) | What the project is, runtime flow, component map |
| 01 | [Runtime Usage](01_runtime_usage.md) | Starting the game, CLI flags, model paths |
| 02 | [Architecture](02_architecture.md) | Layer diagram, threading model, file map |
| 03 | [MuJoCo Environment](03_mujoco_environment.md) | Env class hierarchy, reset, observation, soft reset |
| 04 | [Arm Control](04_arm_control.md) | SAC models, ModelEmbeddedController, grasp/place pipelines |
| 05 | [Chess Logic](05_chess_logic.md) | ChessService, MovePlanner, LogicalPieceTracker, GameOrchestrator |
| 06 | [Physical Layer](06_physical_layer.md) | PhysicalPlanExecutor, MovementExecutor, occupancy, teleport |
| 07 | [Scene & Assets](07_scene_assets.md) | Board geometry, BoardMapper, piece bodies, XML |
| 08 | [Web UI](08_web_ui.md) | Flask API routes, QueuedUIBackend, threading contract |
| 09 | [Training](09_training.md) | Specialist models, curriculum, training loop |
| 10 | [Configuration](10_configuration.md) | All config files with key reference |
| 11 | [Testing](11_testing.md) | Test suite layout, integration tests, all-square sweep |
| 12 | [Production Hardening Audit](12_production_hardening_audit.md) | Adversarial checks, graceful-failure gaps, prioritized hardening backlog |
| 13 | [Final University Mentor Readiness Review](13_final_university_mentor_readiness_review.md) | Merged robustness review, validation of second review, final fix plan |
| 14 | [Selected Hardening Changes](14_selected_hardening_changes.md) | Minimal implemented changes for fault state, startup validation, HTTP/UI errors, and sensor/action checks |
