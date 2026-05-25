"""Headless physical executor for testing and logic-only evaluation scripts."""

from __future__ import annotations

from src.physical.plan_executor import PhysicalExecutionResult, PhysicalPlan


class NoOpPhysicalExecutor:
    """Accept all plans and report success without moving the arm."""

    def __init__(self) -> None:
        """Initialise this object."""
        self.plans: list[PhysicalPlan] = []

    def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult:
        """Accept the plan and report success for every command."""
        self.plans.append(plan)
        return PhysicalExecutionResult(
            success=True,
            command_results=[(command, True) for command in plan.commands],
        )

    def return_to_home(self) -> PhysicalExecutionResult:
        """Report success without moving the arm."""
        return PhysicalExecutionResult(success=True, command_results=[])
