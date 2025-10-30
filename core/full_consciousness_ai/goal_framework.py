"""
Goal and Intention Framework

Framework for conscious goal-setting and intention tracking.
"""

from typing import List, Optional
from collections import deque, defaultdict
from datetime import datetime

from .data_models import ConscientGoal


class GoalIntentionFramework:
    """Framework for conscious goal-setting and intention tracking"""

    def __init__(self):
        self.active_goals = {}
        self.completed_goals = deque(maxlen=1000)
        self.intention_hierarchy = defaultdict(list)

    def create_conscious_goal(self,
                            description: str,
                            priority: float = 0.5,
                            emotional_investment: float = 0.5,
                            expected_completion: Optional[datetime] = None) -> ConscientGoal:
        """Create a new conscious goal with intentions"""
        goal = ConscientGoal(
            description=description,
            priority=priority,
            emotional_investment=emotional_investment,
            expected_completion=expected_completion
        )

        self.active_goals[goal.goal_id] = goal

        # Generate subgoals if complex
        if len(description.split()) > 10:  # Complex goal
            subgoals = self._generate_subgoals(description)
            goal.subgoals = subgoals

        return goal

    def update_goal_progress(self, goal_id: str, progress: float, reflection: str = ""):
        """Update progress on a goal with conscious reflection"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            old_progress = goal.progress
            goal.progress = min(1.0, max(0.0, progress))

            if reflection:
                goal.reflection_notes.append(f"Progress {old_progress:.2f} → {progress:.2f}: {reflection}")

            # Complete goal if finished
            if goal.progress >= 1.0:
                self._complete_goal(goal_id)

    def _generate_subgoals(self, main_goal: str) -> List[str]:
        """Generate subgoals for complex goals"""
        # Simplified subgoal generation
        subgoals = []
        if "learn" in main_goal.lower():
            subgoals.extend(["Gather resources", "Study fundamentals", "Practice application", "Evaluate understanding"])
        elif "create" in main_goal.lower():
            subgoals.extend(["Plan structure", "Develop prototype", "Refine and improve", "Test and validate"])
        elif "understand" in main_goal.lower():
            subgoals.extend(["Research background", "Analyze components", "Synthesize insights", "Apply knowledge"])
        else:
            subgoals.extend(["Define approach", "Execute plan", "Review results"])

        return subgoals

    def _complete_goal(self, goal_id: str):
        """Complete a goal and move to completed goals"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            self.completed_goals.append(goal)
            del self.active_goals[goal_id]


__all__ = ['GoalIntentionFramework']
