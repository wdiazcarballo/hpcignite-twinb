from mesa import Agent   
import random
import torch

class BaseAgent(Agent):
    def __init__(self, unique_id, model, zones,
                 preferred_temp=25.0,
                 comfort_tolerance=1.0,
                 initial_temp=None,
                 gender=None,
                 age=None,
                 route=None,
                 max_delta=3,
                 **kwargs):

        super().__init__(unique_id, model)

        self.device = model.device

        self.agent_id = f"{self.__class__.__name__.lower()}_{unique_id}"
        self.agent_type = self.__class__.__name__.replace("Agent", "").lower()
        self.current_room = random.choice(zones) if zones else "Unknown Zone"

        self.preferred_temp = preferred_temp
        self.comfort_tolerance = comfort_tolerance
        self.gender = gender
        self.age = age
        self.route = route if route is not None else []
        self.max_delta = max_delta

        # ----- Set initial temperature correctly -----
        if initial_temp is None:
            initial_temp = self.preferred_temp

        self.current_temp = torch.tensor(
            float(initial_temp),
            dtype=torch.float32,
            device=self.device
        )

        self.using_ac = False
        self.comfort_level = 0.0
        self.current_day = None
        self.current_hour = None


    def step(self):
        # temp = self.model.get_current_temp(self.current_room) or self.preferred_temp
        # self.current_temp = temp
        temp = self.model.get_current_temp(self.current_room)
        if temp is None:
            temp = float('nan')
        self.current_temp = temp

        self.comfort_level = max(0.0, self.preferred_temp - abs(temp - self.preferred_temp))
        self.using_ac = abs(temp - self.preferred_temp) > self.comfort_tolerance

        self.current_day = getattr(self.model, "current_day", None)
        self.current_hour = getattr(self.model, "current_hour", None)

        self.model.agent_results.append({
            "day": self.current_day,
            "hour": self.current_hour,
            "step": self.model.current_step,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "room": self.current_room,
            "current_temp": self.current_temp,
            "comfort_level": round(self.comfort_level, 2),
            "using_ac": self.using_ac,
            "preferred_temp": self.preferred_temp
        })

class StudentAgent(BaseAgent): pass
class StaffAgent(BaseAgent): pass
class CleanerAgent(BaseAgent): pass
class WardenAgent(BaseAgent): pass
class VisitorAgent(BaseAgent): pass
class PolicyAgent(BaseAgent): pass
# from mesa import Agent   
# import random

# class BaseAgent(Agent):
#     def __init__(self, unique_id, model, zones, **kwargs):
#         super().__init__(unique_id, model)
#         self.agent_id = f"{self.__class__.__name__.lower()}_{unique_id}"
#         self.agent_type = self.__class__.__name__.replace("Agent", "").lower()
#         self.current_room = random.choice(zones) if zones else "Unknown Zone"
#         self.preferred_temp = kwargs.get("preferred_temp", 25.0)
#         self.comfort_tolerance = kwargs.get("comfort_tolerance", 1.0)
#         self.gender = kwargs.get("gender", None)
#         self.age = kwargs.get("age", None)
#         self.route = kwargs.get("route", [])
#         self.max_delta = kwargs.get("max_delta", 3)
#         self.using_ac = False
#         self.comfort_level = 0.0
#         self.current_temp = None
#         self.current_day = None
#         self.current_hour = None

#     def step(self):
#         # temp = self.model.get_current_temp(self.current_room) or self.preferred_temp
#         # self.current_temp = temp
#         temp = self.model.get_current_temp(self.current_room)
#         if temp is None:
#             temp = float('nan')
#         self.current_temp = temp

#         self.comfort_level = max(0.0, self.preferred_temp - abs(temp - self.preferred_temp))
#         self.using_ac = abs(temp - self.preferred_temp) > self.comfort_tolerance

#         self.current_day = getattr(self.model, "current_day", None)
#         self.current_hour = getattr(self.model, "current_hour", None)

#         self.model.agent_results.append({
#             "day": self.current_day,
#             "hour": self.current_hour,
#             "step": self.model.current_step,
#             "agent_id": self.agent_id,
#             "agent_type": self.agent_type,
#             "room": self.current_room,
#             "current_temp": self.current_temp,
#             "comfort_level": round(self.comfort_level, 2),
#             "using_ac": self.using_ac,
#             "preferred_temp": self.preferred_temp
#         })

# class StudentAgent(BaseAgent): pass
# class StaffAgent(BaseAgent): pass
# class CleanerAgent(BaseAgent): pass
# class WardenAgent(BaseAgent): pass
# class VisitorAgent(BaseAgent): pass
# class PolicyAgent(BaseAgent): pass