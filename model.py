import torch
import sys
sys.path.append("/project/lt200291-ignite/Project_chomwong/energyplus/EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64")  # ตัวอย่าง path, ตรวจสอบจริงด้วย `module show energyplus`
from pyenergyplus.api import EnergyPlusAPI
import os
import json
import random
import pandas as pd
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from agent import StudentAgent, StaffAgent, CleanerAgent, WardenAgent, VisitorAgent, PolicyAgent
from utils import sample_value, sample_gender, sample_age

from eppy.modeleditor import IDF


class BuildingModel(Model):
    def __init__(self, config, agents_file=None, idf_path=None, ep_control=False, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.schedule = RandomActivation(self)
        self.config = config
        self.current_step = 0
        self.ep_control = ep_control
        self.zones = config["mesa"].get("zones", [])
        self.last_zone_temps = {z: None for z in self.zones}
        self.agent_results = []

        self.current_day = None
        self.current_hour = None
        # ----------------- DataCollector -----------------
        self.datacollector = DataCollector(
            model_reporters={
                f"AvgTemp_{zone.replace(' ', '_')}": (lambda m, z=zone: m.get_current_temp(z))
                for zone in self.zones
            }
        )

        # ----------------- JSON (agents) -----------------
        if agents_file and os.path.exists(agents_file):
            with open(agents_file, "r") as f:
                self.agent_config = json.load(f)
            self._create_agents_from_json()
        else:
            print("[WARN] agents_file ไม่พบหรือไม่ระบุ → จะไม่มี agent ถูกสร้าง")

        # ----------------- EnergyPlus Setup -----------------
        if self.ep_control:
            idf_zone_names = []
            if idf_path:
                try:
                    idd_path = "/project/lt200291-ignite/Project_chomwong/energyplus/EnergyPlus-25.1.0-1c11a3d85f-Linux-CentOS7.9.2009-x86_64/Energy+.idd"
                    IDF.setiddname(idd_path)
                    idf = IDF(idf_path)
                    idf_zone_names = [z.Name for z in idf.idfobjects.get('ZONE', [])]
                except Exception as e:
                    print(f"[WARN] ไม่สามารถโหลด IDF: {e}")
                    idf_zone_names = []

            self.api = EnergyPlusAPI()
            self.state = self.api.state_manager.new_state()
            self.exchange = self.api.exchange

            def normalize(name):
                return name.lower().strip().replace(" ", "_")

            norm_idf_zones = {normalize(z): z for z in idf_zone_names}
            self.zone_name_map = {z: norm_idf_zones.get(normalize(z), None) for z in self.zones}
            print("Zone map:", self.zone_name_map)

            self.zone_temp_handles = {}
            self.zone_setpoint_handles = {}

            def setup_handles_first_timestep(state):
                for zone in self.zones:
                    ep_zone = self.zone_name_map.get(zone)
                    if ep_zone is None:
                        print(f"[WARN] ไม่มี mapping ระหว่าง zone '{zone}' กับ IDF zone names")
                        continue

                    temp_handle = self.exchange.get_variable_handle(state, "Zone Air Temperature", ep_zone)
                    if temp_handle == -1:
                        print(f"[WARN] ไม่พบ variable 'Zone Air Temperature' ของ zone '{zone}' (ep zone: {ep_zone})")

                    sp_handle = self.exchange.get_actuator_handle(state, "Zone Temperature Control", "SecondarySchool ClgSetp", ep_zone)
                    if sp_handle == -1:
                        print(f"[WARN] ไม่พบ actuator 'Cooling Setpoints' ของ zone '{zone}' (ep zone: {ep_zone})")

                    self.zone_temp_handles[zone] = temp_handle
                    self.zone_setpoint_handles[zone] = sp_handle

                print("Zone temp handles:", self.zone_temp_handles)
                print("Zone setpoint handles:", self.zone_setpoint_handles)

            self.api.runtime.callback_after_predictor_after_hvac_managers(
                self.state, setup_handles_first_timestep
            )

    # ============================================================
    # agents creation
    # ============================================================
    def _create_agents_from_json(self):
        mapping = {
            "student": StudentAgent,
            "staff": StaffAgent,
            "cleaner": CleanerAgent,
            "warden": WardenAgent,
            "visitor": VisitorAgent,
            "policy": PolicyAgent
        }

        total_created = 0
        for agent_type, info in self.agent_config.get("agent_types", {}).items():
            dist = info.get("distribution", {})
            count = dist.get("count", 1)

            for i in range(count):
                preferred_temp_attr = info.get("attributes", {}).get("preferred_temp", {"distribution": "uniform", "min": 25, "max": 25})
                comfort_tolerance_attr = info.get("attributes", {}).get("comfort_tolerance", {"distribution": "uniform", "min": 1, "max": 1})

                preferred_temp = sample_value(preferred_temp_attr)
                comfort_tolerance = sample_value(comfort_tolerance_attr)

                initial_temp = preferred_temp
                
                room = random.choice(self.zones) if self.zones else "Unknown Zone"
                AgentClass = mapping.get(agent_type, StudentAgent)
                agent = AgentClass(
                    unique_id=f"{agent_type}_{i}",
                    model=self,
                    zones=self.zones,
                    current_room=room,
                    preferred_temp=preferred_temp,
                    comfort_tolerance=comfort_tolerance,
                    initial_temp=initial_temp,
                    agent_type=agent_type
                )
                self.schedule.add(agent)
                total_created += 1

        print(f"✅ โหลด agent สำเร็จทั้งหมด: {total_created} ตัว")


    # ============================================================
    #  EP-control methods
    # ============================================================
    def get_current_temp(self, zone_name):
        return self.last_zone_temps.get(zone_name)

    def compute_setpoint_requests(self):
        requests = {}
        per_zone = {}
        for agent in self.schedule.agents:
            if getattr(agent, "using_ac", False) and agent.current_room:
                per_zone.setdefault(agent.current_room, []).append(agent.preferred_temp)
        for zone, temps in per_zone.items():
            temps = [t for t in temps if t is not None]
            if temps:
                requests[zone] = min(temps)
        return requests

    def read_zone_temps_from_ep(self):
        temps = {}
        for zone in self.zones:
            handle = self.zone_temp_handles.get(zone, -1)
            if handle in [None, -1]:
                temps[zone] = None
            else:
                try:
                    temps[zone] = self.exchange.get_variable_value(self.state, handle)
                except Exception:
                    temps[zone] = None
        return temps

    def apply_setpoints_to_ep(self, setpoint_map: dict):
        for zone, val in (setpoint_map or {}).items():
            handle = self.zone_setpoint_handles.get(zone, -1)
            if handle in [None, -1]:
                continue
            try:
                self.exchange.set_actuator_value(self.state, handle, val)
            except Exception:
                continue

    # ============================================================
    #  Agent step logic
    # ============================================================
    def step_agents(self, ep_model=None):
        if ep_model is not None:
            zone_temps = ep_model.read_zone_temps_from_ep()
            for z, t in zone_temps.items():
                if t is not None:
                    self.last_zone_temps[z] = t

        for agent in self.schedule.agents:
            temp = self.last_zone_temps.get(agent.current_room)
            agent.current_temp = temp

            if temp is not None:
                try:
                    agent.comfort_level = max(0.0, agent.preferred_temp - abs(temp - agent.preferred_temp))
                except Exception:
                    agent.comfort_level = None

                agent.using_ac = abs(temp - agent.preferred_temp) > agent.comfort_tolerance
            else:
                agent.comfort_level = None
                agent.using_ac = False

            agent.current_day = getattr(self, "current_day", None)
            agent.current_hour = getattr(self, "current_hour", None)

            self.agent_results.append({
                "day": agent.current_day,
                "hour": agent.current_hour,
                "step": self.current_step,
                "agent_id": agent.unique_id,
                "agent_type": agent.agent_type,
                "room": agent.current_room,
                "current_temp": agent.current_temp,
                "comfort_level": round(agent.comfort_level, 2) if agent.comfort_level is not None else None,
                "using_ac": agent.using_ac,
                "preferred_temp": agent.preferred_temp
            })

        try:
            self.datacollector.collect(self)
        except Exception:
            pass

        self.current_step += 1

    def collect_agent_results(self):
        return getattr(self, "agent_results", [])

    # ============================================================
    #  Export results
    # ============================================================
    def export_zone_csv(self, filename="zone_results.csv"):
        df = self.datacollector.get_model_vars_dataframe()
        df.to_csv(filename)
        print(f"Zone-level results saved to {filename}")

    def export_agent_csv(self, filename="agent_results.csv"):
        df = pd.DataFrame(self.agent_results)
        df.to_csv(filename, index=False)
        print(f"Agent-level results saved to {filename}")

    # # ============================================================
    # #  Quick run + export
    # # ============================================================
    # def run_simulation_and_export(self, steps, zone_file="zone_results.csv", agent_file="agent_results.csv"):
    #     for _ in range(steps):
    #         self.step_agents(ep_model=self if self.ep_control else None)
    #     self.export_zone_csv(zone_file)
    #     self.export_agent_csv(agent_file)



