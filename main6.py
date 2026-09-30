import os
import math
import json
import csv
import random
from collections import deque, OrderedDict, namedtuple
from statistics import mean
from typing import Dict, Iterator, List, Optional, Union
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils
import geopandas as gpd
from geopandas import GeoDataFrame, sjoin
from shapely.geometry import Point
from geopy import distance
import networkx as nx
import igraph as ig
import osmnx as ox
import matplotlib.pyplot as plt
import folium
from mesa import Model, Agent
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from tqdm import tqdm, auto
from scipy.spatial import cKDTree
import pickle
import glob
import subprocess
import mesa
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class DQNPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, action_size)

        self.dropout = nn.Dropout(p=0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = self.dropout(x)

        x = torch.relu(self.fc4(x))
        if x.shape[0] > 1:
            x = self.bn4(x)
        x = self.dropout(x)

        x = torch.relu(self.fc5(x))
        if x.shape[0] > 1:
            x = self.bn5(x)
        x = self.dropout(x)

        q_values = self.fc6(x)
        return q_values
    
state_size = 5
action_size = 11
pretrained_model = DQNPolicy(state_size, action_size) 

class EVAgent_sim(Agent):
    def __init__(self, unique_id, state_size, action_size,
                 vehicle_type, full_battery, init_soc, alpha, trip_plan, soc_threshold, consum_rate,n_number, chance_charge, battery_chance, evmodel):
        super().__init__(unique_id, evmodel)
        self.unique_id = unique_id
        self.state_size = state_size
        self.action_size = action_size
        self.vehicle_type = vehicle_type

        self.full_battery = full_battery
        self.init_soc = init_soc
        self.alpha = alpha
        self.trip_plan = trip_plan
        self.soc_threshold = soc_threshold
        self.consum_rate = consum_rate
        self.n_number = n_number
        self.chance_charge = chance_charge
        self.battery_chance = battery_chance
        self.evmodel = evmodel

        self.beta = 1 - self.alpha
        self.agent_type = 'DQNAgent'

        self.gamma = 0.95
        self.nn_model = DQNPolicy(self.state_size, self.action_size)

        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.orig = None
        self.dest = None
        self.trip_purpose = None
        self.status = None
        self.cost_impact = None
        self.time_impact = None
        self.threshold_impact = None
        self.distance_idx = []
        self.battery_idx = []
        self.route_idx = []
        self.status_idx = []
        self.ev_time_idx = []
        self.soc_idx = []
        self.finance_payment_idx = []
        self.charge_spent_time_idx = []
        self.cost_impact_idx = []
        self.time_impact_idx = []
        self.threshold_impact_idx = []
        self.cost_idx = []
        self.shortest_path = None
        self.current_pos = None
        self.target_pos = None
        self.travelled_dis = 0
        self.consumed_battery = 0
        self.cost = 0
        self.current_soc = 0
        self.ev_time = 0
        self.shortest_path_length = 0
        self.chosen_amount = 0
        self.current_coor = None
        self.radius = 0
        self.left_distance_current_trip = None
        self.left_distance_all_trips = None

        self.finance_payment = 0
        self.travelled_dis = 0
        self.charge_spent_time = 0
        self.reward_idx = []
        self.action_idx = []
        self.charge_times = 0
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.next_state = None
        self.done = None
        self.distance_dict = None
        self.speed = None
        self.charge_start_time = None
        self.charge_end_time = None
        self.chosen_station_idx = []
    
    '''def load_state(self, new_pretrained_state): 
        self.pretrained_model_state = new_pretrained_state
        self.nn_model.load_state_dict(new_pretrained_state)
        self.nn_model.eval()'''
    
    def update_model_state(self, model_state_dict):
        self.nn_model = DQNPolicy(self.state_size, self.action_size)
        self.nn_model.load_state_dict(model_state_dict[self.vehicle_type])
        self.nn_model.eval() 

    def reset_dynamic_state(self):
        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.orig = None
        self.dest = None
        self.trip_purpose = None
        self.status = None
        self.cost_impact = None
        self.time_impact = None
        self.threshold_impact = None
        self.distance_idx = []
        self.battery_idx = []
        self.route_idx = []
        self.status_idx = []
        self.ev_time_idx = []
        self.soc_idx = []
        self.finance_payment_idx = []
        self.charge_spent_time_idx = []
        self.cost_impact_idx = []
        self.time_impact_idx = []
        self.threshold_impact_idx = []
        self.cost_idx = []
        self.shortest_path = None
        self.current_pos = None
        self.target_pos = None
        self.travelled_dis = 0
        self.consumed_battery = 0
        self.cost = 0
        self.current_soc = 0
        self.ev_time = 0
        self.shortest_path_length = 0
        self.chosen_amount = 0
        self.current_coor = None
        self.radius = 0
        self.left_distance_current_trip = None
        self.left_distance_all_trips = None
        self.steps_done = 0

        self.finance_payment = 0
        self.travelled_dis = 0
        self.charge_spent_time = 0
        self.reward_idx = []
        self.action_idx = []
        self.charge_times = 0
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.next_state = None
        self.done = None
        self.distance_dict = None
        self.speed = None
        self.charge_start_time = None
        self.chosen_station_idx =[]
        self.charge_end_time = None
            
    #### useful functions #####
    def update_next_pos(self, shortest_path, current_pos):
        if current_pos in shortest_path:
            current_index = shortest_path.index(current_pos)
            if current_index < len(shortest_path) - 1:
                return shortest_path[current_index + 1]
        return None

    def pos_to_coor(self, current_pos):
        pos_lon, pos_lat = self.evmodel.train_id_mapping_coor[str(current_pos)]
        current_coor = np.array([pos_lat, pos_lon])
        return current_coor

    def get_charge_plan(self, chosen_amount, candidate_stations, distance_dict, alpha, beta, travel_speed):
        candidate_stations_info = {key: self.evmodel.station_info[key] for key in candidate_stations if key in self.evmodel.station_info}
        cost_results = {}
        for i, [price, speed, init_park, add_park] in candidate_stations_info.items():
            distance_to_station = distance_dict[i]
            travel_time_to_station = distance_to_station / travel_speed
            charge_time_record = chosen_amount / speed
            if charge_time_record <= 1:
              log_pay = math.log1p(chosen_amount * price + init_park)
              log_time = math.log1p(charge_time_record + travel_time_to_station)
              station_cost = alpha * log_pay + beta * log_time
            else:
              log_pay = math.log1p(chosen_amount * price + init_park + (charge_time_record-1) * add_park)
              log_time = math.log1p(charge_time_record + travel_time_to_station)
              station_cost = alpha * log_pay + beta * log_time

            cost_results[i] = station_cost

        chosen_station = min(cost_results, key=cost_results.get)
        chosen_price = candidate_stations_info[chosen_station][0]
        chosen_speed = candidate_stations_info[chosen_station][1]
        chosen_init_park = candidate_stations_info[chosen_station][2]
        chosen_add_park = candidate_stations_info[chosen_station][3]
        chosen_cost = cost_results[chosen_station]

        return chosen_station, chosen_price, chosen_speed, chosen_init_park, chosen_add_park, chosen_cost

    def balance_amount (self, chosen_amount, current_battery, full_battery):
        if chosen_amount + current_battery >= full_battery:
            chosen_amount_new = full_battery - current_battery
        else:
            chosen_amount_new = chosen_amount

        return chosen_amount_new

    def get_shortest_path(self, start_pos, end_pos): 
        if (start_pos, end_pos) in self.evmodel.shortest_paths:
            shortest_path = self.evmodel.shortest_paths[(start_pos, end_pos)]
            return shortest_path
        else:
            return print(f'Agent Error: cannot find the shortest path bewteen{start_pos} and {end_pos}')
    
    def get_shortest_path_length(self, start_pos, end_pos):
        if start_pos == end_pos:
            return 0
        if (start_pos, end_pos) in self.evmodel.shortest_paths_distances:
            shortest_path_length = self.evmodel.shortest_paths_distances[(start_pos, end_pos)]
            return shortest_path_length
        else:
            return print(f'Agent Error: cannot find the shortest path length bewteen {start_pos} and {end_pos}')

    def at_destination(self):
        return self.current_pos == self.dest
    
    #### useful functions #####

    def initialisation (self):
        self.trip_start_time = 0
        self.orig = 0
        self.dest = 0
        self.speed = 0
        self.ev_time = 0
        self.current_pos = 0
        self.travelled_dis = 0
        self.battery_need_to_dest = 0
        self.shortest_path = None
        self.shortest_path_length = 0
        self.status = 'start'

    def evaluation (self):
        self.orig = list(self.trip_plan[self.trip_idx].values())[0][0]
        self.dest = list(self.trip_plan[self.trip_idx].values())[0][1]
        self.trip_start_time = list(self.trip_plan[self.trip_idx].values())[0][2]
        self.speed = list(self.trip_plan[self.trip_idx].values())[0][3]
        self.trip_purpose = list(self.trip_plan[self.trip_idx].values())[0][4]
        self.current_pos = self.orig
        self.current_coor = self.pos_to_coor(self.current_pos)
        self.target_pos = self.dest
        self.ev_time = self.trip_start_time

        self.shortest_path = self.get_shortest_path(self.current_pos, self.target_pos)
        self.shortest_path_length = self.get_shortest_path_length(self.current_pos, self.target_pos)
        self.battery_need_to_dest = self.shortest_path_length * self.consum_rate

        self.route_idx.append(self.current_pos)
        self.ev_time_idx.append(self.ev_time)
        self.status_idx.append(self.status)
        self.battery_idx.append(self.current_battery)

    def observe_state(self):
        mean_distance = 14432.53
        std_distance = 27270.52
        dist_to_dest = self.get_shortest_path_length(self.current_pos, self.dest)
        scaled_dist_to_dest = (dist_to_dest - mean_distance) / std_distance
        soc_state = self.current_battery / self.full_battery
        scaled_ev_time = self.ev_time / 1440
        available_stations = sum(self.evmodel.get_station_avail(self.ev_time, station) == 0 for station in self.candidate_stations)
        scaled_available_stations = available_stations/len(self.candidate_stations)
        scaled_trip_left = self.trip_left/len(self.trip_plan)

        general_state = np.array([soc_state, scaled_dist_to_dest, scaled_ev_time, scaled_available_stations, scaled_trip_left])
        return general_state

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.nn_model(state_tensor)
            best_action = np.argmax(action_values.cpu().data.numpy())
            if self.charge_times >= 1:
                return 0
            else:
                return best_action

    def move_driver(self):
        while self.trip_left > 0:
            self.initialisation()
            self.evaluation()
            i = 1
            while i < len(self.shortest_path):
                self.status = 'drive'
                self.candidate_stations = []
                self.current_coor = self.pos_to_coor(self.current_pos)
                self.candidate_stations, self.distance_dict = self.evmodel.get_candidate_stations(self.current_pos, self.current_coor)
                current_state = self.observe_state()
                self.current_state = current_state
                action = self.act(current_state)
                self.last_action = action
                self.action_idx.append(action)

                if action == 0:
                    if self.charge_times >=1:
                        self.target_pos = self.dest
                        self.evmodel.grid.move_agent(self,self.target_pos)
                        self.travelled_dis = self.get_shortest_path_length(self.current_pos, self.target_pos)
                        self.final_path = self.get_shortest_path(self.current_pos, self.target_pos)
                        for node in self.final_path:
                            self.route_idx.append(node)
                        self.consumed_battery = self.travelled_dis*self.consum_rate
                        self.current_battery -= self.consumed_battery
                        self.current_soc = self.current_battery/self.full_battery
                        self.ev_time += self.travelled_dis / self.speed
                        i = len(self.shortest_path)
                        self.current_pos = self.dest
                    else:
                        self.target_pos = self.shortest_path[i]
                        self.evmodel.grid.move_agent(self, self.target_pos)
                        self.current_pos = self.target_pos
                        edge = self.evmodel.graph.get_eid(self.shortest_path[i - 1], self.shortest_path[i])
                        self.travelled_dis = self.evmodel.graph.es[edge]['length']
                        self.consumed_battery = self.travelled_dis * self.consum_rate
                        self.current_battery -= self.consumed_battery
                        self.current_soc = self.current_battery / self.full_battery
                        self.ev_time += self.travelled_dis / self.speed
                        self.route_idx.append(self.current_pos)
                        i += math.ceil(len(self.shortest_path) / 5)

                    self.battery_idx.append(self.current_battery)
                    self.soc_idx.append(self.current_soc)
                    self.status_idx.append(self.status)
                    self.ev_time_idx.append(self.ev_time)
                    self.distance_idx.append(self.travelled_dis)

                else:
                    self.execute_action(action)

                    if self.status == 'fail':
                        next_state = self.observe_state()
                        reward = self.calculate_reward()
                        self.reward_idx.append(reward)
                        self.cost_impact_idx.append(self.cost_impact)
                        self.time_impact_idx.append(self.time_impact)
                        self.threshold_impact_idx.append(self.threshold_impact)
                        done = self.at_destination()
                        self.last_reward = reward
                        self.next_state = next_state
                        self.done = done
                        break
                    else:
                        self.target_pos = self.dest
                        self.shortest_path = self.get_shortest_path(self.current_pos, self.target_pos)
                        self.shortest_path_length = self.get_shortest_path_length(self.current_pos, self.dest)
                        i = 1

                if not self.status == 'fail':
                    next_state = self.observe_state()
                    reward = self.calculate_reward()
                    self.reward_idx.append(reward)
                    self.cost_impact_idx.append(self.cost_impact)
                    self.time_impact_idx.append(self.time_impact)
                    self.threshold_impact_idx.append(self.threshold_impact)
                    done = self.at_destination()

                    self.last_reward = reward
                    self.next_state = next_state
                    self.done = done

                    self.travelled_dis = 0
                    self.charge_spent_time = 0
                    self.finance_payment = 0
                    self.charge_start_time = None
                    self.cost_impact= None
                    self.time_impact = None

                    if done:
                        self.status = 'finish'
                        self.status_idx.append(self.status)
                        self.route_idx.append(self.dest)
                        self.battery_idx.append(self.current_battery)
                        self.soc_idx.append(self.current_soc)
                        break

            self.trip_left -= 1
            self.trip_idx += 1
            if self.trip_left <= 0:
                self.status = 'finish all'
                self.status_idx.append(self.status)
                break
            else:
                self.status = 'start'
                self.status_idx.append(self.status)
        

    def execute_action(self, action):
        if action == 0 or self.charge_times >=1:
            pass
        else:
            self.charge_times += 1
            if self.charge_times > 1:
                print('charge times error')
            charge_percentage = action * 10
            self.current_coor = self.pos_to_coor(self.current_pos)
            self.radius = self.current_battery / self.consum_rate
            self.candidate_stations, self.distance_dict = self.evmodel.get_candidate_stations(self.current_pos, self.current_coor)

            if len(self.candidate_stations) > 0:
                self.chosen_amount = (charge_percentage / 100.0) * (self.full_battery - self.current_battery)
                self.chosen_station, self.chosen_price, self.chosen_speed, self.chosen_init_park, self.chosen_add_park, self.chosen_cost = self.get_charge_plan(self.chosen_amount,self.candidate_stations,self.distance_dict, self.alpha, self.beta, self.speed)
                self.charge()
            else:
                self.fail_trip()


    def charge(self):
        self.chosen_station_pos = self.evmodel.station_node[self.chosen_station]
        self.path_to_station = self.get_shortest_path(self.current_pos, int(self.chosen_station_pos))
        self.dist_to_station = self.get_shortest_path_length(self.current_pos, int(self.chosen_station_pos))
        self.target_pos = self.chosen_station
        i = 1
        while i < len(self.path_to_station):
            self.status = 'driving to station'
            self.evmodel.grid.move_agent(self, self.path_to_station[i])
            self.current_pos = self.path_to_station[i]
            edge = self.evmodel.graph.get_eid(self.path_to_station[i], self.path_to_station[i-1])
            self.travelled_dis = self.evmodel.graph.es[edge]['length']
            self.consumed_battery = self.travelled_dis * self.consum_rate
            self.current_battery -= self.consumed_battery
            self.current_soc = self.current_battery / self.full_battery
            self.ev_time += self.travelled_dis / self.speed

            self.battery_idx.append(self.current_battery)
            self.soc_idx.append(self.current_soc)
            self.route_idx.append(self.current_pos)
            self.status_idx.append(self.status)
            self.ev_time_idx.append(self.ev_time)
            self.distance_idx.append(self.travelled_dis)
            i += 1

        self.current_pos = int(self.chosen_station_pos)

        self.current_battery += self.chosen_amount
        self.current_soc = self.current_battery / self.full_battery
        self.target_pos = self.dest

        self.chosen_station_avail = self.evmodel.get_station_avail(self.ev_time, self.chosen_station)
        if self.chosen_station_avail == 1:
            self.status = 'queue'
            self.status_idx.append(self.status)
            self.status = 'charging'
            self.status_idx.append(self.status)
            self.charge_start_time = self.evmodel.get_queue_finish_time(self.ev_time, self.chosen_station)
            if self.charge_start_time is None:
                self.fail_trip()
                return
            else:
                self.charge_end_time = self.charge_start_time + self.chosen_amount / self.chosen_speed
                self.chosen_station_id, self.chosen_station_start_time, self.chosen_station_end_time = self.evmodel.take_up_station(self.chosen_station, self.charge_start_time, self.charge_end_time)
                self.charge_spent_time = self.charge_end_time - self.ev_time
                self.ev_time = self.charge_end_time
                self.chosen_station_idx.append(self.chosen_station_id)
                self.chosen_station_idx.append(self.chosen_station_start_time)
                self.chosen_station_idx.append(self.chosen_station_end_time)

        else:
            self.status = 'charging'
            self.charge_start_time = self.ev_time
            self.status_idx.append(self.status)
            self.charge_spent_time = self.chosen_amount / self.chosen_speed
            self.charge_end_time = self.charge_start_time + self.charge_spent_time
            self.chosen_station_id, self.chosen_station_start_time, self.chosen_station_end_time = self.evmodel.take_up_station(self.chosen_station, self.charge_start_time, self.charge_end_time)
            self.ev_time += self.charge_spent_time
            self.chosen_station_idx.append(self.chosen_station_id)
            self.chosen_station_idx.append(self.chosen_station_start_time)
            self.chosen_station_idx.append(self.chosen_station_end_time)

        if self.chosen_amount / self.chosen_speed > 1:
            self.finance_payment = self.chosen_init_park + self.chosen_add_park * (self.chosen_amount / self.chosen_speed - 1) + self.chosen_amount * self.chosen_price
        else:
            self.finance_payment = self.chosen_init_park + self.chosen_amount * self.chosen_price

        log_pay = math.log1p(self.finance_payment)
        log_time = math.log1p(self.charge_spent_time + self.dist_to_station / self.speed)
        self.cost = log_pay * self.alpha + self.beta * log_time

        self.finance_payment_idx.append(self.finance_payment)
        self.charge_spent_time_idx.append(self.charge_spent_time)
        self.cost_idx.append(self.cost)
        self.ev_time_idx.append(self.ev_time)
        self.soc_idx.append(self.current_soc)
        self.battery_idx.append(self.current_battery)

        self.status = 'finish charging'
        self.status_idx.append(self.status)
        self.target_pos = self.dest
        self.travelled_dis = self.dist_to_station

    def fail_trip(self):
        self.status = 'fail'
        self.status_idx.append(self.status)
        self.trip_left = 0

    def calculate_reward(self):
        if self.status == 'fail':
            extra = -1
            self.threshold_impact = self.current_battery/self.full_battery- self.soc_threshold
            reward = 20* extra + 5 * self.threshold_impact
        else:
            if self.at_destination():
                extra = 1
            else:
                extra = 0

            log_whole_time = math.log1p(self.travelled_dis / self.speed + self.charge_spent_time)
            log_finance_payment = math.log1p(self.finance_payment)
            mn = self.charge_times/self.n_number
            self.cost_impact = ((self.beta * log_whole_time + self.alpha * log_finance_payment) + 1) ** mn
            self.threshold_impact = self.current_battery/self.full_battery- self.soc_threshold
            if self.charge_start_time is None:
                reward =  10/(self.cost_impact) + 20 * extra + 5 * self.threshold_impact
            else:
                self.time_impact = (math.log1p(self.charge_start_time - self.trip_start_time)) * (self.chance_charge + self.battery_chance +1)
                reward = 10/(self.cost_impact + self.time_impact) + 20 * extra + 5 * self.threshold_impact

        return reward

    def step(self):
        self.move_driver()
   
class EV_model_sim(Model):
    
    def __init__(self, output_path:str, EV_agent: Optional[pd.DataFrame] = None,
                 station_location: Optional[dict] = None, station_info: Optional[dict] = None,
                 station_avail: Optional[dict] = None, station_node: Optional[dict] = None,
                 train_id_mapping_coor: Optional[dict] = None, graph: Optional[ig.Graph] = None,
                 networkx_graph: Optional[nx.Graph] = None, 
                 shortest_paths: Optional[dict] = None, shortest_paths_distances: Optional[dict] = None):

        super().__init__()
        self.num_agents = len(EV_agent) if EV_agent is not None else 0
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.EV_agent = EV_agent
        self.station_location = station_location
        self.station_info = station_info
        self.station_avail = station_avail
        self.station_node = station_node
        self.train_id_mapping_coor = train_id_mapping_coor
        self.graph = graph
        self.networkx_graph = networkx_graph
        self.grid = NetworkGrid(self.networkx_graph)
        self.shortest_paths = shortest_paths
        self.shortest_paths_distances = shortest_paths_distances
        self._station_location_keys = list(self.station_location.keys())

        self.location_list = list(station_location.values())
        self.tree = cKDTree(self.location_list)

        for _, row in EV_agent.iterrows():
            e = EVAgent_sim(row['IndividualID'], state_size = 5, action_size = 11, vehicle_type = row['vehicle_type'], 
                        full_battery=row['full_battery'], init_soc=row['start_battery'], alpha=row['alpha_new'], trip_plan=row['trip_plan_new3'], 
                        soc_threshold=row['threshold_new'], consum_rate=row['consum_rate'], n_number = row['m_new'], chance_charge = row['chance_charge'], 
                        battery_chance = row['battery_chance'], evmodel=self)
            self.schedule.add(e)
            self.grid.place_agent(e, row['o_osmid'])

        self.data_collector = DataCollector(
            agent_reporters={
                'unique_id': lambda agent: agent.unique_id, 
                'status_idx': lambda agent: agent.status_idx, 
                'battery_idx': lambda agent: agent.battery_idx, 
                'route_idx': lambda agent: agent.route_idx, 
                'soc_idx': lambda agent: agent.soc_idx,
                'ev_time_idx': lambda agent: agent.ev_time_idx, 
                'cost_idx': lambda agent: agent.cost_idx, 
                'finance_payment_idx': lambda agent: agent.finance_payment_idx, 
                'distance_idx': lambda agent: agent.distance_idx,
                'reward_idx': lambda agent: agent.reward_idx, 
                'cost_impact_idx': lambda agent: agent.cost_impact_idx,
                'time_impact_idx': lambda agent: agent.time_impact_idx, 
                'threshold_impact_idx': lambda agent: agent.threshold_impact_idx,
                'charge_times': lambda agent: agent.charge_times,
                'chosen_station_idx': lambda agent: agent.chosen_station_idx})

    def is_episode_ended(self):
        return all(agent.at_destination() or agent.trip_left ==0 for agent in self.schedule.agents)

    def step(self, model_state_dict):
        self.station_avail = {key: 0 for key in self.station_avail}
        for agent in self.schedule.agents:
            agent.reset_dynamic_state()
            o_osmid = self.EV_agent.loc[self.EV_agent['IndividualID'] == agent.unique_id, 'o_osmid'].iloc[0]
            self.grid.move_agent(agent, o_osmid)
        self.data_collector = DataCollector(
            agent_reporters={
                'unique_id': lambda agent: agent.unique_id, 
                'status_idx': lambda agent: agent.status_idx, 
                'battery_idx': lambda agent: agent.battery_idx, 
                'route_idx': lambda agent: agent.route_idx, 
                'soc_idx': lambda agent: agent.soc_idx,
                'ev_time_idx': lambda agent: agent.ev_time_idx, 
                'cost_idx': lambda agent: agent.cost_idx, 
                'finance_payment_idx': lambda agent: agent.finance_payment_idx, 
                'distance_idx': lambda agent: agent.distance_idx,
                'reward_idx': lambda agent: agent.reward_idx, 
                'cost_impact_idx': lambda agent: agent.cost_impact_idx,
                'time_impact_idx': lambda agent: agent.time_impact_idx, 
                'threshold_impact_idx': lambda agent: agent.threshold_impact_idx,
                'charge_times': lambda agent: agent.charge_times,
                'chosen_station_idx': lambda agent: agent.chosen_station_idx} 
        )
        for agent in self.schedule.agents:
            agent.update_model_state(model_state_dict)
        self.schedule.step()
        self.data_collector.collect(self)
        return self.data_collector.get_agent_vars_dataframe()

    def run(self, steps: int):
        self.data_collector.collect(self)
        for k in range(steps):
            self.step()
        return self.data_collector.get_agent_vars_dataframe()

    def get_shortest_path_length(self, start_pos, end_pos):
        if start_pos == end_pos:
            return 0  
        if (start_pos, end_pos) in self.shortest_paths_distances:
            shortest_path_length = self.shortest_paths_distances[(start_pos, end_pos)]
            return shortest_path_length
        else:
            print(f'Station Error: cannot find the shortest path length between {start_pos} and {end_pos}')
            
    def get_candidate_stations(self, current_node, current_coor):
        _, indices = self.tree.query(current_coor, k=5)
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            indices = [indices]
        all_candidate_stations = [self._station_location_keys[i] for i in indices]
        all_candidate_stations_node = [self.station_node[key] for key in all_candidate_stations] #if key in self.station_node]
        all_candidate_stations_node = [int(item) for item in all_candidate_stations_node]

        all_candidate_stations_distances = []
        for node in all_candidate_stations_node:
            distance = self.get_shortest_path_length(current_node, node)
            all_candidate_stations_distances.append(distance)
        distance_dict = dict(zip(all_candidate_stations, all_candidate_stations_distances))

        return all_candidate_stations, distance_dict

    def get_station_avail(self, ev_time, station_id):
        rounded_ev_time = ((ev_time + 14) // 15) * 15
        station_int = int(station_id.replace('_', ''))
        availability_key = f"{station_int}_{int(rounded_ev_time)}"
        avail = self.station_avail.get(availability_key)
        return avail

    def get_queue_finish_time(self, ev_time, station_id):
        port_id = int(station_id.replace('_', ''))
        rounded_ev_time = int(((ev_time + 14) // 15) * 15)
        for time_slot in range(rounded_ev_time, 14400 + 1, 15):
            availability_key = f"{port_id}_{time_slot}"
            if self.station_avail.get(availability_key) == 0:
                return time_slot

    def take_up_station(self, station_id, ev_time, expect_charge_end_time):
        port_id = int(station_id.replace('_', ''))
        rounded_ev_time = int(((ev_time + 14) // 15) * 15)
        rounded_end_time = int(((expect_charge_end_time + 14) // 15) * 15)
        rounded_end_time += 15
        for time in range(rounded_ev_time, rounded_end_time, 15):
            availability_key = f"{port_id}_{time}"
            self.station_avail[availability_key] = 1
        return port_id, rounded_ev_time, rounded_end_time

class EVAgent_train(Agent):
    def __init__(self, unique_id, state_size, action_size,
                 vehicle_type, full_battery, init_soc, alpha, trip_plan, soc_threshold, consum_rate,n_number, chance_charge, battery_chance, evmodel, base_model_path):
        super().__init__(unique_id, evmodel)
        self.unique_id = unique_id
        self.state_size = state_size
        self.action_size = action_size
        self.vehicle_type = vehicle_type

        self.full_battery = full_battery
        self.init_soc = init_soc
        self.alpha = alpha
        self.trip_plan = trip_plan
        self.soc_threshold = soc_threshold
        self.consum_rate = consum_rate
        self.n_number = n_number
        self.chance_charge = chance_charge
        self.battery_chance = battery_chance
        self.evmodel = evmodel

        self.beta = 1 - self.alpha
        self.agent_type = 'DQNAgent'

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.nn_model = DQNPolicy(self.state_size, self.action_size)
        self.nn_model.load_state_dict(torch.load(base_model_path, map_location="cpu")) 
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=2000)

        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.orig = None
        self.dest = None
        self.trip_purpose = None
        self.status = None
        self.cost_impact = None
        self.time_impact = None
        self.threshold_impact = None
        self.distance_idx = []
        self.battery_idx = []
        self.route_idx = []
        self.status_idx = []
        self.ev_time_idx = []
        self.soc_idx = []
        self.finance_payment_idx = []
        self.charge_spent_time_idx = []
        self.cost_impact_idx = []
        self.time_impact_idx = []
        self.threshold_impact_idx = []
        self.cost_idx = []
        self.shortest_path = None
        self.current_pos = None
        self.target_pos = None
        self.travelled_dis = 0
        self.consumed_battery = 0
        self.cost = 0
        self.current_soc = 0
        self.ev_time = 0
        self.shortest_path_length = 0
        self.chosen_amount = 0
        self.current_coor = None
        self.radius = 0
        self.left_distance_current_trip = None
        self.left_distance_all_trips = None

        self.finance_payment = 0
        self.travelled_dis = 0
        self.charge_spent_time = 0
        self.reward_idx = []
        self.action_idx = []
        self.charge_times = 0
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.next_state = None
        self.done = None
        self.distance_dict = None
        self.speed = None
        self.charge_start_time = None
        self.charge_end_time = None
        self.chosen_station_idx =[]
        self.chosen_station_id = None
        self.chosen_station_start_time = None 
        self.chosen_station_end_time = None

    def reset_dynamic_state(self):
        self.trip_left = len(self.trip_plan)
        self.trip_idx = 0
        self.current_battery = self.init_soc * self.full_battery

        self.orig = None
        self.dest = None
        self.trip_purpose = None
        self.status = None
        self.cost_impact = None
        self.time_impact = None
        self.threshold_impact = None
        self.distance_idx = []
        self.battery_idx = []
        self.route_idx = []
        self.status_idx = []
        self.ev_time_idx = []
        self.soc_idx = []
        self.finance_payment_idx = []
        self.charge_spent_time_idx = []
        self.cost_impact_idx = []
        self.time_impact_idx = []
        self.threshold_impact_idx = []
        self.cost_idx = []
        self.shortest_path = None
        self.current_pos = None
        self.target_pos = None
        self.travelled_dis = 0
        self.consumed_battery = 0
        self.cost = 0
        self.current_soc = 0
        self.ev_time = 0
        self.shortest_path_length = 0
        self.chosen_amount = 0
        self.current_coor = None
        self.radius = 0
        self.left_distance_current_trip = None
        self.left_distance_all_trips = None
        self.steps_done = 0

        self.finance_payment = 0
        self.travelled_dis = 0
        self.charge_spent_time = 0
        self.reward_idx = []
        self.action_idx = []
        self.charge_times = 0
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.next_state = None
        self.done = None
        self.distance_dict = None
        self.speed = None
        self.charge_start_time = None
        self.charge_end_time = None
        self.chosen_station_idx =[]
        self.chosen_station_id = None
        self.chosen_station_start_time = None 
        self.chosen_station_end_time = None

    #### useful functions #####
    def update_next_pos(self, shortest_path, current_pos):
        if current_pos in shortest_path:
            current_index = shortest_path.index(current_pos)
            if current_index < len(shortest_path) - 1:
                return shortest_path[current_index + 1]
        return None

    def pos_to_coor(self, current_pos):
        pos_lon, pos_lat = self.evmodel.train_id_mapping_coor[str(current_pos)]
        current_coor = np.array([pos_lat, pos_lon])
        return current_coor

    def get_charge_plan(self, chosen_amount, candidate_stations, distance_dict, alpha, beta, travel_speed):
        candidate_stations_info = {key: self.evmodel.station_info[key] for key in candidate_stations if key in self.evmodel.station_info}
        cost_results = {}
        for i, [price, speed, init_park, add_park] in candidate_stations_info.items():
            distance_to_station = distance_dict[i]
            travel_time_to_station = distance_to_station / travel_speed
            charge_time_record = chosen_amount / speed
            if charge_time_record <= 1:
              log_pay = math.log1p(chosen_amount * price + init_park)
              log_time = math.log1p(charge_time_record + travel_time_to_station)
              station_cost = alpha * log_pay + beta * log_time
            else:
              log_pay = math.log1p(chosen_amount * price + init_park + (charge_time_record-1) * add_park)
              log_time = math.log1p(charge_time_record + travel_time_to_station)
              station_cost = alpha * log_pay + beta * log_time

            cost_results[i] = station_cost

        chosen_station = min(cost_results, key=cost_results.get)
        chosen_price = candidate_stations_info[chosen_station][0]
        chosen_speed = candidate_stations_info[chosen_station][1]
        chosen_init_park = candidate_stations_info[chosen_station][2]
        chosen_add_park = candidate_stations_info[chosen_station][3]
        chosen_cost = cost_results[chosen_station]

        return chosen_station, chosen_price, chosen_speed, chosen_init_park, chosen_add_park, chosen_cost

    def balance_amount (self, chosen_amount, current_battery, full_battery):
        if chosen_amount + current_battery >= full_battery:
            chosen_amount_new = full_battery - current_battery
        else:
            chosen_amount_new = chosen_amount

        return chosen_amount_new
    
    def get_shortest_path(self, start_pos, end_pos): 
        if (start_pos, end_pos) in self.evmodel.shortest_paths:
            shortest_path = self.evmodel.shortest_paths[(start_pos, end_pos)]
            return shortest_path
        else:
            return print(f'Agent Error: cannot find the shortest path bewteen{start_pos} and {end_pos}')
    
    def get_shortest_path_length(self, start_pos, end_pos):
        if start_pos == end_pos:
            return 0
        if (start_pos, end_pos) in self.evmodel.shortest_paths_distances:
            shortest_path_length = self.evmodel.shortest_paths_distances[(start_pos, end_pos)]
            return shortest_path_length
        else:
            return print(f'Agent Error: cannot find the shortest path length bewteen {start_pos} and {end_pos}')

    def at_destination(self):
        return self.current_pos == self.dest
    
    #### useful functions #####

    def initialisation (self):
        self.trip_start_time = 0
        self.orig = 0
        self.dest = 0
        self.speed = 0
        self.ev_time = 0
        self.current_pos = 0
        self.travelled_dis = 0
        self.battery_need_to_dest = 0
        self.shortest_path = None
        self.shortest_path_length = 0
        self.status = 'start'

    def evaluation (self):
        self.orig = list(self.trip_plan[self.trip_idx].values())[0][0]
        self.dest = list(self.trip_plan[self.trip_idx].values())[0][1]
        self.trip_start_time = list(self.trip_plan[self.trip_idx].values())[0][2]
        self.speed = list(self.trip_plan[self.trip_idx].values())[0][3]
        self.trip_purpose = list(self.trip_plan[self.trip_idx].values())[0][4]
        self.current_pos = self.orig
        self.current_coor = self.pos_to_coor(self.current_pos)
        self.target_pos = self.dest
        self.ev_time = self.trip_start_time
        self.shortest_path = self.get_shortest_path(self.current_pos, self.target_pos)
        self.shortest_path_length = self.get_shortest_path_length(self.current_pos, self.target_pos)
        self.battery_need_to_dest = self.shortest_path_length * self.consum_rate

        self.route_idx.append(self.current_pos)
        self.ev_time_idx.append(self.ev_time)
        self.status_idx.append(self.status)
        self.battery_idx.append(self.current_battery)

    def observe_state(self):
        mean_distance = 14432.53
        std_distance = 27270.52
        dist_to_dest = self.get_shortest_path_length(self.current_pos, self.dest)
        scaled_dist_to_dest = (dist_to_dest - mean_distance) / std_distance
        soc_state = self.current_battery / self.full_battery
        scaled_ev_time = self.ev_time / 1440
        available_stations = sum(self.evmodel.get_station_avail(self.ev_time, station) == 0 for station in self.candidate_stations)
        scaled_available_stations = available_stations/len(self.candidate_stations)
        scaled_trip_left = self.trip_left/len(self.trip_plan)

        general_state = np.array([soc_state, scaled_dist_to_dest, scaled_ev_time, scaled_available_stations, scaled_trip_left])
        return general_state

    def act(self, state):
        if random.random() <= self.epsilon:
            available_actions = [0] if self.charge_times >= 1 else list(range(self.action_size))
            return random.choice(available_actions)

        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.nn_model(state_tensor)
                best_action = np.argmax(action_values.cpu().data.numpy())
                if self.charge_times >= 1:
                    return 0
                else:
                    return best_action

    def move_driver(self):
        while self.trip_left > 0:
            self.initialisation()
            self.evaluation()
            i = 1
            while i < len(self.shortest_path):
                self.status = 'drive'
                self.candidate_stations = []
                self.current_coor = self.pos_to_coor(self.current_pos)
                self.candidate_stations, self.distance_dict = self.evmodel.get_candidate_stations(self.current_pos, self.current_coor)
                current_state = self.observe_state()
                self.current_state = current_state
                action = self.act(current_state)
                self.last_action = action
                self.action_idx.append(action)

                if action == 0:
                    if self.charge_times >=1:
                        self.target_pos = self.dest
                        #self.evmodel.grid.move_agent(self,self.target_pos)
                        self.travelled_dis = self.get_shortest_path_length(self.current_pos, self.target_pos)
                        self.final_path = self.get_shortest_path(self.current_pos, self.target_pos)
                        for node in self.final_path:
                            self.route_idx.append(node)
                        self.consumed_battery = self.travelled_dis*self.consum_rate
                        self.current_battery -= self.consumed_battery
                        self.current_soc = self.current_battery/self.full_battery
                        self.ev_time += self.travelled_dis / self.speed
                        i = len(self.shortest_path)
                        self.current_pos = self.dest
                    else:
                        self.target_pos = self.shortest_path[i]
                        #self.evmodel.grid.move_agent(self, self.target_pos)
                        self.current_pos = self.target_pos
                        edge = self.evmodel.graph.get_eid(self.shortest_path[i - 1], self.shortest_path[i])
                        self.travelled_dis = self.evmodel.graph.es[edge]['length']
                        self.consumed_battery = self.travelled_dis * self.consum_rate
                        self.current_battery -= self.consumed_battery
                        self.current_soc = self.current_battery / self.full_battery
                        self.ev_time += self.travelled_dis / self.speed
                        self.route_idx.append(self.current_pos)
                        i += 1

                    self.battery_idx.append(self.current_battery)
                    self.soc_idx.append(self.current_soc)
                    self.status_idx.append(self.status)
                    self.ev_time_idx.append(self.ev_time)
                    self.distance_idx.append(self.travelled_dis)

                else:
                    self.execute_action(action)

                    if self.status == 'fail':
                        next_state = self.observe_state()
                        reward = self.calculate_reward()
                        self.reward_idx.append(reward)
                        self.cost_impact_idx.append(self.cost_impact)
                        self.time_impact_idx.append(self.time_impact)
                        self.threshold_impact_idx.append(self.threshold_impact)
                        done = self.at_destination()
                        self.remember(current_state, action, reward, next_state, done)
                        self.replay(64)
                        self.last_reward = reward
                        self.next_state = next_state
                        self.done = done
                        break
                    else:
                        self.target_pos = self.dest
                        self.shortest_path = self.get_shortest_path(self.current_pos, self.target_pos)
                        self.shortest_path_length = self.get_shortest_path_length(self.current_pos, self.dest)
                        i = 1

                if not self.status == 'fail':
                    next_state = self.observe_state()
                    reward = self.calculate_reward()
                    self.reward_idx.append(reward)
                    self.cost_impact_idx.append(self.cost_impact)
                    self.time_impact_idx.append(self.time_impact)
                    self.threshold_impact_idx.append(self.threshold_impact)
                    done = self.at_destination()
                    self.remember(current_state, action, reward, next_state, done)
                    self.replay(64)

                    self.last_reward = reward
                    self.next_state = next_state
                    self.done = done

                    self.travelled_dis = 0
                    self.charge_spent_time = 0
                    self.finance_payment = 0
                    self.charge_start_time = None
                    self.cost_impact= None
                    self.time_impact = None

                    if done:
                        self.status = 'finish'
                        self.status_idx.append(self.status)
                        self.route_idx.append(self.dest)
                        self.battery_idx.append(self.current_battery)
                        self.soc_idx.append(self.current_soc)
                        break

            self.trip_left -= 1
            self.trip_idx += 1
            if self.trip_left <= 0:
                self.status = 'finish all'
                self.status_idx.append(self.status)
                break
            else:
                self.status = 'start'
                self.status_idx.append(self.status)

    def execute_action(self, action):
        if action == 0 or self.charge_times >=1:
            pass
        else:
            self.charge_times += 1
            if self.charge_times > 1:
                print('charge times error')
            charge_percentage = action * 10
            self.current_coor = self.pos_to_coor(self.current_pos)
            self.radius = self.current_battery / self.consum_rate
            self.candidate_stations, self.distance_dict = self.evmodel.get_candidate_stations(self.current_pos, self.current_coor)

            if len(self.candidate_stations) > 0:
                self.chosen_amount = (charge_percentage / 100.0) * (self.full_battery - self.current_battery)
                self.chosen_station, self.chosen_price, self.chosen_speed, self.chosen_init_park, self.chosen_add_park, self.chosen_cost = self.get_charge_plan(self.chosen_amount,self.candidate_stations,self.distance_dict, self.alpha, self.beta, self.speed)
                self.charge()
            else:
                self.fail_trip()


    def charge(self):
        self.chosen_station_pos = self.evmodel.station_node[self.chosen_station]
        self.path_to_station = self.get_shortest_path(self.current_pos, int(self.chosen_station_pos))
        self.dist_to_station = self.get_shortest_path_length(self.current_pos, int(self.chosen_station_pos))
        self.target_pos = self.chosen_station
        i = 1
        while i < len(self.path_to_station):
            self.status = 'driving to station'
            #self.evmodel.grid.move_agent(self, self.path_to_station[i])
            self.current_pos = self.path_to_station[i]
            edge = self.evmodel.graph.get_eid(self.path_to_station[i], self.path_to_station[i-1])
            self.travelled_dis = self.evmodel.graph.es[edge]['length']
            self.consumed_battery = self.travelled_dis * self.consum_rate
            self.current_battery -= self.consumed_battery
            self.current_soc = self.current_battery / self.full_battery
            self.ev_time += self.travelled_dis / self.speed

            self.battery_idx.append(self.current_battery)
            self.soc_idx.append(self.current_soc)
            self.route_idx.append(self.current_pos)
            self.status_idx.append(self.status)
            self.ev_time_idx.append(self.ev_time)
            self.distance_idx.append(self.travelled_dis)
            i += 1

        self.current_pos = int(self.chosen_station_pos)
        self.current_battery += self.chosen_amount
        self.current_soc = self.current_battery / self.full_battery
        self.target_pos = self.dest

        self.chosen_station_avail = self.evmodel.get_station_avail(self.ev_time, self.chosen_station)
        if self.chosen_station_avail == 1:
            self.status = 'queue'
            self.status_idx.append(self.status)
            self.status = 'charging'
            self.status_idx.append(self.status)
            self.charge_start_time = self.evmodel.get_queue_finish_time(self.ev_time, self.chosen_station)
            if self.charge_start_time is None:
                self.fail_trip()
                return
            else:
                self.charge_end_time = self.charge_start_time + self.chosen_amount / self.chosen_speed
                self.chosen_station_id, self.chosen_station_start_time, self.chosen_station_end_time = self.evmodel.take_up_station(self.chosen_station, self.charge_start_time, self.charge_end_time)
                self.charge_spent_time = self.charge_end_time - self.ev_time
                self.ev_time = self.charge_end_time
                self.chosen_station_idx.append(self.chosen_station_id)
                self.chosen_station_idx.append(self.chosen_station_start_time)
                self.chosen_station_idx.append(self.chosen_station_end_time)
        else:
            self.status = 'charging'
            self.charge_start_time = self.ev_time
            self.status_idx.append(self.status)
            self.charge_spent_time = self.chosen_amount / self.chosen_speed
            self.charge_end_time = self.ev_time + self.charge_spent_time
            self.chosen_station_id, self.chosen_station_start_time, self.chosen_station_end_time = self.evmodel.take_up_station(self.chosen_station, self.charge_start_time, self.charge_end_time)
            self.ev_time += self.charge_spent_time
            self.chosen_station_idx.append(self.chosen_station_id)
            self.chosen_station_idx.append(self.chosen_station_start_time)
            self.chosen_station_idx.append(self.chosen_station_end_time)

        if self.chosen_amount / self.chosen_speed > 1:
            self.finance_payment = self.chosen_init_park + self.chosen_add_park * (self.chosen_amount / self.chosen_speed - 1) + self.chosen_amount * self.chosen_price
        else:
            self.finance_payment = self.chosen_init_park + self.chosen_amount * self.chosen_price

        log_pay = math.log1p(self.finance_payment)
        log_time = math.log1p(self.charge_spent_time + self.dist_to_station / self.speed)
        self.cost = log_pay * self.alpha + self.beta * log_time

        self.finance_payment_idx.append(self.finance_payment)
        self.charge_spent_time_idx.append(self.charge_spent_time)
        self.cost_idx.append(self.cost)
        self.ev_time_idx.append(self.ev_time)
        self.soc_idx.append(self.current_soc)
        self.battery_idx.append(self.current_battery)

        self.status = 'finish charging'
        self.chosen_station_idx.append(self.chosen_station)
        self.status_idx.append(self.status)
        self.target_pos = self.dest
        self.travelled_dis = self.dist_to_station

    def fail_trip(self):
        self.status = 'fail'
        self.status_idx.append(self.status)
        self.trip_left = 0

    def calculate_reward(self):
        if self.status == 'fail':
            extra = -1
            self.threshold_impact = self.current_battery/self.full_battery- self.soc_threshold
            reward = 20* extra + 5 * self.threshold_impact
        else:
            if self.at_destination():
                extra = 1
            else:
                extra = 0

            log_whole_time = math.log1p(self.travelled_dis / self.speed + self.charge_spent_time)
            log_finance_payment = math.log1p(self.finance_payment)
            mn = self.charge_times/self.n_number
            self.cost_impact = ((self.beta * log_whole_time + self.alpha * log_finance_payment) + 1) ** mn
            self.threshold_impact = self.current_battery/self.full_battery- self.soc_threshold
            if self.charge_start_time is None:
                reward =  10/(self.cost_impact) + 20 * extra + 5 * self.threshold_impact
            else:
                self.time_impact = (math.log1p(self.charge_start_time - self.trip_start_time)) * (self.chance_charge + self.battery_chance +1)
                reward = 10/(self.cost_impact + self.time_impact) + 20 * extra + 5 * self.threshold_impact

        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = (reward + self.gamma * torch.max(self.nn_model(torch.FloatTensor(next_state).unsqueeze(0))).item())

            current_q_values = self.nn_model(torch.FloatTensor(state).unsqueeze(0))
            target_q_values = current_q_values.clone()
            target_q_values[0][action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def step(self):
        self.move_driver()

class EV_model_train(Model):
    def __init__(self, output_path:str, EV_agent: Optional[pd.DataFrame] = None,
                 station_location: Optional[dict] = None, station_info: Optional[dict] = None,
                 station_avail: Optional[dict] = None, station_node: Optional[dict] = None,
                 train_id_mapping_coor: Optional[dict] = None, graph: Optional[ig.Graph] = None,
                 networkx_graph: Optional[nx.Graph] = None, shortest_paths: Optional[dict] = None, shortest_paths_distances: Optional[dict] = None, 
                 base_model_path: Optional[str] = None):

        super().__init__()
        self.num_agents = len(EV_agent) if EV_agent is not None else 0
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.EV_agent = EV_agent
        self.station_location = station_location
        self.station_info = station_info
        self.station_avail = station_avail
        self.station_node = station_node
        self.train_id_mapping_coor = train_id_mapping_coor
        self.graph = graph
        self.networkx_graph = networkx_graph
        self.grid = NetworkGrid(self.networkx_graph)
        self.shortest_paths = shortest_paths
        self.shortest_paths_distances = shortest_paths_distances
        self.base_model_path = base_model_path

        self.location_list = list(station_location.values())
        self.tree = cKDTree(self.location_list)

        for index, row in EV_agent.iterrows():
            e = EVAgent_train(row['IndividualID'], state_size = 5, action_size = 11, vehicle_type = row['vehicle_type'], full_battery=row['full_battery'], 
                              init_soc=row['start_battery'], alpha=row['alpha_new'], trip_plan=row['trip_plan_new4'], soc_threshold=row['threshold_new'], 
                              consum_rate=row['consum_rate'], n_number = row['m_new'], chance_charge = row['chance_charge'], battery_chance = row['battery_chance'], 
                              evmodel=self, base_model_path=base_model_path)
            self.schedule.add(e)
            self.grid.place_agent(e, row['o_osmid_new'])
        
        '''self.data_collector = DataCollector(
            agent_reporters={
                'unique_id': lambda agent: agent.unique_id if agent.agent_type == 'DQNAgent' else None,
                'status_idx': lambda agent: agent.status_idx if agent.agent_type == 'DQNAgent' else None,
                'battery_idx': lambda agent: agent.battery_idx if agent.agent_type == 'DQNAgent' else None,
                'route_idx': lambda agent: agent.route_idx if agent.agent_type == 'DQNAgent' else None,
                'soc_idx': lambda agent: agent.soc_idx if agent.agent_type == 'DQNAgent' else None,
                'ev_time_idx': lambda agent: agent.ev_time_idx if agent.agent_type == 'DQNAgent' else None,
                'cost_idx': lambda agent: agent.cost_idx if agent.agent_type == 'DQNAgent' else None,
                'finance_payment_idx': lambda agent: agent.finance_payment_idx if agent.agent_type == 'DQNAgent' else None,
                'distance_idx': lambda agent: agent.distance_idx if agent.agent_type == 'DQNAgent' else None,
                'reward_idx': lambda agent: agent.reward_idx if agent.agent_type == 'DQNAgent' else None,
                'cost_impact_idx': lambda agent: agent.cost_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'time_impact_idx': lambda agent: agent.time_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'threshold_impact_idx': lambda agent: agent.threshold_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'charge_times': lambda agent: agent.charge_times if agent.agent_type == 'DQNAgent' else None}
        )'''

    def reset_for_new_episode(self):
        self.station_avail = {key: 0 for key in self.station_avail}
        for agent in self.schedule.agents:
            agent.reset_dynamic_state()
            #o_osmid = self.EV_agent.loc[self.EV_agent['IndividualID'] == agent.unique_id, 'o_osmid_new'].iloc[0]
            #self.grid.place_agent(agent, o_osmid)

        '''self.data_collector = DataCollector(
            agent_reporters={
                'unique_id': lambda agent: agent.unique_id if agent.agent_type == 'DQNAgent' else None,
                'status_idx': lambda agent: agent.status_idx if agent.agent_type == 'DQNAgent' else None,
                'battery_idx': lambda agent: agent.battery_idx if agent.agent_type == 'DQNAgent' else None,
                'route_idx': lambda agent: agent.route_idx if agent.agent_type == 'DQNAgent' else None,
                'soc_idx': lambda agent: agent.soc_idx if agent.agent_type == 'DQNAgent' else None,
                'ev_time_idx': lambda agent: agent.ev_time_idx if agent.agent_type == 'DQNAgent' else None,
                'cost_idx': lambda agent: agent.cost_idx if agent.agent_type == 'DQNAgent' else None,
                'finance_payment_idx': lambda agent: agent.finance_payment_idx if agent.agent_type == 'DQNAgent' else None,
                'distance_idx': lambda agent: agent.distance_idx if agent.agent_type == 'DQNAgent' else None,
                'reward_idx': lambda agent: agent.reward_idx if agent.agent_type == 'DQNAgent' else None,
                'cost_impact_idx': lambda agent: agent.cost_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'time_impact_idx': lambda agent: agent.time_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'threshold_impact_idx': lambda agent: agent.threshold_impact_idx if agent.agent_type == 'DQNAgent' else None,
                'charge_times': lambda agent: agent.charge_times if agent.agent_type == 'DQNAgent' else None}
        )'''

    def is_episode_ended(self):
        return all(agent.at_destination() or agent.trip_left ==0 for agent in self.schedule.agents)

    def step(self, station_avail):
        self.station_avail = station_avail
        self.schedule.step()

    def run(self, steps: int):
        self.data_collector.collect(self)
        for k in range(steps):
            self.step()
        return self.data_collector.get_agent_vars_dataframe()

    # Additional functions for simulation
    def get_shortest_path_length(self, start_pos, end_pos):
        if start_pos == end_pos:
            return 0  
        if (start_pos, end_pos) in self.shortest_paths_distances:
            shortest_path_length = self.shortest_paths_distances[(start_pos, end_pos)]
            return shortest_path_length
        else:
            print(f'Station Error: cannot find the shortest path length between {start_pos} and {end_pos}')

    def get_candidate_stations(self, current_node, current_coor):
        _, indices = self.tree.query(current_coor, k=5)
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            indices = [indices]
        all_candidate_stations = [list(self.station_location.keys())[i] for i in indices]
        all_candidate_stations_node = [self.station_node[key] for key in all_candidate_stations if key in self.station_node]
        all_candidate_stations_node = [int(item) for item in all_candidate_stations_node]

        all_candidate_stations_distances = []
        for node in all_candidate_stations_node:
            distance = self.get_shortest_path_length(current_node, node)
            all_candidate_stations_distances.append(distance)
        distance_dict = dict(zip(all_candidate_stations, all_candidate_stations_distances))

        return all_candidate_stations, distance_dict

    def get_station_avail(self, ev_time, station_id):
        rounded_ev_time = ((ev_time + 14) // 15) * 15
        station_int = int(station_id.replace('_', ''))
        availability_key = f"{station_int}_{int(rounded_ev_time)}"
        avail = self.station_avail.get(availability_key)
        return avail

    def get_queue_finish_time(self, ev_time, station_id):
        port_id = int(station_id.replace('_', ''))
        rounded_ev_time = int(((ev_time + 14) // 15) * 15)
        for time_slot in range(rounded_ev_time, 14400 + 1, 15):
            availability_key = f"{port_id}_{time_slot}"
            if self.station_avail.get(availability_key) == 0:
                return time_slot

    def take_up_station(self, station_id, ev_time, expect_charge_end_time):
        port_id = int(station_id.replace('_', ''))
        rounded_ev_time = int(((ev_time + 14) // 15) * 15)
        rounded_end_time = int(((expect_charge_end_time + 14) // 15) * 15)
        rounded_end_time += 15
        for time in range(rounded_ev_time, rounded_end_time, 15):
            availability_key = f"{port_id}_{time}"
            self.station_avail[availability_key] = 1 
        return port_id, rounded_ev_time, rounded_end_time

NUM_AGENTS = 10
NUM_EPISODES = 2000

def save_policy(agent, episode, vehicle_type): #training
    policy_file = fr"/home/ec2-user/AWS_project/train_results/V{vehicle_type}/V{vehicle_type}_episode_{episode}.pth"
    torch.save(agent.nn_model.state_dict(), policy_file)

def find_policy(vehicle_type, episode): #simulation 
    policy_file = fr"/home/ec2-user/AWS_project/train_results/V{vehicle_type}/V{vehicle_type}_episode_{episode}.pth"
    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"Policy file not found: {policy_file}")
    return policy_file

def load_pretrained_models(unique_vehicle_types, episode): #simulation 
    pretrained_model_state_dict = {}
    for vehicle_type in unique_vehicle_types:
        try:
            policy_file = find_policy(vehicle_type, episode)
            pretrained_model_state = torch.load(policy_file, map_location=torch.device('cpu'))
            pretrained_model_state_dict[vehicle_type] = pretrained_model_state
        except FileNotFoundError as e:
            print(f"Error in loading policy for {vehicle_type}: {str(e)}")
    return pretrained_model_state_dict

def update_station_avail(station_avail, chosen_stations):
    for station_id, start_slot, end_slot in chosen_stations:
        for time_slot in range(start_slot, end_slot +15, 15):
            key = f"{station_id}_{time_slot}"
            station_avail[key] = 0
    return station_avail

def load_agent_data(agent_id, episode):# main loop
    df = pd.read_pickle(f'agent_{agent_id}_episode_{episode}_data.pkl')
    return df

def get_training_ev_chosen_stations(df):
    base_ids = ['2022000381', '2022000442', '2022001894', '2022002390', 
                '2022003296', '2022005127', '2022006409', '2022007885', 
                '202200800', '2022009111']
    all_chosen_stations = []
    for base_id in base_ids:
        matching_rows = df[df['unique_id'].astype(str).str.startswith(base_id)]
        for _, row in matching_rows.iterrows():
            if isinstance(row['chosen_station_idx'], list) and row['chosen_station_idx'] !=[]:
                all_chosen_stations.append(row['chosen_station_idx'])
    return all_chosen_stations


graph = ig.Graph.Read_GraphML(r'/home/ec2-user/AWS_project/data/GB_network_relabeled_H1.graphml')
networkx_graph = ig.Graph.to_networkx(graph)

with open(r'/home/ec2-user/AWS_project/data/station_avail_1day.json', 'r') as file:
    station_avail = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/station_location.json', 'r') as file:
    station_location = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/station_info_incl_park.json', 'r') as file:
    station_info = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/station_node.json', 'r') as file:
    station_node = json.load(file)   
with open(r'/home/ec2-user/AWS_project/data/id_mapping_H1.json', 'r') as file:
    id_mapping_coor = json.load(file)

EV_agent = pd.read_json(r'/home/ec2-user/AWS_project/data/all_ev_10cluster_update_0709_Novnew.json', orient='records', lines=True)

def modify_duplicate_ids(EV_agent):
    modified_EV_agent = EV_agent.copy()
    duplicates = modified_EV_agent[modified_EV_agent.duplicated('IndividualID', keep=False)]
    grouped = duplicates.groupby('IndividualID')
    def modify_ids(group):
        if len(group) > 1:
            original_id = group['IndividualID'].iloc[0]
            for i, idx in enumerate(group.index[1:], start=1):
                modified_EV_agent.loc[idx, 'IndividualID'] = int(f"{original_id}{i}")
        return group

    grouped.apply(modify_ids)
    assert modified_EV_agent['IndividualID'].nunique() == len(modified_EV_agent), "IDs are still not unique"
    
    return modified_EV_agent

EV_agent_unique = modify_duplicate_ids(EV_agent)

with open(r"/home/ec2-user/AWS_project/data/updated_paths3.pkl", 'rb') as f:
    simulate_shortest_paths = pickle.load(f)
with open(r"/home/ec2-user/AWS_project/data/updated_distances3.pkl", 'rb') as f:
    simulate_shortest_path_distances = pickle.load(f)

unique_vehicle_types = [0,1,2,3,4,5,6,7,8,9]

train_graph = ig.Graph.Read_GraphML(r'/home/ec2-user/AWS_project/data/train_graph_H1_0709.graphml')
networkx_graph_train = ig.Graph.to_networkx(train_graph)
train_EV_agent = pd.read_json(r'/home/ec2-user/AWS_project/data/train_ev_10cluster_update_map_0709_Novnew.json', orient='records', lines=True)

train_EV_agent0 = train_EV_agent[0:1]
train_EV_agent1 = train_EV_agent[1:2]
train_EV_agent2 = train_EV_agent[2:3]
train_EV_agent3 = train_EV_agent[3:4]
train_EV_agent4 = train_EV_agent[4:5]
train_EV_agent5 = train_EV_agent[5:6]
train_EV_agent6 = train_EV_agent[6:7]
train_EV_agent7 = train_EV_agent[7:8]
train_EV_agent8 = train_EV_agent[8:9]
train_EV_agent9 = train_EV_agent[9:10]

base_model_path0 = r'/home/ec2-user/AWS_project/data/V0_baseline.pth'
base_model_path1 = r'/home/ec2-user/AWS_project/data/V1_baseline.pth'
base_model_path2 = r'/home/ec2-user/AWS_project/data/V2_baseline.pth'
base_model_path3 = r'/home/ec2-user/AWS_project/data/V3_baseline.pth'
base_model_path4 = r'/home/ec2-user/AWS_project/data/V4_baseline.pth'
base_model_path5 = r'/home/ec2-user/AWS_project/data/V5_baseline.pth'
base_model_path6 = r'/home/ec2-user/AWS_project/data/V6_baseline.pth'
base_model_path7 = r'/home/ec2-user/AWS_project/data/V7_baseline.pth'
base_model_path8 = r'/home/ec2-user/AWS_project/data/V8_baseline.pth'
base_model_path9 = r'/home/ec2-user/AWS_project/data/V9_baseline.pth'

with open(r'/home/ec2-user/AWS_project/data/train_station_avail_0709.json', 'r') as file:
    train_station_avail = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/train_station_location_0709.json', 'r') as file:
    train_station_location = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/train_station_info_0709.json', 'r') as file:
    train_station_info = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/train_station_node_0709.json', 'r') as file:
    train_station_node = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/train_id_mapping.json', 'r') as file:
    train_id_mapping = json.load(file)
with open(r'/home/ec2-user/AWS_project/data/train_id_mapping_coor.json', 'r') as file:
    train_id_mapping_coor = json.load(file)
    
new_station_node = {}
for station_id, original_node_id_str in train_station_node.items():
    if original_node_id_str in train_id_mapping:
        new_node_id = train_id_mapping[original_node_id_str]
        new_station_node[station_id] = str(new_node_id)

with open(r'/home/ec2-user/AWS_project/data/train_shortest_paths_distances_update2.pkl', 'rb') as f:
    train_shortest_paths_distances = pickle.load(f)
with open(r'/home/ec2-user/AWS_project/data/train_shortest_paths_updated1.pkl', 'rb') as f:
    train_shortest_paths = pickle.load(f)

print('start load model code') ##################################################################################################################

EVmodel_sim = EV_model_sim('simulation',
                EV_agent=EV_agent_unique,
                station_location=station_location,
                station_info=station_info,
                station_avail=station_avail.copy(),
                station_node=station_node,
                train_id_mapping_coor=id_mapping_coor,
                graph=graph,
                networkx_graph=networkx_graph,
                shortest_paths=simulate_shortest_paths, 
                shortest_paths_distances=simulate_shortest_path_distances)

current_station_avail = station_avail.copy()

EVmodel_train0 = EV_model_train('trail0', EV_agent=train_EV_agent0, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path0)

EVmodel_train1 = EV_model_train('trail1', EV_agent=train_EV_agent1, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path1)

EVmodel_train2 = EV_model_train('trail2', EV_agent=train_EV_agent2, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path2)

EVmodel_train3 = EV_model_train('trail3', EV_agent=train_EV_agent3, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path3)

EVmodel_train4 = EV_model_train('trail4', EV_agent=train_EV_agent4, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path4)

EVmodel_train5 = EV_model_train('trail5', EV_agent=train_EV_agent5, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path5)

EVmodel_train6 = EV_model_train('trail6', EV_agent=train_EV_agent6, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path6)

EVmodel_train7 = EV_model_train('trail7', EV_agent=train_EV_agent7, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path7)

EVmodel_train8 = EV_model_train('trail8', EV_agent=train_EV_agent8, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path8)

EVmodel_train9 = EV_model_train('trail9', EV_agent=train_EV_agent9, station_location=train_station_location, station_info=train_station_info,
                   station_avail = current_station_avail.copy(), station_node=new_station_node, train_id_mapping_coor= train_id_mapping_coor,
                   graph=train_graph, networkx_graph=networkx_graph_train, shortest_paths=train_shortest_paths, shortest_paths_distances=train_shortest_paths_distances, base_model_path= base_model_path9)

print('model loaded') 
print('start running') 
 
for episode in tqdm(range(NUM_EPISODES)):

    '''EVmodel_train0.station_avail = current_station_avail.copy()
    EVmodel_train1.station_avail = current_station_avail.copy()
    EVmodel_train2.station_avail = current_station_avail.copy()
    EVmodel_train3.station_avail = current_station_avail.copy()
    EVmodel_train4.station_avail = current_station_avail.copy()
    EVmodel_train5.station_avail = current_station_avail.copy()
    EVmodel_train6.station_avail = current_station_avail.copy()
    EVmodel_train7.station_avail = current_station_avail.copy()
    EVmodel_train8.station_avail = current_station_avail.copy()
    EVmodel_train9.station_avail = current_station_avail.copy()'''

    print('start training')
    EVmodel_train0.step(current_station_avail.copy())
    for agent in EVmodel_train0.schedule.agents:
            save_policy(agent, episode, 0)
    EVmodel_train0.reset_for_new_episode()

    EVmodel_train1.step(current_station_avail.copy())
    for agent in EVmodel_train1.schedule.agents:
            save_policy(agent, episode, 1)
    EVmodel_train1.reset_for_new_episode()

    EVmodel_train2.step(current_station_avail.copy())
    for agent in EVmodel_train2.schedule.agents:
            save_policy(agent, episode, 2)
    EVmodel_train2.reset_for_new_episode()

    EVmodel_train3.step(current_station_avail.copy())
    for agent in EVmodel_train3.schedule.agents:
            save_policy(agent, episode, 3)
    EVmodel_train3.reset_for_new_episode()

    EVmodel_train4.step(current_station_avail.copy())
    for agent in EVmodel_train4.schedule.agents:
            save_policy(agent, episode, 4)
    EVmodel_train4.reset_for_new_episode()

    EVmodel_train5.step(current_station_avail.copy())
    for agent in EVmodel_train5.schedule.agents:
            save_policy(agent, episode, 5)
    EVmodel_train5.reset_for_new_episode()

    EVmodel_train6.step(current_station_avail.copy())
    for agent in EVmodel_train6.schedule.agents:
            save_policy(agent, episode, 6)
    EVmodel_train6.reset_for_new_episode()

    EVmodel_train7.step(current_station_avail.copy())
    for agent in EVmodel_train7.schedule.agents:
            save_policy(agent, episode, 7)
    EVmodel_train7.reset_for_new_episode()

    EVmodel_train8.step(current_station_avail.copy())
    for agent in EVmodel_train8.schedule.agents:
            save_policy(agent, episode, 8)
    EVmodel_train8.reset_for_new_episode()

    EVmodel_train9.step(current_station_avail.copy())
    for agent in EVmodel_train9.schedule.agents:
            save_policy(agent, episode, 9)
    EVmodel_train9.reset_for_new_episode()

    print('Start simulation')
    model_state_dict = load_pretrained_models(unique_vehicle_types, episode)
    df = EVmodel_sim.step(model_state_dict)
    if episode == 0 or episode in [i for i in range(0, 1001, 50)]:
        df.to_csv(f'/home/ec2-user/AWS_project/simulation_results/episode_{episode}_data.csv', index=False)
    
    updated_station_avail = EVmodel_sim.station_avail
    chosen_stations = get_training_ev_chosen_stations(df)
    with open(f'/home/ec2-user/AWS_project/simulation_results/station_avail_episode_{episode}.pkl', 'wb') as f:
        pickle.dump(updated_station_avail, f)

    if chosen_stations != []:
        modified_station_avail = update_station_avail(updated_station_avail.copy(), chosen_stations)
        current_station_avail = modified_station_avail
    else:
        current_station_avail = updated_station_avail.copy()
