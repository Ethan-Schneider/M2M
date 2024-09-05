Router communication.
At every iteration, Router Service should receive the following messages:
LevelTopologyUpdate – configuration of deck, driveways and aisles 
BotModelUpdates – bot dimensions and max speed 
BotStatusUpdates – pose, speed, charge level, action queue for each bot 
SpeedLimitRegionsUpdate - max speed within rectangular regions 
RouteRequestUpdate – this triggers Router replan cycle
The RouteRequestUpdate message should be sent after all other types of messages. Router creates snapshot of all received data and starts to create responses when RouteRequestUpdate message is received.

Some iteration cycles also may have the following messages: 
ObstacleUpdates – rectangular regions where bots are not allowed to enter 
PlanRejections – rejected plans by LC from previous iteration 
ConfigurationUpdates, ObstacleRemoved, BotRemoved – even more rare 

Messages which Router sends
After aggregating inputs and planning new routes for bots, Router sends the following messages: 
VisualizationData – data about bot states and route reservations for export to “Replayer Files” (reproducing animation of bot movements)
TeamPlan – collection of new waypoints and target goal region for all bots 

Sometimes, Router also sends: 
ReachabilityDataUpdate – data about what regions are reachable and which are not 
ServiceStatusUpdate, ServiceStatusAcknowledged – rare, mostly useful for starting up the Router 


Typical communication with Router:

Level control ----> LevelTopologyUpdate 	----> Router service
Level control ----> BotModelUpdates 		----> Router service
Level control ----> BotStatusUpdates 		----> Router service
Level control ----> SpeedLimitRegionsUpdate 	----> Router service
Level control ----> RouteRequestUpdate 		----> Router service
Router service aggregates data and computes responses
Level control <---- VisualizationData 		<---- Router service
Level control <---- TeamPlan 			<---- Router service

 