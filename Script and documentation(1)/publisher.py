#!/usr/bin/env python

import pika
import sys
import json

from pika import BasicProperties
from rabbitmq import RabbitMQ

rabbitmq = RabbitMQ()
exchange_name = "Symbotic.Routing"
request_time_utc = "2024-06-03T09:39:31.577224Z"

#Symbotic.Routing.Event.SpeedLimitRegions.Updated
routing_key = "Symbotic.Routing.Event.SpeedLimitRegions.Updated.Level-1"
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.Routing.Event.SpeedLimitRegions.Updated'})
message = {
    "SpeedLimitRegions":
    [
        {
            "Bounds":  { "BottomLeft": { "XMm": 0, "YMm": 0 }, "TopRight": { "XMm": 105656, "YMm": 7480 } }, #[0,0,105656,7480]
            "SpeedLimit": { "MmPerSec": 4000 }
        }
    ],
    #[0,0,105656,7480]
    "DeckBoundaries":
    [
        {
            "BottomLeft": { "XMm": 0, "YMm": 0 },
            "TopRight": { "XMm": 105656, "YMm": 7480 },
        }
    ]
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))

# Symbotic.SystemModel.Event.Level.Topology.Updated
routing_key = 'Symbotic.SystemModel.Event.Level.Topology.Updated.Level-1'
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.SystemModel.Event.Level.Topology.Updated'})
message = {
    "DeckBoundary": 
    {
        "BottomLeft": { "XMm": 0, "YMm": 0 },
        "TopRight": { "XMm": 105656, "YMm": 7480 },
    },
    #All resource boundaries: [1675,8063,2437,146676],[3751,8063,4513,146676],[5828,8063,6590,146676],[7904,8063,8666,146676],[9981,8063,10743,146676],[12057,8063,12819,146676],[14134,8063,14896,146676],[16210,8063,16972,146676],[18287,8063,19049,146676],[20363,8063,21125,146676],[22439,8063,23201,146676],[24516,8063,25278,146676],[26592,8063,27354,146676],[28669,8063,29431,146676],[30745,8063,31507,146676],[32822,8063,33584,146676],[34898,8063,35660,146676],[36975,8063,37737,146676],[39051,8063,39813,146676],[41127,8063,41889,146676],[43204,8063,43966,146676],[45280,8063,46042,146676],[47357,8063,48119,146676],[49433,8063,50195,146676],[51510,8063,52272,146676],[53586,8063,54348,146676],[55663,8063,56425,146676],[57739,8063,58501,146676],[63207,8063,63969,146676],[65283,8063,66045,146676],[67360,8063,68122,146676],[69436,8063,70198,146676],[71513,8063,72275,146676],[73589,8063,74351,146676],[75666,8063,76428,146676],[77742,8063,78504,146676],[79819,8063,80581,146676],[81895,8063,82657,146676],[83971,8063,84733,146676],[86048,8063,86810,146676],[88124,8063,88886,146676],[90201,8063,90963,146676],[92277,8063,93039,146676],[94354,8063,95116,146676],[96430,8063,97192,146676],[98507,8063,99269,146676],[100583,8063,101345,146676],[102659,8063,103421,146676],[6352,-5188,7114,-583],[14905,-7067,15667,-583],[19363,-7067,20125,-583],[9436,-7067,10198,-583],[13893,-7067,14655,-583],[25854,-7067,26616,-583],[30327,-7067,31089,-583],[20360,-7067,21122,-583],[24858,-7067,25620,-583],[33720,-6762,34482,-583],[41798,-6762,42560,-583],[48212,-6762,48974,-583],[56329,-6762,57091,-583],[64207,-7067,64969,-583],[68690,-7067,69452,-583],[58707,-7067,59469,-583],[63195,-7067,63957,-583],[75181,-7067,75943,-583],[79664,-7067,80426,-583],[69702,-7067,70464,-583],[74169,-7067,74931,-583],[81290,-6762,82052,-583],[89392,-6762,90154,-583],[95575,-6762,96337,-583],[103684,-6762,104446,-583]]
    "ResourceBoundaries":
    [
        # Aisles
        { "BottomLeft": { "XMm": 1675, "YMm": 8063 }, "TopRight": { "XMm": 2437, "YMm": 146676 } }, #[1675,8063,2437,146676]
        { "BottomLeft": { "XMm": 1675, "YMm": 8063 }, "TopRight": { "XMm": 2437, "YMm": 146676 } }, #[3751,8063,4513,146676]
        { "BottomLeft": { "XMm": 5828, "YMm": 8063 }, "TopRight": { "XMm": 6590, "YMm": 146676 } }, #[5828,8063,6590,146676]
        { "BottomLeft": { "XMm": 7904, "YMm": 8063 }, "TopRight": { "XMm": 8666, "YMm": 146676 } }, #[7904,8063,8666,146676]
        { "BottomLeft": { "XMm": 9981, "YMm": 8063 }, "TopRight": { "XMm": 10743, "YMm": 146676 } }, #[9981,8063,10743,146676]
        { "BottomLeft": { "XMm": 12057, "YMm": 8063 }, "TopRight": { "XMm": 12819, "YMm": 146676 } }, #[12057,8063,12819,146676]
        #...
        { "BottomLeft": { "XMm": 39051, "YMm": 8063 }, "TopRight": { "XMm": 39813, "YMm": 146676 } }, #[39051,8063,39813,146676]
        # Driveways
        { "BottomLeft": { "XMm": 6352, "YMm": -5188 }, "TopRight": { "XMm": 7114, "YMm": -583 } }, #[6352,-5188,7114,-583]
        { "BottomLeft": { "XMm": 14905, "YMm": -7067 }, "TopRight": { "XMm": 15667, "YMm": -583 } }, #[14905,-7067,15667,-583]
        { "BottomLeft": { "XMm": 19363, "YMm": -7067 }, "TopRight": { "XMm": 20125, "YMm": -583 } }, #[19363,-7067,20125,-583]
        { "BottomLeft": { "XMm": 9436, "YMm": -7067 }, "TopRight": { "XMm": 10198, "YMm": -583 } }, #[9436,-7067,10198,-583]
        { "BottomLeft": { "XMm": 13893, "YMm": -7067 }, "TopRight": { "XMm": 14655, "YMm": -583 } }, #[13893,-7067,14655,-583]
        { "BottomLeft": { "XMm": 25854, "YMm": -7067 }, "TopRight": { "XMm": 26616, "YMm": -583 } }, #[25854,-7067,26616,-583]
        { "BottomLeft": { "XMm": 25854, "YMm": -7067 }, "TopRight": { "XMm": 26616, "YMm": -583 } }, #[25854,-7067,26616,-583]
        { "BottomLeft": { "XMm": 30327, "YMm": -7067 }, "TopRight": { "XMm": 31089, "YMm": -583 } }, #[30327,-7067,31089,-583]
        #{ "BottomLeft": { "XMm": , "YMm":  }, "TopRight": { "XMm": , "YMm":  } }, #
    ],
    #[0,0,105656,7480]
    "DeckBoundaries":
    [
        {
            "BottomLeft": { "XMm": 0, "YMm": 0 },
            "TopRight": { "XMm": 105656, "YMm": 7480 },
        }
    ]
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))

# Symbotic.SystemState.Event.Level.Obstacle.Updated
routing_key = 'Symbotic.SystemState.Event.Level.Obstacle.Updated.Level-1.Obstacle-1'
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.SystemState.Event.Level.Obstacle.Updated'})
message = {
    "Obstacle": 
    {
        "Id": 1,
        "Bounds":
        {
            "BottomLeft": { "XMm": 16000, "YMm": 10000 },
            "TopRight": { "XMm": 17000, "YMm": 20000 },
        }
    }
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))

# Bot models.
routing_key = 'Symbotic.SystemModel.Event.Bot.Updated.Level-1.Bot-1'
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.SystemModel.Event.Bot.Updated'})
message = {
    "BotModel": 
    {
        "Id": 1,
        "OriginToRearLeftCornerOffset": { "XMm": -278, "YMm": 389 },
        "OriginToFrontRightCornerOffset": { "XMm": 1165, "YMm": -389 },
        "MaxAcceleration": { "MmPerSecSquared": 1500 },
        "MaxDeceleration": { "MmPerSecSquared": 1500 },
        "MaxSpeed": { "MmPerSec": 10000 }
    }
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))

# Bot statuses.
routing_key = 'Symbotic.SystemState.Event.Bot.Updated.Level-1.Bot-1'
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.SystemState.Event.Bot.Updated'})
message = {
    "BotStatus": 
    {
        "Id": 1,
        "UpdateTimeUTC": request_time_utc,
        "Pose": { "Position": { "XMm": 30708, "YMm": -6306 }, "Yaw": { "MRad": 1571 } },
        "LinearVelocity": { "MmPerSec": 0 },
        "ActionQueue": [],
        "Charge": { "MilliVolts": 46300 },
        "OperationalState": "Idle",
    }
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))

#Symbotic.Routing.Event.RouteRequests.Updated
routing_key = 'Symbotic.Routing.Event.RouteRequests.Updated.Level-1'
properties = BasicProperties(headers={'X-Message-Type': 'Symbotic.Routing.Event.RouteRequests.Updated'})
message = {
    'RequestTimeUTC': request_time_utc,
    "RouteConstraints":
    [
        {
            "WorkItemId":1,
            "BotId":1,
            'TimeEligibleForCommandUTC': request_time_utc,
            "SegmentConstraints":
            [
                {
                    "TaskId":1,
                    "SegmentId":1,
                    "CandidateGoalRegions":
                    [
                        {
                            "Bounds": { "BottomLeft": { "XMm": "39357", "YMm": "99112" }, "TopRight": { "XMm": "39507", "YMm": "99262" } },
                            "CandidateOrientations": [{"MRad": 1571}],
                        }
                    ],
                    "DestinationSpaceReservation": { "BottomLeft": { "XMm": 39043, "YMm": 98834 }, "TopRight": { "XMm": 39821, "YMm": 100427 }},
                    "DestinationDwellDuration": { "MSec": 18000 },
                    "SegmentSpeedLimit": { "MmPerSec": 0 },
                    "PrerequisiteSegmentIds": []
                }
            ]
        } 
    ]
}

rabbitmq.publish(exchange_name, routing_key, properties, json.dumps(message))
print(f" [x] Sent")