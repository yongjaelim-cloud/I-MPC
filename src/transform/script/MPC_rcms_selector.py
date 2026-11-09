#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional global 4-seed RCMS coordinator.

Input: /mpc_rcms_control_seq0..3 (MPCControlSeq)
Output: /mpc_rcms_control_seq (MPCControlSeq)
Policy: collision-free avoidance always wins over mitigation.
This coordinator DOES NOT send actuator commands; the existing tracker/ego_mover
must explicitly subscribe to the output and accept mitigation-mode messages.
"""
import time
import threading

import rospy
from drone_msgs.msg import MPCControlSeq
from std_msgs.msg import String


class RCMSSelector:
    def __init__(self):
        self.lock=threading.Lock()
        self.cache={}
        self.last_key=None
        self.max_age_s=float(rospy.get_param('~max_age_s',0.20))
        self.rate_hz=float(rospy.get_param('~hz',20.))
        self.pub=rospy.Publisher('/mpc_rcms_control_seq',MPCControlSeq,queue_size=10)
        self.mode_pub=rospy.Publisher('/mpc_rcms_mode',String,queue_size=10)
        self.subs=[]
        for i in range(4):
            self.subs.append(rospy.Subscriber('/mpc_rcms_control_seq%d'%i,
                                             MPCControlSeq,self.callback,callback_args=i,queue_size=1))

    @staticmethod
    def kind(msg):
        stage=str(getattr(msg,'trajectory_stage',''))
        typ=int(getattr(msg,'trajectory_type',-1))
        if stage=='rcms_avoidance' or (not stage and typ==1):
            return 'AVOIDANCE'
        if stage=='rcms_mitigation' or (not stage and typ==2):
            return 'MITIGATION'
        return 'FAILED'

    def callback(self,msg,i):
        with self.lock:
            self.cache[int(i)]=(time.monotonic(),msg)

    def step(self):
        with self.lock:
            now=time.monotonic()
            options=[(i,m) for i,(t,m) in self.cache.items()
                     if now-t<=self.max_age_s and self.kind(m)!='FAILED']
        if not options:
            return
        def rank(it):
            i,msg=it
            stage_priority=0 if self.kind(msg)=='AVOIDANCE' else 1
            j=float(getattr(msg,'global_cost',getattr(msg,'cost',float('inf'))))
            if not (-1.e200<j<1.e200):
                j=float('inf')
            return (stage_priority,j,i)
        winner_i,winner_msg=min(options,key=rank)
        key=(int(winner_i),int(getattr(winner_msg,'seq_id',-1)),self.kind(winner_msg))
        if key == self.last_key:
            return
        self.last_key=key
        self.pub.publish(winner_msg)
        self.mode_pub.publish(String(data=self.kind(winner_msg)))
        rospy.loginfo_throttle(1.,'[rcms_selector] planner=%d seq=%d mode=%s',
                               winner_i,key[1],key[2])


def main():
    rospy.init_node('MPC_rcms_selector',anonymous=False)
    selector=RCMSSelector()
    rate=rospy.Rate(selector.rate_hz)
    while not rospy.is_shutdown():
        selector.step()
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break

if __name__=='__main__':
    main()
