cmake_minimum_required(VERSION 2.8.3)
project(drone_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    message_generation
    std_msgs
)

add_message_files(DIRECTORY msg
  FILES
  Determined_traj.msg
  Sampled_traj.msg
  Ego_state.msg
  Guidance.msg
  obstacle_drone.msg
  obstacle_polygon.msg
  Ego_goal_setting.msg
  Categorized_traj.msg
  Cal_time.msg
  MPCControlSeq.msg
  Setup.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
)
