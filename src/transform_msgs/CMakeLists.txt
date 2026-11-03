cmake_minimum_required(VERSION 2.8.3)
project(transform_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    message_generation
    std_msgs
)

add_message_files(DIRECTORY msg
  FILES
    acu_new_ioniqw_TX.msg
    vehicle_frenet_info.msg
    frenet_path.msg
    frenet_control.msg
    frenet_obstacle.msg
    local_refpath.msg
    obstacle.msg
    obstacle_drone.msg
    mppi_time.msg
    trajectory_sequence.msg
    gain.msg
    error_for_controller.msg
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
