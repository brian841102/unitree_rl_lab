// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

// SMP Steering command: [tar_dir_x, tar_dir_y, tar_speed, face_dir_x, face_dir_y]
// In the robot's local heading frame (x = forward, y = left; yaw-invariant).
// Joystick mapping:
//   Left stick  = 2D travel velocity: up→forward, down→backward, left/right→strafe.
//                 Stick magnitude sets the speed (full deflection = tar_speed_max).
//   Right stick x = face-direction / turn command: right→turn right, left→turn left.
REGISTER_OBSERVATION(steering_commands)
{
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["steering"]["ranges"];
    const float tar_speed_max = cfg["tar_speed_max"][0].as<float>();

    // ---- Left stick -> target travel direction + speed (heading frame) ----
    float vx = joystick->ly();          // up  = +forward
    float vy = -joystick->lx();         // left = +y (strafe left)
    float mag = std::sqrt(vx * vx + vy * vy);

    constexpr float kMoveDeadzone = 0.1f;   // matches training stand_speed_threshold
    float tar_speed, dir_x, dir_y;
    if (mag < kMoveDeadzone) {// No velocity command -> stand still. tar_dir is arbitrary at zero speed.
        tar_speed = 0.0f;
        dir_x = 1.0f;   // default forward
        dir_y = 0.0f;
    } else {
        dir_x = vx / mag;
        dir_y = vy / mag;
        tar_speed = std::min(mag, 1.0f) * tar_speed_max;  // full stick = max speed
    }

    // ---- Right stick x -> face-direction turn command (heading frame) ----
    // Limited to +/-90 deg so extremes never wrap past 180 deg (which cause hard-left cmd into a right turn)
    constexpr float kMaxFaceAngle = M_PI / 2.0f;
    constexpr float kTurnDeadzone = 0.05f;
    float rx = joystick->rx();
    if (std::abs(rx) < kTurnDeadzone) rx = 0.0f;
    float face_angle = -std::clamp(rx, -1.0f, 1.0f) * kMaxFaceAngle;

    std::vector<float> obs(5);
    // Target direction in local heading frame
    obs[0] = dir_x;
    obs[1] = dir_y;
    // Target speed
    obs[2] = tar_speed;
    // Face direction in local heading frame
    obs[3] = std::cos(face_angle);
    obs[4] = std::sin(face_angle);
    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

}
}