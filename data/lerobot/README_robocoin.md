# RoboCOIN Action Schemas

Generated from `/root/nasbak/cjy/robot_raw/robocoin/RoboCOIN` on 2026-07-13 UTC.

This file documents the meaning of the raw `action` vector for the local RoboCOIN datasets. The action schema is not globally fixed: each dataset declares its own action fields in `meta/info.json -> features.action.names`.

## Summary

- Dataset roots scanned: `738`
- Unique action schemas: `21`
- Units are encoded in field names: `_rad` means radians, `_rad_s` means radians per second, `_m` means meters, and quaternion fields are unitless quaternion components.
- `*_gripper_open` is a scalar gripper opening value; `*_hand_joint_*` fields are dexterous-hand joint values, not discrete open/close labels.
- Training code may pad heterogeneous action vectors to a larger target dimension and use `action_mask`; the indices below describe the raw local `action` vector before padding.

## Schema Index

| # | Robot Type | Shape | Count | Example datasets |
|---:|---|---:|---:|---|
| 1 | `Airbot_MMK2` | `36` | 143 | `Airbot_MMK2_click_pen`, `Airbot_MMK2_close_door_left`, `Airbot_MMK2_close_door_right`, `Airbot_MMK2_close_doors`, `Airbot_MMK2_close_drawer` |
| 2 | `Agilex_Cobot_Magic` | `26` | 72 | `Agilex_Cobot_Magic_classify_objects_eight`, `Agilex_Cobot_Magic_classify_objects_six`, `Agilex_Cobot_Magic_close_drawer_bottom`, `Agilex_Cobot_Magic_close_drawer_top`, `Agilex_Cobot_Magic_close_drawer_upper` |
| 3 | `discover_robotics_aitbot_mmk2` | `36` | 70 | `AIRBOT_MMK2_beauty_sponge_and_cake_to_place`, `AIRBOT_MMK2_bowl_storage_pepper`, `AIRBOT_MMK2_boxs_storage`, `AIRBOT_MMK2_building_block_storage`, `AIRBOT_MMK2_cake_storage` |
| 4 | `galaxea_r1_lite` | `14` | 52 | `R1_Lite_boil_water_in_a_kettle`, `R1_Lite_catch_the_water`, `R1_Lite_clean_the_floor`, `R1_Lite_clean_the_sink`, `R1_Lite_clean_toilet` |
| 5 | `leju_robot` | `54` | 49 | `leju_robot_box_storage_parcel`, `leju_robot_box_storage_parcel_a`, `leju_robot_box_storage_parcel_b`, `leju_robot_box_storage_parcel_c`, `leju_robot_box_storage_parcel_d` |
| 6 | `agilex_cobot_decoupled_magic` | `26` | 40 | `Agilex_Cobot_Magic_basket_storage_banana`, `Agilex_Cobot_Magic_basket_storage_bread`, `Agilex_Cobot_Magic_basket_storage_egg_yolk_pastry`, `Agilex_Cobot_Magic_basket_storage_long_bread`, `Agilex_Cobot_Magic_basket_storage_orange` |
| 7 | `agilex_cobot_decoupled_magic` | `14` | 39 | `Agilex_Cobot_Magic_fold_clothes`, `Agilex_Split_Aloha_box_storage_chopsticks`, `Agilex_Split_Aloha_cap_the_pen_a`, `Agilex_Split_Aloha_catch_the_ball`, `Agilex_Split_Aloha_classification_of_fruits_and_vegetables` |
| 8 | `Galaxea_R1_Lite` | `14` | 39 | `Galaxea_R1_Lite_arrange_baai_then_brain`, `Galaxea_R1_Lite_change_baai_into_brain`, `Galaxea_R1_Lite_classify_object_five`, `Galaxea_R1_Lite_classify_object_four`, `Galaxea_R1_Lite_classify_object_green_tablecloth` |
| 9 | `AI2_Alphabot_2` | `34` | 37 | `AI2_Alphabot_2_arrange_teaset`, `AI2_Alphabot_2_clink_glasses`, `AI2_Alphabot_2_grind_ink`, `AI2_Alphabot_2_heat_test_tube`, `AI2_Alphabot_2_insert_hose_into_hole` |
| 10 | `unknown` | `30` | 35 | `G1edu-u3_little_tray_storage_apple_b`, `G1edu-u3_little_tray_storage_lemon_b`, `G1edu-u3_pick_apple_a`, `G1edu-u3_pick_apple_b`, `G1edu-u3_pick_crumpled_paper_aa` |
| 11 | `ruantong_a2d` | `34` | 30 | `AgiBot-g1_battery_storage_b`, `AgiBot-g1_battery_storage_c`, `AgiBot-g1_box_storage_a`, `AgiBot-g1_box_storage_b`, `AgiBot-g1_box_storage_c` |
| 12 | `Leju_Kuavo_4` | `40` | 30 | `Leju_Kuavo_4_box_storage_parcel_1`, `Leju_Kuavo_4_box_storage_parcel_10`, `Leju_Kuavo_4_box_storage_parcel_11`, `Leju_Kuavo_4_box_storage_parcel_2`, `Leju_Kuavo_4_box_storage_parcel_4` |
| 13 | `realman_rmc_aidal` | `28` | 27 | `RMC-AIDA-L_basket_storage_banana`, `RMC-AIDA-L_basket_storage_egg_yolk_pastry`, `RMC-AIDA-L_basket_storage_long_bread`, `RMC-AIDA-L_basket_storage_orange`, `RMC-AIDA-L_basket_storage_peach` |
| 14 | `galaxea_r1_lite` | `18` | 19 | `R1_Lite_build_blocks`, `R1_Lite_move_the_position_of_the_apple`, `R1_Lite_move_the_position_of_the_black_marker`, `R1_Lite_move_the_position_of_the_brush`, `R1_Lite_move_the_position_of_the_coffee_capsule` |
| 15 | `yinhe` | `16` | 13 | `Galbot_g1_fold_clothe_b`, `Galbot_g1_fold_clothe_c`, `Galbot_g1_fold_clothe_e`, `Galbot_g1_steamer_storage_baozi_a`, `Galbot_g1_steamer_storage_baozi_b` |
| 16 | `alpha_bot_2` | `28` | 10 | `alpha_bot_2_carry_the_clothes_basket`, `alpha_bot_2_item_reversal`, `alpha_bot_2_move_the_table`, `alpha_bot_2_operate_the_microwave_oven`, `alpha_bot_2_pass_the_sandbag` |
| 17 | `aloha` | `26` | 9 | `Agilex_Cobot_Magic_fold_towel_blue`, `Agilex_Cobot_Magic_fold_towel_brown`, `Agilex_Cobot_Magic_fold_towel_purple`, `Agilex_Cobot_Magic_heat_burger`, `Agilex_Cobot_Magic_heat_sandwich` |
| 18 | `Realman_RMC-AIDA-L` | `28` | 8 | `Realman_RMC_AIDA_L_arrange_flowers`, `Realman_RMC_AIDA_L_fold_towel`, `Realman_RMC_AIDA_L_hang_clothes`, `Realman_RMC_AIDA_L_pass_bowl`, `Realman_RMC_AIDA_L_storage_block_basket` |
| 19 | `Unitree_G1_Dex3_phecda` | `28` | 7 | `G1edu-u3_basket_storage_apple_b`, `G1edu-u3_bowl_storage_grape_singletry`, `G1edu-u3_pullBowl_storage_bread_a`, `G1edu-u3_pullBowl_storage_bread_b`, `G1edu-u3_pullBowl_storage_bread_unordered_C` |
| 20 | `unitree_g1` | `28` | 5 | `G1edu-u3_basket_storage_apple`, `G1edu-u3_food_storage`, `G1edu-u3_plate_storage_doll`, `G1edu-u3_plate_storage_rabbit_doll`, `G1edu-u3_stack_bowls` |
| 21 | `ruantong_a2d` | `17` | 4 | `AgiBot-g1_box_storage_tool`, `Tianqin_A2_box_storage_part`, `Tianqin_A2_container_storage_graphics_card`, `Tianqin_A2_place_the_paper_box` |

## Detailed Schemas

### 1. `Airbot_MMK2` action schema

- Shape: `36`
- Dtype: `float32`
- Dataset count: `143`
- Example datasets: `Airbot_MMK2_click_pen`, `Airbot_MMK2_close_door_left`, `Airbot_MMK2_close_door_right`, `Airbot_MMK2_close_doors`, `Airbot_MMK2_close_drawer`, `Airbot_MMK2_close_lid`, `Airbot_MMK2_cover_lid`, `Airbot_MMK2_cut_scallion`, `Airbot_MMK2_dial_number`, `Airbot_MMK2_doodled_line`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 12 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 13 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 14 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 15 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 16 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 17 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 18 | `left_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 19 | `left_hand_joint_8_rad` | joint angle or Euler angle in radians |
| 20 | `left_hand_joint_9_rad` | joint angle or Euler angle in radians |
| 21 | `left_hand_joint_10_rad` | joint angle or Euler angle in radians |
| 22 | `left_hand_joint_11_rad` | joint angle or Euler angle in radians |
| 23 | `left_hand_joint_12_rad` | joint angle or Euler angle in radians |
| 24 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 25 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 26 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 27 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 28 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 29 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 30 | `right_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 31 | `right_hand_joint_8_rad` | joint angle or Euler angle in radians |
| 32 | `right_hand_joint_9_rad` | joint angle or Euler angle in radians |
| 33 | `right_hand_joint_10_rad` | joint angle or Euler angle in radians |
| 34 | `right_hand_joint_11_rad` | joint angle or Euler angle in radians |
| 35 | `right_hand_joint_12_rad` | joint angle or Euler angle in radians |

### 2. `Agilex_Cobot_Magic` action schema

- Shape: `26`
- Dtype: `float32`
- Dataset count: `72`
- Example datasets: `Agilex_Cobot_Magic_classify_objects_eight`, `Agilex_Cobot_Magic_classify_objects_six`, `Agilex_Cobot_Magic_close_drawer_bottom`, `Agilex_Cobot_Magic_close_drawer_top`, `Agilex_Cobot_Magic_close_drawer_upper`, `Agilex_Cobot_Magic_connect_block`, `Agilex_Cobot_Magic_erase_board`, `Agilex_Cobot_Magic_erase_board_left`, `Agilex_Cobot_Magic_erase_board_left_side`, `Agilex_Cobot_Magic_erase_board_passing_left_to_right`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_gripper_open` | gripper opening scalar |
| 7 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 8 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 9 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 10 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 11 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 12 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 15 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 16 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 17 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 18 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 19 | `right_gripper_open` | gripper opening scalar |
| 20 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 21 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 22 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 23 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 24 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 25 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |

### 3. `discover_robotics_aitbot_mmk2` action schema

- Shape: `36`
- Dtype: `float32`
- Dataset count: `70`
- Example datasets: `AIRBOT_MMK2_beauty_sponge_and_cake_to_place`, `AIRBOT_MMK2_bowl_storage_pepper`, `AIRBOT_MMK2_boxs_storage`, `AIRBOT_MMK2_building_block_storage`, `AIRBOT_MMK2_cake_storage`, `AIRBOT_MMK2_chop_the_scallions`, `AIRBOT_MMK2_clean_the_desktop`, `AIRBOT_MMK2_clean_the_desktop_a`, `AIRBOT_MMK2_close_the_computer`, `AIRBOT_MMK2_cup_storage`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 12 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 13 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 14 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 15 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 16 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 17 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 18 | `left_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 19 | `left_hand_joint_8_rad` | joint angle or Euler angle in radians |
| 20 | `left_hand_joint_9_rad` | joint angle or Euler angle in radians |
| 21 | `left_hand_joint_10_rad` | joint angle or Euler angle in radians |
| 22 | `left_hand_joint_11_rad` | joint angle or Euler angle in radians |
| 23 | `left_hand_joint_12_rad` | joint angle or Euler angle in radians |
| 24 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 25 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 26 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 27 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 28 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 29 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 30 | `right_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 31 | `right_hand_joint_8_rad` | joint angle or Euler angle in radians |
| 32 | `right_hand_joint_9_rad` | joint angle or Euler angle in radians |
| 33 | `right_hand_joint_10_rad` | joint angle or Euler angle in radians |
| 34 | `right_hand_joint_11_rad` | joint angle or Euler angle in radians |
| 35 | `right_hand_joint_12_rad` | joint angle or Euler angle in radians |

### 4. `galaxea_r1_lite` action schema

- Shape: `14`
- Dtype: `float32`
- Dataset count: `52`
- Example datasets: `R1_Lite_boil_water_in_a_kettle`, `R1_Lite_catch_the_water`, `R1_Lite_clean_the_floor`, `R1_Lite_clean_the_sink`, `R1_Lite_clean_toilet`, `R1_Lite_connect_the_router_cable`, `R1_Lite_cook_a_meal`, `R1_Lite_cover_the_pot_lid`, `R1_Lite_dispose_of_leftover_food`, `R1_Lite_drawer_storage_hair_dryer`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_gripper_open` | gripper opening scalar |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_gripper_open` | gripper opening scalar |

### 5. `leju_robot` action schema

- Shape: `54`
- Dtype: `float32`
- Dataset count: `49`
- Example datasets: `leju_robot_box_storage_parcel`, `leju_robot_box_storage_parcel_a`, `leju_robot_box_storage_parcel_b`, `leju_robot_box_storage_parcel_c`, `leju_robot_box_storage_parcel_d`, `leju_robot_box_storage_parcel_f`, `leju_robot_box_storage_parcel_g`, `leju_robot_box_storage_parcel_h`, `leju_robot_box_storage_parcel_i`, `leju_robot_box_storage_parcel_j`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_leg_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_leg_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_leg_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_leg_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_leg_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_leg_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `right_leg_joint_1_rad` | joint angle or Euler angle in radians |
| 21 | `right_leg_joint_2_rad` | joint angle or Euler angle in radians |
| 22 | `right_leg_joint_3_rad` | joint angle or Euler angle in radians |
| 23 | `right_leg_joint_4_rad` | joint angle or Euler angle in radians |
| 24 | `right_leg_joint_5_rad` | joint angle or Euler angle in radians |
| 25 | `right_leg_joint_6_rad` | joint angle or Euler angle in radians |
| 26 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 27 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 28 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 29 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 30 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 31 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 32 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 33 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 34 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 35 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 36 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 37 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 38 | `head_joint_1_rad` | joint angle or Euler angle in radians |
| 39 | `head_joint_2_rad` | joint angle or Euler angle in radians |
| 40 | `left_arm_joint_1_vel_rad_s` | joint angular velocity in radians per second |
| 41 | `left_arm_joint_2_vel_rad_s` | joint angular velocity in radians per second |
| 42 | `left_arm_joint_3_vel_rad_s` | joint angular velocity in radians per second |
| 43 | `left_arm_joint_4_vel_rad_s` | joint angular velocity in radians per second |
| 44 | `left_arm_joint_5_vel_rad_s` | joint angular velocity in radians per second |
| 45 | `left_arm_joint_6_vel_rad_s` | joint angular velocity in radians per second |
| 46 | `left_arm_joint_7_vel_rad_s` | joint angular velocity in radians per second |
| 47 | `right_arm_joint_1_vel_rad_s` | joint angular velocity in radians per second |
| 48 | `right_arm_joint_2_vel_rad_s` | joint angular velocity in radians per second |
| 49 | `right_arm_joint_3_vel_rad_s` | joint angular velocity in radians per second |
| 50 | `right_arm_joint_4_vel_rad_s` | joint angular velocity in radians per second |
| 51 | `right_arm_joint_5_vel_rad_s` | joint angular velocity in radians per second |
| 52 | `right_arm_joint_6_vel_rad_s` | joint angular velocity in radians per second |
| 53 | `right_arm_joint_7_vel_rad_s` | joint angular velocity in radians per second |

### 6. `agilex_cobot_decoupled_magic` action schema

- Shape: `26`
- Dtype: `float32`
- Dataset count: `40`
- Example datasets: `Agilex_Cobot_Magic_basket_storage_banana`, `Agilex_Cobot_Magic_basket_storage_bread`, `Agilex_Cobot_Magic_basket_storage_egg_yolk_pastry`, `Agilex_Cobot_Magic_basket_storage_long_bread`, `Agilex_Cobot_Magic_basket_storage_orange`, `Agilex_Cobot_Magic_basket_storage_peach`, `Agilex_Cobot_Magic_fold_the_pants`, `Agilex_Cobot_Magic_plate_storage`, `Agilex_Cobot_Magic_pour_rice`, `Agilex_Cobot_Magic_pour_tea`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_gripper_open` | gripper opening scalar |
| 7 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 8 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 9 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 10 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 11 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 12 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 15 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 16 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 17 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 18 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 19 | `right_gripper_open` | gripper opening scalar |
| 20 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 21 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 22 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 23 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 24 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 25 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |

### 7. `agilex_cobot_decoupled_magic` action schema

- Shape: `14`
- Dtype: `float32`
- Dataset count: `39`
- Example datasets: `Agilex_Cobot_Magic_fold_clothes`, `Agilex_Split_Aloha_box_storage_chopsticks`, `Agilex_Split_Aloha_cap_the_pen_a`, `Agilex_Split_Aloha_catch_the_ball`, `Agilex_Split_Aloha_classification_of_fruits_and_vegetables`, `Agilex_Split_Aloha_classification_of_fruits_and_vegetables_a`, `Agilex_Split_Aloha_classification_of_tableware`, `Agilex_Split_Aloha_clean_up_the_tableware`, `Agilex_Split_Aloha_clear_the_desktop`, `Agilex_Split_Aloha_close_book`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_gripper_open` | gripper opening scalar |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_gripper_open` | gripper opening scalar |

### 8. `Galaxea_R1_Lite` action schema

- Shape: `14`
- Dtype: `float32`
- Dataset count: `39`
- Example datasets: `Galaxea_R1_Lite_arrange_baai_then_brain`, `Galaxea_R1_Lite_change_baai_into_brain`, `Galaxea_R1_Lite_classify_object_five`, `Galaxea_R1_Lite_classify_object_four`, `Galaxea_R1_Lite_classify_object_green_tablecloth`, `Galaxea_R1_Lite_classify_object_six`, `Galaxea_R1_Lite_classify_object_three`, `Galaxea_R1_Lite_fold_towel_twice`, `Galaxea_R1_Lite_mix_blue_yellow_large_test_tube`, `Galaxea_R1_Lite_mix_blue_yellow_left_large_test_tube`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 12 | `left_gripper_open` | gripper opening scalar |
| 13 | `right_gripper_open` | gripper opening scalar |

### 9. `AI2_Alphabot_2` action schema

- Shape: `34`
- Dtype: `float32`
- Dataset count: `37`
- Example datasets: `AI2_Alphabot_2_arrange_teaset`, `AI2_Alphabot_2_clink_glasses`, `AI2_Alphabot_2_grind_ink`, `AI2_Alphabot_2_heat_test_tube`, `AI2_Alphabot_2_insert_hose_into_hole`, `AI2_Alphabot_2_insert_straw`, `AI2_Alphabot_2_insert_water_pipe`, `AI2_Alphabot_2_move_basket`, `AI2_Alphabot_2_move_test_tube`, `AI2_Alphabot_2_open_microwave`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 8 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 9 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 10 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 11 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 12 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 15 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 16 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 17 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 18 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 19 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 20 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 21 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 22 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 23 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 24 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 25 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 26 | `left_gripper_open` | gripper opening scalar |
| 27 | `right_gripper_open` | gripper opening scalar |
| 28 | `neck_joint_1_rad` | joint angle or Euler angle in radians |
| 29 | `neck_joint_2_rad` | joint angle or Euler angle in radians |
| 30 | `torso_joint_1_rad` | joint angle or Euler angle in radians |
| 31 | `torso_joint_2_rad` | joint angle or Euler angle in radians |
| 32 | `torso_joint_3_rad` | joint angle or Euler angle in radians |
| 33 | `torso_joint_4_rad` | joint angle or Euler angle in radians |

### 10. `unknown` action schema

- Shape: `30`
- Dtype: `float32`
- Dataset count: `35`
- Example datasets: `G1edu-u3_little_tray_storage_apple_b`, `G1edu-u3_little_tray_storage_lemon_b`, `G1edu-u3_pick_apple_a`, `G1edu-u3_pick_apple_b`, `G1edu-u3_pick_crumpled_paper_aa`, `G1edu-u3_pick_cup_a`, `G1edu-u3_pick_empty_bottle_ab`, `G1edu-u3_pick_leftover_food_ac`, `G1edu-u3_pick_metal_bowl_aa`, `G1edu-u3_pick_metal_bowl_ab`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `left_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 21 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 22 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 23 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 24 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 25 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 26 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 27 | `right_hand_joint_7_rad` | joint angle or Euler angle in radians |

### 11. `ruantong_a2d` action schema

- Shape: `34`
- Dtype: `float32`
- Dataset count: `30`
- Example datasets: `AgiBot-g1_battery_storage_b`, `AgiBot-g1_battery_storage_c`, `AgiBot-g1_box_storage_a`, `AgiBot-g1_box_storage_b`, `AgiBot-g1_box_storage_c`, `AgiBot-g1_box_storage_cardboard_box_a`, `AgiBot-g1_box_storage_cardboard_box_b`, `AgiBot-g1_box_storage_cardboard_box_c`, `AgiBot-g1_box_storage_e`, `AgiBot-g1_box_storage_part_a`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_end_pos_x_m` | Cartesian position coordinate in meters |
| 15 | `left_end_pos_y_m` | Cartesian position coordinate in meters |
| 16 | `left_end_pos_z_m` | Cartesian position coordinate in meters |
| 17 | `left_end_quat_x` | end-effector orientation quaternion component |
| 18 | `left_end_quat_y` | end-effector orientation quaternion component |
| 19 | `left_end_quat_z` | end-effector orientation quaternion component |
| 20 | `left_end_quat_w` | end-effector orientation quaternion component |
| 21 | `right_end_pos_x_m` | Cartesian position coordinate in meters |
| 22 | `right_end_pos_y_m` | Cartesian position coordinate in meters |
| 23 | `right_end_pos_z_m` | Cartesian position coordinate in meters |
| 24 | `right_end_quat_x` | end-effector orientation quaternion component |
| 25 | `right_end_quat_y` | end-effector orientation quaternion component |
| 26 | `right_end_quat_z` | end-effector orientation quaternion component |
| 27 | `right_end_quat_w` | end-effector orientation quaternion component |
| 28 | `waist_yaw_rad` | joint angle or Euler angle in radians |
| 29 | `waist_pitch_rad` | joint angle or Euler angle in radians |
| 30 | `head_yaw_rad` | joint angle or Euler angle in radians |
| 31 | `head_pitch_rad` | joint angle or Euler angle in radians |
| 32 | `left_gripper_open` | gripper opening scalar |
| 33 | `right_gripper_open` | gripper opening scalar |

### 12. `Leju_Kuavo_4` action schema

- Shape: `40`
- Dtype: `float32`
- Dataset count: `30`
- Example datasets: `Leju_Kuavo_4_box_storage_parcel_1`, `Leju_Kuavo_4_box_storage_parcel_10`, `Leju_Kuavo_4_box_storage_parcel_11`, `Leju_Kuavo_4_box_storage_parcel_2`, `Leju_Kuavo_4_box_storage_parcel_4`, `Leju_Kuavo_4_box_storage_parcel_5`, `Leju_Kuavo_4_box_storage_parcel_8`, `Leju_Kuavo_4_box_storage_parcel_9`, `Leju_Kuavo_4_hotel_services`, `Leju_Kuavo_4_moving_parts`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_leg_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_leg_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_leg_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_leg_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_leg_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_leg_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `right_leg_joint_1_rad` | joint angle or Euler angle in radians |
| 21 | `right_leg_joint_2_rad` | joint angle or Euler angle in radians |
| 22 | `right_leg_joint_3_rad` | joint angle or Euler angle in radians |
| 23 | `right_leg_joint_4_rad` | joint angle or Euler angle in radians |
| 24 | `right_leg_joint_5_rad` | joint angle or Euler angle in radians |
| 25 | `right_leg_joint_6_rad` | joint angle or Euler angle in radians |
| 26 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 27 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 28 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 29 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 30 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 31 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 32 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 33 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 34 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 35 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 36 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 37 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 38 | `head_joint_1_rad` | joint angle or Euler angle in radians |
| 39 | `head_joint_2_rad` | joint angle or Euler angle in radians |

### 13. `realman_rmc_aidal` action schema

- Shape: `28`
- Dtype: `float32`
- Dataset count: `27`
- Example datasets: `RMC-AIDA-L_basket_storage_banana`, `RMC-AIDA-L_basket_storage_egg_yolk_pastry`, `RMC-AIDA-L_basket_storage_long_bread`, `RMC-AIDA-L_basket_storage_orange`, `RMC-AIDA-L_basket_storage_peach`, `RMC-AIDA-L_box_up_down`, `RMC-AIDA-L_clean_table`, `RMC-AIDA-L_desktop_organization`, `RMC-AIDA-L_fold_shirt`, `RMC-AIDA-L_fold_shorts`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_gripper_open` | gripper opening scalar |
| 8 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 9 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 10 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 11 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 12 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 13 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 14 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 21 | `left_gripper_open` | gripper opening scalar |
| 22 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 23 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 24 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 25 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 26 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 27 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |

### 14. `galaxea_r1_lite` action schema

- Shape: `18`
- Dtype: `float32`
- Dataset count: `19`
- Example datasets: `R1_Lite_build_blocks`, `R1_Lite_move_the_position_of_the_apple`, `R1_Lite_move_the_position_of_the_black_marker`, `R1_Lite_move_the_position_of_the_brush`, `R1_Lite_move_the_position_of_the_coffee_capsule`, `R1_Lite_move_the_position_of_the_cookie`, `R1_Lite_move_the_position_of_the_duck`, `R1_Lite_move_the_position_of_the_glass`, `R1_Lite_move_the_position_of_the_long_bread`, `R1_Lite_move_the_position_of_the_milk`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_gripper_open` | gripper opening scalar |
| 15 | `right_gripper_open` | gripper opening scalar |

### 15. `yinhe` action schema

- Shape: `16`
- Dtype: `float32`
- Dataset count: `13`
- Example datasets: `Galbot_g1_fold_clothe_b`, `Galbot_g1_fold_clothe_c`, `Galbot_g1_fold_clothe_e`, `Galbot_g1_steamer_storage_baozi_a`, `Galbot_g1_steamer_storage_baozi_b`, `Galbot_g1_steamer_storage_baozi_c`, `Galbot_g1_steamer_storage_baozi_d`, `Galbot_g1_steamer_storage_baozi_e`, `Galbot_g1_steamer_storage_baozi_f`, `Galbot_g1_steamer_storage_baozi_g`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `left_gripper_open` | gripper opening scalar |
| 8 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 15 | `right_gripper_open` | gripper opening scalar |

### 16. `alpha_bot_2` action schema

- Shape: `28`
- Dtype: `float32`
- Dataset count: `10`
- Example datasets: `alpha_bot_2_carry_the_clothes_basket`, `alpha_bot_2_item_reversal`, `alpha_bot_2_move_the_table`, `alpha_bot_2_operate_the_microwave_oven`, `alpha_bot_2_pass_the_sandbag`, `alpha_bot_2_press_the_button_a`, `alpha_bot_2_press_the_button_b`, `alpha_bot_2_recover_after_touching_an_obstacle`, `alpha_bot_2_stack_building_blocks`, `alpha_bot_2_sticker`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 8 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 9 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 10 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 11 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 12 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 15 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 16 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 17 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 18 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 19 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 20 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 21 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 22 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 23 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 24 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 25 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 26 | `left_gripper_open` | gripper opening scalar |
| 27 | `right_gripper_open` | gripper opening scalar |

### 17. `aloha` action schema

- Shape: `26`
- Dtype: `float32`
- Dataset count: `9`
- Example datasets: `Agilex_Cobot_Magic_fold_towel_blue`, `Agilex_Cobot_Magic_fold_towel_brown`, `Agilex_Cobot_Magic_fold_towel_purple`, `Agilex_Cobot_Magic_heat_burger`, `Agilex_Cobot_Magic_heat_sandwich`, `Agilex_Cobot_Magic_make_sandwiche`, `Agilex_Cobot_Magic_place_towel_flat`, `Agilex_Cobot_Magic_storage_towel`, `Agilex_Cobot_Magic_sweep_coffee_beans`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 7 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 8 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 9 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 10 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 11 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 12 | `left_gripper_open` | gripper opening scalar |
| 13 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 15 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 16 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 17 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 18 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 19 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 20 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 21 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 22 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 23 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 24 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 25 | `right_gripper_open` | gripper opening scalar |

### 18. `Realman_RMC-AIDA-L` action schema

- Shape: `28`
- Dtype: `float32`
- Dataset count: `8`
- Example datasets: `Realman_RMC_AIDA_L_arrange_flowers`, `Realman_RMC_AIDA_L_fold_towel`, `Realman_RMC_AIDA_L_hang_clothes`, `Realman_RMC_AIDA_L_pass_bowl`, `Realman_RMC_AIDA_L_storage_block_basket`, `Realman_RMC_AIDA_L_storage_peach_box`, `Realman_RMC_AIDA_L_storage_peach_drawer`, `Realman_RMC_AIDA_L_storage_towel_basket`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_gripper_open` | gripper opening scalar |
| 8 | `right_eef_pos_x_m` | Cartesian position coordinate in meters |
| 9 | `right_eef_pos_y_m` | Cartesian position coordinate in meters |
| 10 | `right_eef_pos_z_m` | Cartesian position coordinate in meters |
| 11 | `right_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 12 | `right_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 13 | `right_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |
| 14 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 21 | `left_gripper_open` | gripper opening scalar |
| 22 | `left_eef_pos_x_m` | Cartesian position coordinate in meters |
| 23 | `left_eef_pos_y_m` | Cartesian position coordinate in meters |
| 24 | `left_eef_pos_z_m` | Cartesian position coordinate in meters |
| 25 | `left_eef_rot_euler_x_rad` | joint angle or Euler angle in radians |
| 26 | `left_eef_rot_euler_y_rad` | joint angle or Euler angle in radians |
| 27 | `left_eef_rot_euler_z_rad` | joint angle or Euler angle in radians |

### 19. `Unitree_G1_Dex3_phecda` action schema

- Shape: `28`
- Dtype: `float32`
- Dataset count: `7`
- Example datasets: `G1edu-u3_basket_storage_apple_b`, `G1edu-u3_bowl_storage_grape_singletry`, `G1edu-u3_pullBowl_storage_bread_a`, `G1edu-u3_pullBowl_storage_bread_b`, `G1edu-u3_pullBowl_storage_bread_unordered_C`, `G1edu-u3_pullBowl_storage_bread_unordered_a`, `G1edu-u3_pullBowl_storage_bread_unordered_b`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `left_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 21 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 22 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 23 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 24 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 25 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 26 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 27 | `right_hand_joint_7_rad` | joint angle or Euler angle in radians |

### 20. `unitree_g1` action schema

- Shape: `28`
- Dtype: `float32`
- Dataset count: `5`
- Example datasets: `G1edu-u3_basket_storage_apple`, `G1edu-u3_food_storage`, `G1edu-u3_plate_storage_doll`, `G1edu-u3_plate_storage_rabbit_doll`, `G1edu-u3_stack_bowls`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 7 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 14 | `left_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 15 | `left_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 16 | `left_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 17 | `left_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 18 | `left_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 19 | `left_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 20 | `left_hand_joint_7_rad` | joint angle or Euler angle in radians |
| 21 | `right_hand_joint_1_rad` | joint angle or Euler angle in radians |
| 22 | `right_hand_joint_2_rad` | joint angle or Euler angle in radians |
| 23 | `right_hand_joint_3_rad` | joint angle or Euler angle in radians |
| 24 | `right_hand_joint_4_rad` | joint angle or Euler angle in radians |
| 25 | `right_hand_joint_5_rad` | joint angle or Euler angle in radians |
| 26 | `right_hand_joint_6_rad` | joint angle or Euler angle in radians |
| 27 | `right_hand_joint_7_rad` | joint angle or Euler angle in radians |

### 21. `ruantong_a2d` action schema

- Shape: `17`
- Dtype: `float32`
- Dataset count: `4`
- Example datasets: `AgiBot-g1_box_storage_tool`, `Tianqin_A2_box_storage_part`, `Tianqin_A2_container_storage_graphics_card`, `Tianqin_A2_place_the_paper_box`

| Index | Field | Meaning |
|---:|---|---|
| 0 | `robot_joint_3_rad` | joint angle or Euler angle in radians |
| 1 | `left_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 2 | `left_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 3 | `left_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 4 | `left_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 5 | `left_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 6 | `left_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 7 | `left_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 8 | `right_arm_joint_1_rad` | joint angle or Euler angle in radians |
| 9 | `right_arm_joint_2_rad` | joint angle or Euler angle in radians |
| 10 | `right_arm_joint_3_rad` | joint angle or Euler angle in radians |
| 11 | `right_arm_joint_4_rad` | joint angle or Euler angle in radians |
| 12 | `right_arm_joint_5_rad` | joint angle or Euler angle in radians |
| 13 | `right_arm_joint_6_rad` | joint angle or Euler angle in radians |
| 14 | `right_arm_joint_7_rad` | joint angle or Euler angle in radians |
| 15 | `left_gripper_open` | gripper opening scalar |
| 16 | `right_gripper_open` | gripper opening scalar |

## How To Inspect One Dataset

```bash
python - <<'PY'
import json
from pathlib import Path
info = json.loads(Path("/root/nasbak/cjy/robot_raw/robocoin/RoboCOIN/AIRBOT_MMK2_bowl_storage_pepper/meta/info.json").read_text())
for i, name in enumerate(info["features"]["action"]["names"]):
    print(i, name)
PY
```
