import pandas as pd
import time
import random
import math
import argparse
import copy


# 数据处理：处理教学周的问题
def preprocess_course_data(courses_df):
    """
    预处理课程数据，解析教学周模式与单双周信息
    """

    def parse_week_pattern(pattern):
        if not isinstance(pattern, str):
            return []
        weeks = []
        for part in pattern.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                weeks.extend(range(start, end + 1))
            else:
                try:
                    weeks.append(int(part))
                except:
                    pass
        return sorted(weeks)

    def determine_week_parity(delivery_semester):
        if not isinstance(delivery_semester, str):
            return 'ALL'
        lower_str = delivery_semester.lower()
        if 'odd' in lower_str or 'uneven' in lower_str:
            return 'ODD'
        elif 'even' in lower_str:
            return 'EVEN'
        else:
            return 'ALL'

    courses_df['Teaching_Weeks'] = courses_df['Teaching Week Pattern'].apply(parse_week_pattern)
    courses_df['Week_Parity'] = courses_df['Delivery Semester'].apply(determine_week_parity)

    for idx, row in courses_df.iterrows():
        if row['Week_Parity'] == 'ODD':
            courses_df.at[idx, 'Teaching_Weeks'] = [w for w in row['Teaching_Weeks'] if w % 2 == 1]
        elif row['Week_Parity'] == 'EVEN':
            courses_df.at[idx, 'Teaching_Weeks'] = [w for w in row['Teaching_Weeks'] if w % 2 == 0]

    return courses_df

# 解析duration并转换为时间槽数
def parse_duration(duration_str):
    """将持续时间字符串(如'1:00'或'2:00')转换为时间槽数"""
    if not isinstance(duration_str, str):
        return 2  # 默认1小时，占用2个时间槽

    try:
        # 解析格式为'小时:分钟'的字符串
        if ':' in duration_str:
            hours, minutes = map(int, duration_str.split(':'))
            # 计算总分钟数，然后除以30分钟得到时间槽数
            slots = (hours * 60 + minutes) / 30
            # 确保返回整数时间槽
            return int(round(slots))

        # 也可能是纯数字表示小时
        elif duration_str.isdigit():
            return int(duration_str) * 2  # 每小时2个时间槽
    except:
        pass

    return 2  # 默认值

# 处理不同活动类型和合并workshop
def preprocess_course_with_activities(courses_df):
    """
    预处理课程数据，处理不同的活动类型（lecture, workshop等）
    并合并符合条件的workshop:
    - 相同course code
    - 相同Teaching Week Pattern
    - 相同Activity Type Name (workshop或computer workshop)
    - 合并后不超过120人
    """
    # 先使用原有的preprocess_course_data处理教学周信息
    courses_df = preprocess_course_data(courses_df)

    # 确保Activity Type Name列存在
    if 'Activity Type Name' not in courses_df.columns:
        print("警告: 数据中缺少'Activity Type Name'列，无法处理不同活动类型")
        return courses_df

    # 创建一个新的DataFrame来存储处理后的结果
    processed_courses = []

    # 先处理非workshop类型课程（直接添加）
    non_workshop_df = courses_df[~courses_df['Activity Type Name'].isin(['workshop', 'computer workshop'])]
    for _, row in non_workshop_df.iterrows():
        processed_courses.append(row.to_dict())

    # 识别要合并的workshop组
    workshop_df = courses_df[courses_df['Activity Type Name'].str.lower().isin(['*workshop', 'computer workshop']) |
                             courses_df['Activity Type Name'].str.lower().isin(['workshop', '*workshop']) |
                             courses_df['Activity Type Name'].str.contains('workshop', case=False)]
    # 按照course code、Teaching Week Pattern和Activity Type Name分组
    workshop_groups = []
    for (course_code, week_pattern, activity_type), group in workshop_df.groupby(
            ['Course Code', 'Teaching Week Pattern', 'Activity Type Name']):
        workshop_groups.append((course_code, week_pattern, activity_type, group))

    print(f"找到 {len(workshop_groups)} 组可能合并的workshop组")

    # 处理每个组
    for group_info in workshop_groups:
        course_code, week_pattern, activity_type, group = group_info

        # 转换为列表以便处理
        workshops = group.to_dict('records')

        # 如果只有一个workshop，直接添加
        if len(workshops) == 1:
            processed_courses.append(workshops[0])
            continue

        print(f"处理 {course_code} {activity_type} 组, 共 {len(workshops)} 个workshop")

        # 尝试合并workshops
        merged_workshops = []
        current_group = []
        current_size = 0

        # 按照size排序，便于更好地组合
        sorted_workshops = sorted(workshops, key=lambda x: x['Real Size'])

        for workshop in sorted_workshops:
            # 如果添加这个workshop后总人数不超过120，则添加到当前组
            if current_size + workshop['Real Size'] <= 120:
                current_group.append(workshop)
                current_size += workshop['Real Size']
            else:
                # 当前组已满，保存并开始新的组
                if current_group:
                    merged_workshops.append(current_group)
                current_group = [workshop]
                current_size = workshop['Real Size']

        # 添加最后一组
        if current_group:
            merged_workshops.append(current_group)

        print(f"  合并为 {len(merged_workshops)} 组")

        # 将合并后的workshop添加到结果中
        for group_idx, group in enumerate(merged_workshops):
            if len(group) == 1:
                # 只有一个workshop，直接添加
                workshop_entry = group[0].copy()
                workshop_entry['Merged_ID'] = None
                workshop_entry['Is_Merged'] = False
                workshop_entry['Merged_Count'] = 1
                workshop_entry['Group_Index'] = group_idx
                processed_courses.append(workshop_entry)
            else:
                # 合并多个workshop
                main_workshop = group[0].copy()  # 以第一个workshop为基础

                # 记录原始ID，用于后续展开
                original_ids = []
                for w in group:
                    if 'ID' in w:
                        original_ids.append(w['ID'])
                    else:
                        # 如果没有ID，使用行索引
                        original_ids.append(f"{course_code}_{activity_type}_{w.get('Row_ID', 'unknown')}")

                # 更新合并信息
                main_workshop['Real Size'] = sum(w['Real Size'] for w in group)
                main_workshop['Merged_IDs'] = original_ids
                main_workshop['Is_Merged'] = True
                main_workshop['Merged_Count'] = len(group)
                main_workshop['Group_Index'] = group_idx

                # 记录被合并的workshop信息（为了展开时使用）
                merged_details = []
                for w in group:
                    merged_details.append({
                        'ID': w.get('ID', f"{course_code}_{activity_type}_{w.get('Row_ID', 'unknown')}"),
                        'Real Size': w['Real Size']
                    })
                main_workshop['Merged_Details'] = merged_details

                processed_courses.append(main_workshop)

    # 将处理后的结果转换回DataFrame
    result_df = pd.DataFrame(processed_courses)

    print(f"处理前课程总数: {len(courses_df)}")
    print(f"处理后课程总数: {len(result_df)}")
    print(f"其中，合并的workshop组数: {sum(1 for _, row in result_df.iterrows() if row.get('Is_Merged', False))}")

    return result_df

def convert_weekday_to_week_and_day(weekday_index):
    """
    根据数学模型的weekday索引转换
    例如：11 -> (2, 1)
    """
    week = (weekday_index - 1) // 5 + 1
    day_in_week = (weekday_index - 1) % 5 + 1
    return week, day_in_week

def convert_slot_to_time(time_slot):
    """
    将时间段索引(1-18)转换为时间字符串，如'09:00'
    """
    if not isinstance(time_slot, int) or time_slot < 1 or time_slot > 19:
        return None
    hour = 9 + ((time_slot - 1) // 2)
    minute = 30 if (time_slot - 1) % 2 == 1 else 0
    return "{:02d}:{:02d}".format(hour, minute)

def normalize_room_name(room_name):
    """
    标准化教室名称，以便于匹配
    - 转为小写
    - 移除多余空格
    - 替换常见的分隔符变体
    """
    if not isinstance(room_name, str):
        return ""

    # 转为小写
    normalized = room_name.lower()

    # 替换常见分隔符变体
    normalized = normalized.replace(" - ", " ")
    normalized = normalized.replace("_", " ")

    # 移除多余空格
    normalized = " ".join(normalized.split())

    return normalized

def calculate_energy(solution, course_code_to_row, room_name_to_row, course_to_index, room_to_index,
                     index_to_course, index_to_room, course_students, utilization_weight=1.0, conflict_weight=100.0):
    """
    计算解的能量值：
      - 硬约束（同一时间同一教室只能安排一门课、教室容量不足、课程每周只安排一次等）受到高惩罚；
      - 软约束包括：鼓励高教室利用率以及惩罚学生课程冲突（同一时间安排了某学生选的多门课）。

    更新：支持不同时长的课程，并将学生冲突改为软约束。
    """
    energy = 0
    hard_constraint_weight = 1000

    # 约束1：同一时间同一教室只能安排一门课程
    # 更新以考虑课程持续时间
    room_time_conflicts = {}

    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 对课程占用的每个时间槽进行检查
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            key = (weekday_idx, curr_slot, room_idx)
            room_time_conflicts[key] = room_time_conflicts.get(key, 0) + 1

    for count in room_time_conflicts.values():
        if count > 1:
            energy += (count - 1) * hard_constraint_weight

    # 约束2：同一课程同一时间不能安排在多个教室
    # 这约束不需要变化，因为已经由课程唯一索引保证

    # 约束3：教室容量不足
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 检查课程索引和教室索引是否有效
        if course_idx not in index_to_course or room_idx not in index_to_room:
            energy += hard_constraint_weight  # 无效索引，添加惩罚
            continue

        course_code = index_to_course[course_idx]
        course_row = course_code_to_row.get(course_idx)  # 使用课程索引而非课程代码
        if course_row is None:
            energy += hard_constraint_weight  # 找不到课程信息，添加惩罚
            continue

        course_size = course_row['Real Size']
        room_name = index_to_room[room_idx]
        room_row = room_name_to_row.get(room_name)
        if room_row is None:
            energy += hard_constraint_weight  # 找不到教室信息，添加惩罚
            continue

        room_capacity = room_row['CAP']
        if course_size > room_capacity:
            energy += (course_size - room_capacity) * hard_constraint_weight

    # 约束4：每门课程在每个教学周只安排一次课
    course_week_counts = {}
    for (course_idx, weekday_idx, time_slot), _ in solution.items():
        week, _ = convert_weekday_to_week_and_day(weekday_idx)
        key = (course_idx, week)
        course_week_counts[key] = course_week_counts.get(key, 0) + 1

    for count in course_week_counts.values():
        if count > 1:
            energy += (count - 1) * hard_constraint_weight

    # 约束5：确保长课程不会跨越午休时间或一天结束
    # 午休时间为12:00-13:30（对应时间槽7-9）
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 检查是否跨越午休
        if (time_slot <= 6 and time_slot + duration_slots > 6) or (time_slot <= 9 and time_slot + duration_slots > 9):
            energy += hard_constraint_weight * 2  # 跨越午休，添加更高惩罚

        # 检查是否超出一天时间范围
        if time_slot + duration_slots > 19:
            energy += hard_constraint_weight  # 超出时间范围，添加惩罚

        # 约束6（新增）：确保课程持续时间与原始数据一致
        course_row = course_code_to_row.get(course_idx)
        if course_row is not None:
            expected_duration = parse_duration(course_row.get('Duration', '1:00'))
            if duration_slots != expected_duration:
                energy += hard_constraint_weight * 5  # 持续时间不一致，添加高惩罚

    # 软约束1：教室利用率（使用传入的权重）
    total_utilization = 0
    count = 0

    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        if course_idx not in index_to_course or room_idx not in index_to_room:
            continue

        course_code = index_to_course[course_idx]
        course_row = course_code_to_row.get(course_idx)
        if course_row is None:
            continue

        course_size = course_row['Real Size']
        room_name = index_to_room[room_idx]
        room_row = room_name_to_row.get(room_name)
        if room_row is None:
            continue

        room_capacity = room_row['CAP']
        utilization = course_size / room_capacity * 100
        total_utilization += utilization * duration_slots  # 考虑时长影响
        count += duration_slots  # 计算总时间槽数而非课程数

    if count > 0:
        avg_utilization = total_utilization / count
        # 鼓励高利用率（100% - 平均利用率）* 权重
        energy += (100 - avg_utilization) * utilization_weight

    # 软约束2：学生课程冲突率
    # 创建更详细的时间槽-课程映射
    time_slot_courses = {}
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        course_code = index_to_course.get(course_idx)
        if not course_code:
            continue

        # 获取课程的学期和周信息
        course_row = course_code_to_row.get(course_idx)
        if not course_row:
            continue

        semester_str = course_row.get('Delivery Semester', 'Unknown')
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 标准化学期信息
        if isinstance(semester_str, str):
            semester_str = semester_str.lower()
            if 'semester1' in semester_str or 'semester 1' in semester_str:
                semester = "Semester1"
            elif 'semester2' in semester_str or 'semester 2' in semester_str:
                semester = "Semester2"
            else:
                semester = semester_str
        else:
            semester = "Unknown"

        # 对课程占用的每个时间槽进行检查
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            # 使用四元组作为键：(学期, 周, 日, 时间槽)，确保只在相同语境下比较课程
            time_key = (semester, week, day, curr_slot)

            if time_key not in time_slot_courses:
                time_slot_courses[time_key] = []
            time_slot_courses[time_key].append(course_code)

    conflict_count = 0
    for time_key, course_codes in time_slot_courses.items():
        n = len(course_codes)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    # 提取基础课程代码（移除可能的活动类型后缀）
                    base_course_i = course_codes[i].split('_')[0] if '_' in course_codes[i] else course_codes[i]
                    base_course_j = course_codes[j].split('_')[0] if '_' in course_codes[j] else course_codes[j]

                    s1 = course_students.get(base_course_i, set())
                    s2 = course_students.get(base_course_j, set())
                    common_students = len(s1.intersection(s2))
                    conflict_count += common_students

    # 修改：将学生冲突改为软约束，惩罚与冲突数量成比例
    # 使用更合理的权重，默认值从50000降低到100
    energy += conflict_count * conflict_weight

    return energy

def generate_initial_solution_with_activities(regular_course_indices, index_to_course_row,
                                              rooms_df, room_to_index, blocked_slots):
    """
    为常规课程生成初始解，支持不同活动类型和合并的workshop。
    在生成过程中排除blocked_slots中已被大课程占用的教室。
    返回格式：{(course_idx, weekday_idx, time_slot): (room_idx, duration_slots)}
    """
    solution = {}

    for course_idx in regular_course_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        teaching_weeks = course_row['Teaching_Weeks']

        # 获取课程持续时间
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # 记录原始持续时间和解析后的槽数，以便调试
        print(f"课程 {course_row['Course Code']} 原始持续时间: {original_duration}, 解析为 {duration_slots} 个时间槽")

        # 筛选容量足够的教室
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size:
                suitable_rooms.append(room_idx)

        if not suitable_rooms:
            activity_type = course_row.get('Activity Type Name', 'Unknown')
            print(f"警告: 课程 {course_row['Course Code']} ({activity_type}, 人数: {course_size}) 没有足够大的教室")
            continue

        for week in teaching_weeks:
            # 尝试多次寻找合适的时间和教室
            max_attempts = 20
            success = False

            for _ in range(max_attempts):
                day = random.randint(1, 5)
                # 确保长课程不会超出一天的时间范围或跨越午休时间
                valid_start_slots = []
                for potential_start in range(1, 19 - duration_slots + 1):
                    # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
                    if potential_start <= 6 and potential_start + duration_slots > 6:
                        continue
                    if potential_start <= 9 and potential_start + duration_slots > 9:
                        continue
                    valid_start_slots.append(potential_start)

                if not valid_start_slots:
                    continue  # 没有有效的开始时间，尝试下一次循环

                time_slot = random.choice(valid_start_slots)
                weekday_idx = 5 * (week - 1) + day

                # 检查该课程占用的所有时间槽是否都没有被阻塞
                all_slots_available = True
                available_rooms = list(suitable_rooms)  # 复制一份可用教室列表

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in blocked_slots:
                        # 移除该时间槽已被占用的教室
                        available_rooms = [r for r in available_rooms if r not in blocked_slots[key]]
                        if not available_rooms:
                            all_slots_available = False
                            break

                # 如果所有时间槽都有可用教室，则安排课程
                if all_slots_available and available_rooms:
                    chosen_room = random.choice(available_rooms)

                    # 将课程安排在连续的时间槽中，保存起始时间槽和持续时间
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # 更新blocked_slots，将该课程占用的所有时间槽都标记为已占用
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in blocked_slots:
                            blocked_slots[key] = set()
                        blocked_slots[key].add(chosen_room)

                    success = True
                    break

            # 如果尝试多次仍找不到合适的时间和教室，则记录警告
            if not success:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(
                    f"警告: 无法为课程 {course_row['Course Code']} ({activity_type}) 在第{week}周找到合适的时间和教室")

    return solution


def generate_neighbor_with_activities(solution, regular_course_indices, index_to_course_row,
                                      rooms_df, room_to_index, index_to_course,
                                      blocked_slots, course_students):
    """
    生成邻域解，支持不同活动类型和合并的workshop：
    优先选择有冲突的课程进行调整，修改其时间和/或教室，
    确保新选择的教室不被大课程占用。
    考虑课程的持续时间。
    """
    # 创建当前解的深拷贝，确保修改不会影响原解
    new_solution = copy.deepcopy(solution)

    if not solution:
        return new_solution

    # 筛选出常规课程的键，确保这些键在当前解中存在
    regular_keys = [key for key in solution.keys() if key[0] in regular_course_indices]

    if not regular_keys:
        return new_solution

    # 识别有冲突的课程键
    conflict_keys, _ = identify_conflict_courses_with_activities(solution, course_students,
                                                                 index_to_course, index_to_course_row)
    # 筛选只属于常规课程的冲突键
    valid_conflict_keys = [key for key in conflict_keys if key in solution and key[0] in regular_course_indices]

    # 优先选择有冲突的课程进行调整
    if valid_conflict_keys and random.random() < 0.95:  # 95%概率选择冲突课程
        key = random.choice(valid_conflict_keys)
    else:
        # 随机选择任意常规课程
        key = random.choice(regular_keys)

    # 验证键是否存在于解决方案中
    if key not in new_solution:
        print(f"警告: 键 {key} 不存在于解决方案中。跳过此邻域生成。")
        return solution  # 返回原始解决方案

    course_idx, weekday_idx, time_slot = key
    room_idx, duration_slots = new_solution[key]

    # 计算周和日
    week, _ = convert_weekday_to_week_and_day(weekday_idx)

    # 获取课程信息
    course_row = index_to_course_row.get(course_idx)
    if course_row is None:
        print(f"警告: 找不到课程索引 {course_idx} 的信息，跳过邻域生成")
        return new_solution

    course_size = course_row['Real Size']

    # 确定操作类型: 1=改变时间, 2=改变教室, 3=同时改变时间和教室
    operation = random.randint(1, 3)
    new_key = key

    # 从当前blocked_slots中创建一个副本
    current_blocked_slots = copy.deepcopy(blocked_slots)

    # 首先从blocked_slots中移除当前课程占用的时间槽
    for slot_offset in range(duration_slots):
        curr_slot = time_slot + slot_offset
        block_key = (weekday_idx, curr_slot)
        if block_key in current_blocked_slots and room_idx in current_blocked_slots[block_key]:
            current_blocked_slots[block_key].remove(room_idx)

    if operation in (1, 3):  # 改变时间 或 同时改变时间和教室
        # 生成新的时间，但确保不会跨越午休时间
        new_day = random.randint(1, 5)

        # 找出所有有效的开始时间槽
        valid_start_slots = []
        for potential_start in range(1, 19 - duration_slots + 1):
            # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
            if potential_start <= 6 and potential_start + duration_slots > 6:
                continue
            if potential_start <= 9 and potential_start + duration_slots > 9:
                continue
            valid_start_slots.append(potential_start)

        if not valid_start_slots:
            # 没有有效的开始时间，返回原始解
            return new_solution

        new_time_slot = random.choice(valid_start_slots)
        new_weekday_idx = 5 * (week - 1) + new_day

        # 检查新时间槽是否所有都可用（对于操作1，仍使用当前教室）
        all_slots_available = True
        for slot_offset in range(duration_slots):
            curr_slot = new_time_slot + slot_offset
            block_key = (new_weekday_idx, curr_slot)
            if block_key in current_blocked_slots and room_idx in current_blocked_slots[block_key]:
                all_slots_available = False
                break

        if all_slots_available:
            # 删除原始安排 - 这里已经验证过key在new_solution中存在
            del new_solution[key]

            if operation == 1:  # 仅改变时间
                new_solution[(course_idx, new_weekday_idx, new_time_slot)] = (room_idx, duration_slots)
                new_key = (course_idx, new_weekday_idx, new_time_slot)

                # 更新blocked_slots
                for slot_offset in range(duration_slots):
                    curr_slot = new_time_slot + slot_offset
                    block_key = (new_weekday_idx, curr_slot)
                    if block_key not in blocked_slots:
                        blocked_slots[block_key] = set()
                    blocked_slots[block_key].add(room_idx)

                return new_solution
            else:  # 同时改变时间和教室（先记录新时间）
                new_key = (course_idx, new_weekday_idx, new_time_slot)
        else:
            # 如果新时间不可用，返回原始解
            if operation == 1:
                return new_solution

    if operation in (2, 3):  # 改变教室 或 同时改变时间和教室
        # 筛选符合条件的教室：容量足够且不在blocked_slots中
        suitable_rooms = []

        # 对于操作2，检查原始时间；对于操作3，检查新时间
        check_weekday_idx = weekday_idx if operation == 2 else new_key[1]
        check_time_slot = time_slot if operation == 2 else new_key[2]

        for _, room in rooms_df.iterrows():
            r_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size and r_idx != room_idx:  # 排除当前教室
                # 检查教室在所有时间槽是否都可用
                all_slots_available = True
                for slot_offset in range(duration_slots):
                    curr_slot = check_time_slot + slot_offset
                    block_key = (check_weekday_idx, curr_slot)
                    if block_key in current_blocked_slots and r_idx in current_blocked_slots[block_key]:
                        all_slots_available = False
                        break

                if all_slots_available:
                    suitable_rooms.append(r_idx)

        if suitable_rooms:
            new_room_idx = random.choice(suitable_rooms)

            if operation == 2:  # 仅改变教室
                # 删除原始安排 - 这里已经验证过key在new_solution中存在
                if key in new_solution:  # 再次验证，以确保安全
                    del new_solution[key]
                    new_solution[(course_idx, weekday_idx, time_slot)] = (new_room_idx, duration_slots)

                    # 更新blocked_slots
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        block_key = (weekday_idx, curr_slot)
                        if block_key not in blocked_slots:
                            blocked_slots[block_key] = set()
                        blocked_slots[block_key].add(new_room_idx)

            else:  # 同时改变时间和教室
                # 删除原始安排 - 这里已经验证过key在new_solution中存在
                if key in new_solution:  # 再次验证，以确保安全
                    del new_solution[key]
                    new_solution[new_key] = (new_room_idx, duration_slots)

                    # 更新blocked_slots
                    for slot_offset in range(duration_slots):
                        curr_slot = new_key[2] + slot_offset
                        block_key = (new_key[1], curr_slot)
                        if block_key not in blocked_slots:
                            blocked_slots[block_key] = set()
                        blocked_slots[block_key].add(new_room_idx)

    return new_solution

def identify_conflict_courses_with_activities(solution, course_students, index_to_course, index_to_course_row):
    """
    识别当前解中存在冲突的课程，支持不同活动类型和合并的workshop。
    考虑课程的持续时间，严格按照"同一学期、相同教学周、相同weekday、有重合时间"的定义。
    返回：有冲突的课程键集合 (course_idx, weekday_idx, time_slot)。
    """
    conflict_keys = set()  # 改为存储完整的键

    # 获取课程的学期信息
    def get_normalized_semester(course_idx):
        course_row = index_to_course_row.get(course_idx)
        if not course_row:
            return "Unknown"

        semester_str = course_row.get('Delivery Semester', 'Unknown')
        if not isinstance(semester_str, str):
            return "Unknown"

        # 统一格式为小写
        semester_str = semester_str.lower()

        # 提取学期信息（semester1或semester2）
        if 'semester1' in semester_str or 'semester 1' in semester_str:
            return "Semester1"
        elif 'semester2' in semester_str or 'semester 2' in semester_str:
            return "Semester2"
        else:
            return semester_str  # 保持原样

    # 存储冲突的课程索引集合
    conflict_course_indices = set()

    # 按照(学期、周、日、时间槽)分组
    time_slot_courses = {}
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 提取周和日信息
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 获取学期信息
        course_code = index_to_course.get(course_idx)
        if not course_code:
            continue

        semester = get_normalized_semester(course_idx)

        # 对课程占用的每个时间槽进行检查
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset

            # 使用四元组作为键：(学期, 周, 日, 时间槽)
            key = (semester, week, day, curr_slot)

            if key not in time_slot_courses:
                time_slot_courses[key] = []

            time_slot_courses[key].append((course_idx, weekday_idx, time_slot))  # 存储完整的键

    # 对于每个时间点，如果有多个课程，检查是否有学生冲突
    for key, course_keys in time_slot_courses.items():
        if len(course_keys) > 1:
            for i in range(len(course_keys)):
                for j in range(i + 1, len(course_keys)):
                    course_key_i = course_keys[i]
                    course_key_j = course_keys[j]

                    course_idx_i = course_key_i[0]
                    course_idx_j = course_key_j[0]

                    course_code_i = index_to_course.get(course_idx_i)
                    course_code_j = index_to_course.get(course_idx_j)

                    if not course_code_i or not course_code_j:
                        continue

                    # 提取基础课程代码（移除可能的活动类型后缀）
                    base_course_i = course_code_i.split('_')[0] if '_' in course_code_i else course_code_i
                    base_course_j = course_code_j.split('_')[0] if '_' in course_code_j else course_code_j

                    students_i = course_students.get(base_course_i, set())
                    students_j = course_students.get(base_course_j, set())

                    # 如果有共同选课学生，标记为冲突课程
                    if students_i.intersection(students_j):
                        conflict_keys.add(course_key_i)
                        conflict_keys.add(course_key_j)
                        conflict_course_indices.add(course_idx_i)
                        conflict_course_indices.add(course_idx_j)

    # 额外检查：确保所有返回的键都在当前解中
    valid_conflict_keys = {key for key in conflict_keys if key in solution}

    # 如果需要，可以同时返回冲突课程索引集合以保持兼容性
    return valid_conflict_keys, conflict_course_indices


def convert_solution_to_schedule_with_activities(solution, all_courses_df, rooms_df,
                                                 index_to_course, index_to_room, index_to_course_row):
    """
    将解转换为排课表格式，支持不同活动类型和合并的workshop。
    对于合并的workshop，会展开为多条记录，每条记录使用相同的时间和教室。
    考虑课程的持续时间，并确保结束时间正确。
    """
    schedule = []

    # 存储合并workshop的信息以便展开
    merged_workshops = {}

    # 先处理所有非合并的课程
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 获取课程信息
        course_row = index_to_course_row.get(course_idx)
        if not course_row:
            continue

        # 如果是合并的workshop，先存储起来稍后展开
        if course_row.get('Is_Merged', False):
            merged_workshops[(course_idx, weekday_idx, time_slot, room_idx, duration_slots)] = course_row
            continue

        # 处理非合并课程
        course_code = index_to_course[course_idx]
        room_name = index_to_room[room_idx]

        # 找到对应的教室信息
        room_rows = rooms_df[rooms_df['ROOM NAME'] == room_name]
        if len(room_rows) == 0:
            continue
        room_row = room_rows.iloc[0]

        # 计算周和日
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 获取活动类型
        activity_type = course_row.get('Activity Type Name', 'Unknown')

        # 计算结束时间 - 修正结束时间计算，确保与课程持续时间一致
        end_time_slot = time_slot + duration_slots
        end_time = convert_slot_to_time(end_time_slot)

        # 添加课程安排
        schedule.append({
            'Course Code': course_code,
            'Course Name': course_row.get('Course Name', ''),
            'Activity Type': activity_type,
            'Delivery Semester': course_row.get('Delivery Semester', 'Unknown'),
            'Week': week,
            'Day': day,
            'Weekday Index': weekday_idx,
            'Room': room_name,
            'Room Capacity': room_row['CAP'],
            'Start Time Slot': time_slot,
            'Start Time': convert_slot_to_time(time_slot),
            'End Time Slot': end_time_slot,
            'End Time': end_time,
            'Duration Slots': duration_slots,
            'Duration': course_row.get('Duration', '1:00'),  # 保存原始持续时间
            'Class Size': course_row['Real Size'],
            'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
            'Is Merged Workshop': False
        })

    # 处理合并的workshop，展开为多条记录
    for (course_idx, weekday_idx, time_slot, room_idx, duration_slots), course_row in merged_workshops.items():
        course_code = index_to_course[course_idx]
        room_name = index_to_room[room_idx]

        # 找到对应的教室信息
        room_rows = rooms_df[rooms_df['ROOM NAME'] == room_name]
        if len(room_rows) == 0:
            continue
        room_row = room_rows.iloc[0]

        # 计算周和日
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 获取活动类型和合并信息
        activity_type = course_row.get('Activity Type Name', 'Unknown')
        merged_details = course_row.get('Merged_Details', [])

        # 计算结束时间 - 修正结束时间计算
        end_time_slot = time_slot + duration_slots
        end_time = convert_slot_to_time(end_time_slot)

        # 类型检查，确保merged_details是可迭代对象
        if isinstance(merged_details, list) and merged_details:
            # 为每个被合并的workshop创建记录
            for detail in merged_details:
                workshop_id = detail.get('ID', 'Unknown')
                workshop_size = detail.get('Real Size', 0)

                schedule.append({
                    'Course Code': course_code,
                    'Course Name': course_row.get('Course Name', ''),
                    'Activity Type': activity_type,
                    'Delivery Semester': course_row.get('Delivery Semester', 'Unknown'),
                    'Week': week,
                    'Day': day,
                    'Weekday Index': weekday_idx,
                    'Room': room_name,
                    'Room Capacity': room_row['CAP'],
                    'Start Time Slot': time_slot,
                    'Start Time': convert_slot_to_time(time_slot),
                    'End Time Slot': end_time_slot,
                    'End Time': end_time,
                    'Duration Slots': duration_slots,
                    'Duration': course_row.get('Duration', '1:00'),  # 保存原始持续时间
                    'Class Size': workshop_size,  # 使用原始workshop的人数
                    'Total Merged Size': course_row['Real Size'],  # 合并后的总人数
                    'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
                    'Is Merged Workshop': True,
                    'Workshop ID': workshop_id
                })
        else:
            # 如果没有详细信息，创建一条记录
            print(f"警告: merged_details不是列表类型或为空，类型为{type(merged_details)}")
            schedule.append({
                'Course Code': course_code,
                'Course Name': course_row.get('Course Name', ''),
                'Activity Type': activity_type,
                'Delivery Semester': course_row.get('Delivery Semester', 'Unknown'),
                'Week': week,
                'Day': day,
                'Weekday Index': weekday_idx,
                'Room': room_name,
                'Room Capacity': room_row['CAP'],
                'Start Time Slot': time_slot,
                'Start Time': convert_slot_to_time(time_slot),
                'End Time Slot': end_time_slot,
                'End Time': end_time,
                'Duration Slots': duration_slots,
                'Duration': course_row.get('Duration', '1:00'),  # 保存原始持续时间
                'Class Size': course_row['Real Size'],
                'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
                'Is Merged Workshop': True,
                'Merged Count': course_row.get('Merged_Count', 0)
            })

    # 将return语句移到函数最外层
    return pd.DataFrame(schedule)

def compute_student_conflict_with_activities(schedule_df, course_students):
    """
    计算排课表中学生课程冲突的总数与冲突率，支持不同活动类型和合并的workshop。
    严格按照"同一学期、相同教学周、相同weekday、有重合时间"的定义查找冲突课程。

    课程冲突率 = 所有学生有冲突的课的门数 / 所有学生选的所有课的门数
    例如若学生a有两门课发生冲突，则该学生有冲突的门数计2门。

    注意：学期只有semester1和semester2两种，标有"odd weeks only"或"even weeks only"只是表示
    单双周信息，不影响学期判断。
    """
    # 收集所有选课学生（全体学生并集）以及统计总选课数
    total_students = set()
    total_course_selections = 0  # 所有学生选的所有课的门数

    # 计算总选课数和总学生数
    for course_code, students in course_students.items():
        total_students.update(students)
        total_course_selections += len(students)  # 每个课程的选课人数累加

    # 正则化学期信息，只保留semester1或semester2
    def normalize_semester(semester_str):
        if not isinstance(semester_str, str):
            return "Unknown"

        # 统一格式为小写
        semester_str = semester_str.lower()

        # 提取学期信息（semester1或semester2）
        if 'semester1' in semester_str or 'semester 1' in semester_str:
            return "Semester1"
        elif 'semester2' in semester_str or 'semester 2' in semester_str:
            return "Semester2"
        else:
            return semester_str  # 保持原样，如果无法识别

    # 预处理：规范化学期信息
    schedule_df['Normalized_Semester'] = schedule_df['Delivery Semester'].apply(normalize_semester)

    # 按照同一规范化学期、同一教学周以及同一天（Day）分组
    grouped = schedule_df.groupby(['Normalized_Semester', 'Week', 'Day'])

    # 存储冲突信息
    conflict_pairs = []

    # 遍历每个组，查找时间冲突的课程对
    for group_key, group_df in grouped:
        semester, week, day = group_key

        # 在同一组内找出时间相同的课程
        courses_count = len(group_df)
        if courses_count > 1:
            # 转换为列表以便索引访问
            courses_list = group_df.to_dict('records')

            # 按时间槽分组存储课程
            time_slot_courses = {}
            for course in courses_list:
                # 修改这里：使用'Start Time Slot'而不是'Time Slot'
                slot = course['Start Time Slot']
                if slot not in time_slot_courses:
                    time_slot_courses[slot] = []
                time_slot_courses[slot].append(course)

            # 检查每个时间槽中是否有多门课程
            for slot, courses in time_slot_courses.items():
                if len(courses) > 1:
                    # 对于同一时间槽内的每对课程
                    for i in range(len(courses)):
                        for j in range(i + 1, len(courses)):
                            # 提取基础课程代码（不考虑可能的活动类型后缀）
                            course_i = courses[i]['Course Code']
                            course_j = courses[j]['Course Code']
                            base_course_i = course_i.split('_')[0] if '_' in course_i else course_i
                            base_course_j = course_j.split('_')[0] if '_' in course_j else course_j

                            conflict_pairs.append((
                                base_course_i,
                                base_course_j,
                                (semester, week, day, slot)  # 记录详细冲突信息
                            ))

    print(f"找到 {len(conflict_pairs)} 组时间冲突的课程对")

    # 检查每对冲突课程是否有共同选课学生
    conflict_students = set()  # 有冲突的学生集合
    student_conflict_courses = {}  # 每个学生的冲突课程集合
    conflict_details = []

    for course_i, course_j, conflict_info in conflict_pairs:
        students_i = course_students.get(course_i, set())
        students_j = course_students.get(course_j, set())

        # 找出同时选了这两门课的学生
        common_students = students_i.intersection(students_j)

        if common_students:
            semester, week, day, time_slot = conflict_info
            time_str = convert_slot_to_time(time_slot)

            conflict_details.append({
                "学期": semester,
                "教学周": week,
                "星期": day,
                "时间": time_str,
                "课程1": course_i,
                "课程2": course_j,
                "冲突学生数": len(common_students)
            })

            # 更新每个学生的冲突课程集合
            for student in common_students:
                if student not in student_conflict_courses:
                    student_conflict_courses[student] = set()
                student_conflict_courses[student].add(course_i)
                student_conflict_courses[student].add(course_j)

            conflict_students.update(common_students)

    # 计算所有学生有冲突的课的门数
    total_conflict_courses = 0
    for student, courses in student_conflict_courses.items():
        total_conflict_courses += len(courses)

    # 输出详细的冲突信息
    if conflict_details:
        print("\n详细冲突信息:")
        for i, detail in enumerate(conflict_details, 1):
            print(f"{i}. 学期:{detail['学期']}, 教学周:{detail['教学周']}, 星期:{detail['星期']}, "
                  f"时间:{detail['时间']}, 课程:{detail['课程1']}和{detail['课程2']}, "
                  f"冲突学生数:{detail['冲突学生数']}人")

    # 计算冲突率（有冲突的课程数/总选课数）
    conflict_rate = 0
    if total_course_selections > 0:
        conflict_rate = (total_conflict_courses / total_course_selections) * 100

    print("\n---------- 学生课程冲突统计 ----------")
    print(f"全体选课学生数: {len(total_students)}")
    print(f"有课程冲突的学生数: {len(conflict_students)}")
    print(f"所有学生选的所有课的门数: {total_course_selections}")
    print(f"所有学生有冲突的课的门数: {total_conflict_courses}")
    print(f"课程冲突率: {conflict_rate:.2f}%")

    return total_conflict_courses, conflict_rate, len(conflict_students), len(total_students)


def simulated_annealing_scheduling(
        enrollment_file='math_student_enrollment.xlsx',
        courses_file='df_final_cleaned_1.xlsx',
        rooms_file='Timetabling_KB_Rooms.xlsx',
        max_iterations=100000,
        initial_temperature=1000,
        cooling_rate=0.995,
        utilization_weight=1.0,
        conflict_weight=50000.0
):
    """
    使用模拟退火求解排课问题，支持处理不同活动类型和合并workshop：
      1. 大课程定义为：Real Size > Planned Size，这些课程强制安排在其指定教室；
      2. 处理不同活动类型（lecture, workshop等），相同course code的不同活动单独排课；
      3. 合并符合条件的workshop（相同course code、教学周模式、活动类型），合并后不超过120人；
      4. 输出中会展开合并的workshop，确保被合并的workshop使用相同的时间和教室；
      5. 确保课程的持续时间与原始数据一致，不会被分割成多个小时间段。
    """
    try:
        courses_df = pd.read_excel(courses_file)
        rooms_df = pd.read_excel(rooms_file)
        enrollment_df = pd.read_excel(enrollment_file)

        # 在课程数据中添加行索引，用于标识
        courses_df['Row_ID'] = courses_df.index

        # 创建教室名称的标准化映射
        rooms_df['Normalized_Name'] = rooms_df['ROOM NAME'].apply(normalize_room_name)

        # 创建从原始名称到标准化名称的映射
        room_name_map = {}
        for _, row in rooms_df.iterrows():
            room_name_map[normalize_room_name(row['ROOM NAME'])] = row['ROOM NAME']

        # 在课程数据中创建标准化的教室名称
        courses_df['Normalized_Room_Name'] = courses_df['ROOM NAME'].apply(normalize_room_name)

        # 显示课程持续时间的统计
        print("\n---------- 课程持续时间统计 ----------")
        duration_counts = courses_df['Duration'].value_counts()
        for duration, count in duration_counts.items():
            slots = parse_duration(duration)
            print(f"持续时间 {duration} ({slots} 个时间槽): {count} 个课程")

        # 预处理课程数据，包括处理不同活动类型和合并workshop
        processed_courses_df = preprocess_course_with_activities(courses_df)

        # 新增：根据条件将课程分为大课程与常规课程
        # 大课程定义为：Real Size > Planned Size
        large_course_df = processed_courses_df[processed_courses_df['Real Size'] > processed_courses_df['Planned Size']]
        regular_courses_df = processed_courses_df[
            processed_courses_df['Real Size'] <= processed_courses_df['Planned Size']]

        all_courses_df = processed_courses_df.copy()  # 保存所有课程数据用于最终能量计算

        print("\n---------- 课程与教室统计 ----------")
        print("处理后课程总数: {}".format(len(processed_courses_df)))
        print("大课程数量: {}".format(len(large_course_df)))
        print("常规课程数量: {}".format(len(regular_courses_df)))
        print("教室数量: {}".format(len(rooms_df)))
        print("学生选课记录: {}".format(len(enrollment_df)))

        # 打印教室名称匹配情况
        distinct_room_names = processed_courses_df['Normalized_Room_Name'].unique()
        matched_rooms = 0
        for room_name in distinct_room_names:
            if room_name in room_name_map:
                matched_rooms += 1

        print("大课程指定教室匹配率: {}/{} ({:.2f}%)".format(
            matched_rooms, len(distinct_room_names),
            (matched_rooms / len(distinct_room_names)) * 100 if len(distinct_room_names) > 0 else 0
        ))
    except Exception as e:
        print("数据加载失败: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return None

    # 构建课程与教室映射
    # 修改映射逻辑：使用课程索引作为键，而不是课程代码
    course_to_index = {}
    index_to_course = {}
    index_to_course_row = {}  # 新增：索引到课程行的映射
    regular_course_indices = set()
    large_course_indices = set()

    # 为每行课程创建唯一索引
    for i, row in all_courses_df.iterrows():
        course_code = row['Course Code']

        # 如果有活动类型，将其添加到课程代码中以区分不同活动
        if 'Activity Type Name' in row:
            activity_type = row['Activity Type Name']
            unique_id = f"{course_code}_{activity_type}_{i}"  # 使用行索引确保唯一性
        else:
            unique_id = f"{course_code}_{i}"

        # 课程索引从1开始
        idx_val = len(course_to_index) + 1
        course_to_index[unique_id] = idx_val
        index_to_course[idx_val] = course_code
        index_to_course_row[idx_val] = row.to_dict()  # 存储整行数据

        # 判断是大课程还是常规课程
        if row['Real Size'] > row.get('Planned Size', 0):
            large_course_indices.add(idx_val)
        else:
            regular_course_indices.add(idx_val)

    room_to_index = {room: idx + 1 for idx, room in enumerate(rooms_df['ROOM NAME'])}
    index_to_room = {idx + 1: room for idx, room in enumerate(rooms_df['ROOM NAME'])}

    # 根据enrollment文件构建课程与选课学生的映射
    course_students = {}
    try:
        if 'course_id' in enrollment_df.columns and 'student_id' in enrollment_df.columns:
            # 如果是math_student_enrollment.xlsx格式
            for _, row in enrollment_df.iterrows():
                course_code = row['course_id']
                student_id = row['student_id']
                if course_code not in course_students:
                    course_students[course_code] = set()
                course_students[course_code].add(student_id)
        elif 'Course ID' in enrollment_df.columns and 'Student ID' in enrollment_df.columns:
            # 如果是Anon Enrollment Data_new.xlsx格式
            for _, row in enrollment_df.iterrows():
                course_code = row['Course ID']
                student_id = row['Student ID']
                if course_code not in course_students:
                    course_students[course_code] = set()
                course_students[course_code].add(student_id)
        else:
            print("警告: 无法识别选课数据格式，学生课程冲突率将无法计算")
    except Exception as e:
        print(f"构建课程与学生映射时出错: {str(e)}")
        course_students = {}

    # --- 处理大课程 ---
    # 对于大课程，强制安排到指定教室
    large_courses_solution = {}
    for unique_id, idx_val in course_to_index.items():
        if idx_val not in large_course_indices:
            continue

        course_row = index_to_course_row[idx_val]
        normalized_room_name = course_row['Normalized_Room_Name']

        # 获取课程持续时间
        duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(duration)

        print(f"大课程 {unique_id} 持续时间: {duration} ({duration_slots} 个时间槽)")

        # 使用标准化后的教室名称查找匹配
        if normalized_room_name in room_name_map:
            # 找到匹配，使用标准化映射获取实际教室名称
            actual_room_name = room_name_map[normalized_room_name]
            designated_room_name = actual_room_name
        else:
            # 检查指定教室是否直接存在
            room_name = course_row['ROOM NAME']
            if room_name in room_to_index:
                designated_room_name = room_name
            else:
                print("警告：大课程 {} 的指定教室 {} 不存在，尝试寻找替代教室".format(
                    unique_id, room_name))

                # 寻找容量足够的替代教室
                course_size = course_row['Real Size']
                alternative_rooms = []
                for _, room in rooms_df.iterrows():
                    if room['CAP'] >= course_size:
                        alternative_rooms.append(room['ROOM NAME'])

                if alternative_rooms:
                    designated_room_name = random.choice(alternative_rooms)
                    print("  为大课程 {} 选择替代教室: {}".format(unique_id, designated_room_name))
                else:
                    print("  错误：大课程 {} 没有可用的替代教室".format(unique_id))
                    continue

        designated_room_idx = room_to_index[designated_room_name]

        # 为每个教学周安排一个时间
        teaching_weeks = course_row['Teaching_Weeks']
        for week in teaching_weeks:
            # 尝试多次寻找合适的时间
            max_attempts = 30
            success = False

            for _ in range(max_attempts):
                day = random.randint(1, 5)

                # 找出所有有效的开始时间槽（避开午休时间）
                valid_start_slots = []
                for potential_start in range(1, 19 - duration_slots + 1):
                    # 检查是否跨越午休时间
                    if potential_start <= 6 and potential_start + duration_slots > 6:
                        continue
                    if potential_start <= 9 and potential_start + duration_slots > 9:
                        continue
                    valid_start_slots.append(potential_start)

                if not valid_start_slots:
                    print(f"  警告: 无法为大课程 {unique_id} 找到有效的开始时间（持续时间: {duration_slots} 个时间槽)")
                    continue

                time_slot = random.choice(valid_start_slots)
                weekday_idx = 5 * (week - 1) + day

                # 使用元组(room_idx, duration_slots)而不是单个整数
                large_courses_solution[(idx_val, weekday_idx, time_slot)] = (designated_room_idx, duration_slots)
                success = True
                break

            if not success:
                print(f"  警告: 无法为大课程 {unique_id} 在第 {week} 周找到合适的时间（尝试 {max_attempts} 次）")

    # 构造blocked_slots：记录每个(weekday_idx, time_slot)内已被大课程占用的教室
    blocked_slots = {}
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in large_courses_solution.items():
        # 对课程占用的每个时间槽进行标记
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            key = (weekday_idx, curr_slot)
            if key not in blocked_slots:
                blocked_slots[key] = set()
            blocked_slots[key].add(room_idx)

    # --- 处理常规课程 ---
    print("\n生成常规课程初始解...")
    current_solution = generate_initial_solution_with_activities(regular_course_indices, index_to_course_row,
                                                                 rooms_df, room_to_index, blocked_slots)

    if current_solution is None:
        print("无法生成初始解，请检查约束条件")
        return None

    # 合并大课程解和常规课程解
    current_solution.update(large_courses_solution)

    # 创建索引映射和查找表
    room_name_to_row = {}
    for _, row in rooms_df.iterrows():
        room_name = row['ROOM NAME']
        room_name_to_row[room_name] = row

    # 计算初始解能量
    current_energy = calculate_energy(current_solution, index_to_course_row, room_name_to_row,
                                      course_to_index, room_to_index, index_to_course, index_to_room,
                                      course_students, utilization_weight, conflict_weight)

    best_solution = current_solution.copy()
    best_energy = current_energy

    print("初始解能量值: {}".format(current_energy))

    # 模拟退火主循环
    temperature = initial_temperature
    iteration = 0
    no_improvement = 0

    print("\n开始模拟退火...")
    start_time = time.time()

    while iteration < max_iterations and temperature > 0.1 and no_improvement < 10000:
        # 创建blocked_slots的深拷贝，用于生成邻域解
        temp_blocked_slots = copy.deepcopy(blocked_slots)

        # 生成邻域解（仅修改常规课程部分）
        new_solution = generate_neighbor_with_activities(current_solution, regular_course_indices,
                                                         index_to_course_row, rooms_df, room_to_index,
                                                         index_to_course, temp_blocked_slots, course_students)
        # 合并大课程解（大课程保持固定安排）
        new_solution.update(large_courses_solution)

        # 计算新解能量
        new_energy = calculate_energy(new_solution, index_to_course_row, room_name_to_row,
                                      course_to_index, room_to_index, index_to_course, index_to_room,
                                      course_students, utilization_weight, conflict_weight)

        # 计算能量差
        energy_delta = new_energy - current_energy

        # 接受准则
        if energy_delta < 0 or random.random() < math.exp(-energy_delta / temperature):
            current_solution = new_solution
            current_energy = new_energy
            blocked_slots = temp_blocked_slots  # 更新blocked_slots

            # 更新最佳解
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
                no_improvement = 0
                print("迭代 {}, 温度 {:.2f}, 找到更好解: {}".format(iteration, temperature, best_energy))
            else:
                no_improvement += 1
        else:
            no_improvement += 1

        # 冷却
        temperature *= cooling_rate
        iteration += 1

        # 定期输出状态
        if iteration % 1000 == 0:
            elapsed_time = time.time() - start_time
            print("迭代 {}, 温度 {:.2f}, 当前能量 {}, 最佳能量 {}, 用时 {:.2f}秒".format(
                iteration, temperature, current_energy, best_energy, elapsed_time))

    # --- 生成结果 ---
    print("\n模拟退火完成，总迭代次数: {}, 最终温度: {:.2f}".format(iteration, temperature))
    print("最佳解能量值: {}".format(best_energy))

    # 使用修改后的函数生成课表，支持活动类型和合并workshop
    schedule_df = convert_solution_to_schedule_with_activities(best_solution, all_courses_df, rooms_df,
                                                               index_to_course, index_to_room, index_to_course_row)

    output_file = 'sa_course_schedule_with_activities.xlsx'
    schedule_df.to_excel(output_file, index=False)
    print("排课完成，共安排 {} 个课程时段".format(len(schedule_df)))
    print("结果已保存到 {}".format(output_file))

    # --- 评估优化指标 ---
    # 计算平均教室利用率
    schedule_df['使用率'] = (schedule_df['Class Size'] / schedule_df['Room Capacity'] * 100).round(2)
    avg_utilization = schedule_df['使用率'].mean()

    # 验证课程持续时间是否一致
    duration_check = {}
    for _, row in schedule_df.iterrows():
        course_code = row['Course Code']
        duration = row['Duration']
        duration_slots = row['Duration Slots']

        if course_code not in duration_check:
            duration_check[course_code] = {}

        if duration not in duration_check[course_code]:
            duration_check[course_code][duration] = set()

        duration_check[course_code][duration].add(duration_slots)

    duration_errors = []
    for course, durations in duration_check.items():
        for duration, slots_set in durations.items():
            if len(slots_set) > 1:
                duration_errors.append((course, duration, list(slots_set)))

    if duration_errors:
        print("\n警告: 发现持续时间不一致的课程:")
        for course, duration, slots in duration_errors:
            print(f"  课程 {course}, 原始持续时间 {duration}, 解析为不同的时间槽: {slots}")
    else:
        print("\n所有课程的持续时间都一致")

    # 计算学生课程冲突指标，返回额外的冲突详情
    conflict_count, conflict_rate, conflict_students, total_students, conflict_details = compute_student_conflict_with_activities(
        schedule_df, course_students)

    print("\n优化指标:")
    print("平均教室利用率: {:.2f}%".format(avg_utilization))
    print("学生课程冲突总数: {}".format(conflict_count))
    print("有冲突的学生数: {} (共{}名学生)".format(conflict_students, total_students))
    print("学生冲突率: {:.2f}%".format(conflict_rate))

    # 创建冲突信息的DataFrame
    if conflict_details:
        conflict_df = pd.DataFrame(conflict_details)

        # 创建ExcelWriter对象，使用已创建的文件
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
            # 将冲突信息写入新的工作表
            conflict_df.to_excel(writer, sheet_name='课程冲突详情', index=False)

            # 创建持续时间检查工作表
            if duration_errors:
                duration_error_data = []
                for course, duration, slots in duration_errors:
                    duration_error_data.append({
                        '课程代码': course,
                        '原始持续时间': duration,
                        '解析为时间槽': ', '.join(map(str, slots))
                    })
                pd.DataFrame(duration_error_data).to_excel(writer, sheet_name='持续时间检查', index=False)

            # 创建优化指标总结工作表
            summary_data = {
                '指标': ['平均教室利用率', '学生课程冲突总数', '有冲突的学生数', '总学生数', '学生冲突率'],
                '值': [
                    f"{avg_utilization:.2f}%",
                    conflict_count,
                    conflict_students,
                    total_students,
                    f"{conflict_rate:.2f}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='优化指标总结', index=False)

        print(f"冲突详情和优化指标已写入 {output_file}")
    else:
        print("没有发现课程冲突")

    return schedule_df, course_students, conflict_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="课程排课系统（支持不同活动类型和合并workshop）")
    parser.add_argument('--enrollment_file', type=str, default='math_student_enrollment.xlsx',
                        help='选课数据文件路径')
    parser.add_argument('--courses_file', type=str, default='df_final_cleaned_1.xlsx',
                        help='课程数据文件路径')
    parser.add_argument('--rooms_file', type=str, default='Timetabling_KB_Rooms.xlsx',
                        help='教室数据文件路径')
    parser.add_argument('--max_iterations', type=int, default=200000,
                        help='最大迭代次数')
    parser.add_argument('--initial_temperature', type=float, default=1000,
                        help='初始温度')
    parser.add_argument('--cooling_rate', type=float, default=0.995,
                        help='冷却率')
    parser.add_argument('--utilization_weight', type=float, default=1,
                        help='教室利用率权重')
    parser.add_argument('--conflict_weight', type=float, default=50000,
                        help='学生课程冲突权重')
    args = parser.parse_args()

    result_schedule, course_students, conflict_details = simulated_annealing_scheduling(
        enrollment_file=args.enrollment_file,
        courses_file=args.courses_file,
        rooms_file=args.rooms_file,
        max_iterations=300000,  # 增加迭代次数
        initial_temperature=5000,  # 提高初始温度
        cooling_rate=0.9995,  # 更慢的冷却速率
        utilization_weight=1.0,  # 保持教室利用率权重
        conflict_weight=100.0  # 显著降低冲突权重，使其成为真正的软约束
    )

    conflict_count, conflict_rate, num_conflict_students, total_students, _ = compute_student_conflict_with_activities(
        result_schedule, course_students)

    print("全体学生数:", total_students)
    print("冲突学生数:", num_conflict_students)
    print("累计冲突计数:", conflict_count)
    print("学生课程冲突率: {:.2f}%".format(conflict_rate))