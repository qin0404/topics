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
            # 默认使用所有教学周(1-12)，避免完全跳过课程
            return list(range(1, 13))

        weeks = []
        for part in pattern.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    weeks.extend(range(start, end + 1))
                except:
                    # 解析失败时添加日志，但不会完全跳过
                    print(f"警告: 无法解析教学周范围 '{part}'，使用默认值")
                    continue
            else:
                try:
                    weeks.append(int(part))
                except:
                    # 解析失败时添加日志
                    print(f"警告: 无法解析教学周 '{part}'，跳过此项")
                    continue

        # 检查解析结果，如果为空则使用默认值
        if not weeks:
            print(f"警告: 教学周解析结果为空，使用默认教学周(1-12)")
            return list(range(1, 13))

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
    - 相同Activity Type Name (workshop或computer workshop)
    - 合并后人数在30-120之间

    增加了lecture优先级标记，确保lecture被优先处理
    """
    # 先使用原有的preprocess_course_data处理教学周信息
    courses_df = preprocess_course_data(courses_df)

    # 确保Activity Type Name列存在
    if 'Activity Type Name' not in courses_df.columns:
        print("警告: 数据中缺少'Activity Type Name'列，无法处理不同活动类型")
        return courses_df

    # 创建一个新的DataFrame来存储处理后的结果
    processed_courses = []

    # 先处理lecture类型课程（直接添加并标记）
    lecture_df = courses_df[courses_df['Activity Type Name'].str.lower().str.contains('lecture', case=False)]
    for _, row in lecture_df.iterrows():
        course_data = row.to_dict()
        course_data['Is_Lecture'] = True  # 添加标记表示这是lecture
        course_data['Priority'] = 1  # 高优先级
        processed_courses.append(course_data)

    print(f"找到 {len(lecture_df)} 个lecture类型课程，设为最高优先级")

    # 处理其他非workshop非lecture类型课程
    other_df = courses_df[~(courses_df['Activity Type Name'].str.lower().str.contains('lecture', case=False) |
                            courses_df['Activity Type Name'].str.lower().str.contains('workshop', case=False))]

    for _, row in other_df.iterrows():
        course_data = row.to_dict()
        course_data['Is_Lecture'] = False
        course_data['Priority'] = 2  # 中等优先级
        processed_courses.append(course_data)

    print(f"找到 {len(other_df)} 个其他类型课程，设为中等优先级")

    # 处理workshop类型课程
    workshop_df = courses_df[courses_df['Activity Type Name'].str.lower().str.contains('workshop', case=False)]

    # 修改：只按课程代码和活动类型分组，不考虑教学周模式
    workshop_groups = []
    for (course_code, activity_type), group in workshop_df.groupby(['Course Code', 'Activity Type Name']):
        workshop_groups.append((course_code, None, activity_type, group))

    print(f"找到 {len(workshop_groups)} 组可能合并的workshop组")

    # 处理每个组
    for group_info in workshop_groups:
        course_code, week_pattern, activity_type, group = group_info

        # 转换为列表以便处理
        workshops = group.to_dict('records')

        # 如果只有一个workshop，直接添加
        if len(workshops) == 1:
            workshop_entry = workshops[0].copy()
            workshop_entry['Merged_ID'] = None
            workshop_entry['Is_Merged'] = False
            workshop_entry['Merged_Count'] = 1
            workshop_entry['Group_Index'] = 0
            workshop_entry['Is_Lecture'] = False
            workshop_entry['Priority'] = 3  # 最低优先级
            processed_courses.append(workshop_entry)
            continue

        print(f"处理 {course_code} {activity_type} 组, 共 {len(workshops)} 个workshop")

        # 尝试合并workshops
        merged_workshops = []
        current_group = []
        current_size = 0

        # 按照size排序，便于更好地组合
        sorted_workshops = sorted(workshops, key=lambda x: x['Real Size'])

        for workshop in sorted_workshops:
            # 更灵活的合并条件：允许30-120人之间的合并
            if current_size + workshop['Real Size'] <= 120 and (
                    current_size + workshop['Real Size'] >= 30 or not current_group):
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
                workshop_entry['Is_Lecture'] = False
                workshop_entry['Priority'] = 3  # 最低优先级
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
                main_workshop['Is_Lecture'] = False
                main_workshop['Priority'] = 3  # 最低优先级

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
    print(f"lecture课程数: {sum(1 for _, row in result_df.iterrows() if row.get('Is_Lecture', False))}")

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
    # 修改：降低硬约束权重
    hard_constraint_weight = 300

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
            # 修改：减轻容量不足的惩罚，按比例计算
            capacity_ratio = course_size / room_capacity
            if capacity_ratio < 1.2:  # 超出不到20%
                energy += (course_size - room_capacity) * hard_constraint_weight * 0.5
            else:
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
        # 检查是否跨越午休 - 修改：降低午休时间的惩罚
        if (time_slot <= 6 and time_slot + duration_slots > 6) or (time_slot <= 9 and time_slot + duration_slots > 9):
            energy += hard_constraint_weight * 0.1  # 减轻跨越午休的惩罚

        # 检查是否超出一天时间范围
        if time_slot + duration_slots > 19:
            energy += hard_constraint_weight  # 超出时间范围，添加惩罚

        # 约束6（新增）：确保课程持续时间与原始数据一致
        course_row = course_code_to_row.get(course_idx)
        if course_row is not None:
            expected_duration = parse_duration(course_row.get('Duration', '1:00'))
            if duration_slots != expected_duration:
                energy += hard_constraint_weight * 3  # 持续时间不一致，但降低惩罚

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
    energy += conflict_count * conflict_weight

    # 新增约束：确保课程在每周相同的时间 - 放宽约束
    course_day_time = {}
    for (course_idx, weekday_idx, time_slot), _ in solution.items():
        # 获取周和日信息
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        if course_idx not in course_day_time:
            course_day_time[course_idx] = {}

        if day not in course_day_time[course_idx]:
            course_day_time[course_idx][day] = set()

        course_day_time[course_idx][day].add(time_slot)

    # 检查每门课程是否在每周的相同时间 - 降低惩罚
    for course_idx, day_slots in course_day_time.items():
        for day, slots in day_slots.items():
            if len(slots) > 1:  # 同一门课在同一个星期几有多个不同的时间
                energy += len(slots) * hard_constraint_weight * 0.3  # 降低惩罚

    return energy


def generate_initial_solution_with_activities(regular_course_indices, index_to_course_row,
                                              rooms_df, room_to_index, blocked_slots):
    """
    为常规课程生成初始解，支持不同活动类型和合并的workshop。
    在生成过程中排除blocked_slots中已被大课程占用的教室。
    优化教室分配策略，提高整体教室利用率。
    增加了lecture优先安排逻辑，确保所有lecture被排入课表。

    返回格式：{(course_idx, weekday_idx, time_slot): (room_idx, duration_slots)}
    """
    solution = {}
    course_success_count = 0  # 成功排课的课程数
    total_week_count = 0  # 总教学周数
    successful_week_count = 0  # 成功排课的教学周数

    # 新增：跟踪被跳过的课程及原因
    skipped_courses = {'room_capacity': [], 'time_conflict': [], 'other': []}

    # 按优先级排序课程索引（确保lecture先被处理）
    lecture_indices = []
    non_lecture_indices = []

    for course_idx in regular_course_indices:
        course_row = index_to_course_row.get(course_idx)
        if course_row and course_row.get('Is_Lecture', False):
            lecture_indices.append(course_idx)
        else:
            non_lecture_indices.append(course_idx)

    print(f"优先处理 {len(lecture_indices)} 个lecture课程...")

    # 为lecture创建一个专用的blocked_slots副本
    lecture_blocked_slots = copy.deepcopy(blocked_slots)

    # 先处理所有lecture课程，确保它们被安排
    for course_idx in lecture_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"未知_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # 检查教学周是否为空
        if not teaching_weeks:
            print(f"警告: lecture课程 {course_code} 没有有效的教学周，使用默认教学周(1-12)")
            teaching_weeks = list(range(1, 13))

        # 获取课程持续时间
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # 记录原始持续时间和解析后的槽数，以便调试
        print(f"Lecture课程 {course_code} 原始持续时间: {original_duration}, 解析为 {duration_slots} 个时间槽")

        # 跟踪教室使用频率
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # 筛选容量足够的教室并按适配度排序（优化教室选择）
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size * 0.8:  # 允许使用容量略小的教室
                # 计算教室与课程人数的适配度（越接近1越好）
                fit_ratio = course_size / room['CAP']
                usage_count = room_usage_count.get(room_idx, 0)
                # 综合考虑适配度和使用频率
                suitable_rooms.append((room_idx, fit_ratio, usage_count))

        # 首先按适配度排序（优先选择最合适的），然后按使用频率排序（优先选择使用较少的）
        suitable_rooms.sort(key=lambda x: (abs(x[1] - 0.8), x[2]))
        suitable_rooms = [room_idx for room_idx, _, _ in suitable_rooms]

        # 如果找不到容量足够的教室，寻找容量最接近的教室
        if not suitable_rooms:
            print(f"警告: lecture课程 {course_code} (人数: {course_size}) 没有足够大的教室，尝试最接近的教室")

            # 按容量差异排序找出最接近的教室（优先选择容量略大于课程人数的）
            sorted_rooms = sorted(rooms_df.to_dict('records'),
                                  key=lambda x: (
                                  0 if x['CAP'] >= course_size * 0.7 else 1, abs(x['CAP'] - course_size)))
            if sorted_rooms:
                closest_room_idx = room_to_index[sorted_rooms[0]['ROOM NAME']]
                suitable_rooms = [closest_room_idx]
                print(
                    f"  选择容量为 {sorted_rooms[0]['CAP']} 的最接近教室给lecture课程 {course_code} (人数: {course_size})")
            else:
                print(f"  错误: 找不到任何教室给lecture课程 {course_code}，将使用最大教室")

                # 找到最大容量的教室
                largest_room = max(rooms_df.to_dict('records'), key=lambda x: x['CAP'])
                largest_room_idx = room_to_index[largest_room['ROOM NAME']]
                suitable_rooms = [largest_room_idx]
                print(
                    f"  选择最大教室 {largest_room['ROOM NAME']} (容量: {largest_room['CAP']}) 给lecture课程 {course_code}")

        total_week_count += len(teaching_weeks)
        week_success = False

        # 记录该课程的固定时间槽（确保每周相同时间上课）
        fixed_day = random.randint(1, 5)

        # 为这门课程选择一个固定的时间槽
        valid_fixed_slots = []
        for potential_start in range(1, 19 - duration_slots + 1):
            # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
            if potential_start <= 6 and potential_start + duration_slots > 6:
                # 修改：为lecture增加允许跨越午休的概率到90%
                if random.random() < 0.9:
                    valid_fixed_slots.append(potential_start)
                continue
            if potential_start <= 9 and potential_start + duration_slots > 9:
                # 修改：为lecture增加允许跨越午休的概率到90%
                if random.random() < 0.9:
                    valid_fixed_slots.append(potential_start)
                continue
            valid_fixed_slots.append(potential_start)

        # 如果找不到不跨午休的时间槽，使用所有可能的时间槽
        if not valid_fixed_slots:
            for potential_start in range(1, 19 - duration_slots + 1):
                valid_fixed_slots.append(potential_start)

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # 尝试多次寻找合适的时间和教室
            max_attempts = 1000  # 大幅增加lecture的尝试次数
            success = False

            # 使用固定的日期和时间槽
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            for attempt in range(max_attempts):
                # 检查该课程占用的所有时间槽是否都没有被阻塞
                all_slots_available = True
                available_rooms = list(suitable_rooms)  # 复制一份可用教室列表

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in lecture_blocked_slots:
                        # 移除该时间槽已被占用的教室
                        available_rooms = [r for r in available_rooms if r not in lecture_blocked_slots[key]]
                        if not available_rooms:
                            all_slots_available = False
                            break

                # 如果所有时间槽都有可用教室，则安排课程
                if all_slots_available and available_rooms:
                    # 优先选择最合适的教室
                    chosen_room = available_rooms[0]

                    # 将课程安排在连续的时间槽中，保存起始时间槽和持续时间
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # 更新blocked_slots，将该课程占用的所有时间槽都标记为已占用
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in lecture_blocked_slots:
                            lecture_blocked_slots[key] = set()
                        lecture_blocked_slots[key].add(chosen_room)

                    success = True
                    successful_week_count += 1
                    break

                # 如果当前日期和时间槽不可用，尝试更换教室或重新选择时间槽
                if attempt > max_attempts // 2:
                    # 尝试其他时间槽
                    day = random.randint(1, 5)

                    valid_start_slots = []
                    for potential_start in range(1, 19 - duration_slots + 1):
                        # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
                        if potential_start <= 6 and potential_start + duration_slots > 6:
                            # 在后半段尝试中几乎总是允许跨午休
                            if random.random() < 0.95:
                                valid_start_slots.append(potential_start)
                            continue
                        if potential_start <= 9 and potential_start + duration_slots > 9:
                            # 在后半段尝试中几乎总是允许跨午休
                            if random.random() < 0.95:
                                valid_start_slots.append(potential_start)
                            continue
                        valid_start_slots.append(potential_start)

                    if not valid_start_slots:
                        for potential_start in range(1, 19 - duration_slots + 1):
                            valid_start_slots.append(potential_start)

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # 如果尝试多次仍找不到合适的时间和教室，则强制安排
            if not success:
                print(f"警告: 无法为lecture课程 {course_code} 在第{week}周找到合适的时间和教室，强制安排")

                # 尝试不同的日期
                day = random.randint(1, 5)
                time_slot = random.randint(1, 15)  # 选择较早的时间，避免超出范围
                weekday_idx = 5 * (week - 1) + day

                # 找到一个最大的教室
                if rooms_df.empty:
                    print("严重错误: 没有教室信息！")
                    continue

                largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
                if largest_rooms:
                    largest_room_idx = room_to_index[largest_rooms[0]['ROOM NAME']]

                    # 强制安排，也许会导致冲突，但确保lecture被安排
                    solution[(course_idx, weekday_idx, time_slot)] = (largest_room_idx, duration_slots)

                    # 更新blocked_slots
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in lecture_blocked_slots:
                            lecture_blocked_slots[key] = set()
                        lecture_blocked_slots[key].add(largest_room_idx)

                    success = True
                    successful_week_count += 1
                    print(f"  强制安排lecture课程 {course_code} 在第{week}周，星期{day}，时间槽{time_slot}，使用最大教室")
                else:
                    print(f"  严重错误: 找不到任何教室给lecture课程 {course_code}")

            if success:
                week_success = True

        if week_success:
            course_success_count += 1
        else:
            print(f"严重错误: lecture课程 {course_code} 完全无法安排")

    # 更新主blocked_slots，包含所有已安排的lecture
    for key, rooms in lecture_blocked_slots.items():
        if key not in blocked_slots:
            blocked_slots[key] = set()
        blocked_slots[key].update(rooms)

    # 现在处理非lecture课程
    print(f"处理 {len(non_lecture_indices)} 个非lecture课程...")

    for course_idx in non_lecture_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"未知_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # 检查教学周是否为空
        if not teaching_weeks:
            print(f"警告: 课程 {course_code} 没有有效的教学周，使用默认教学周(1-12)")
            teaching_weeks = list(range(1, 13))

        # 获取课程持续时间
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # 记录原始持续时间和解析后的槽数，以便调试
        print(f"课程 {course_code} 原始持续时间: {original_duration}, 解析为 {duration_slots} 个时间槽")

        # 跟踪教室使用频率
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # 筛选容量足够的教室并按适配度排序（优化教室选择）
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size:
                # 计算教室与课程人数的适配度（越接近1越好）
                fit_ratio = course_size / room['CAP']
                usage_count = room_usage_count.get(room_idx, 0)
                # 综合考虑适配度和使用频率
                suitable_rooms.append((room_idx, fit_ratio, usage_count))

        # 首先按适配度排序（优先选择最合适的），然后按使用频率排序（优先选择使用较少的）
        suitable_rooms.sort(key=lambda x: (abs(x[1] - 0.8), x[2]))
        suitable_rooms = [room_idx for room_idx, _, _ in suitable_rooms]

        # 修改：如果找不到容量足够的教室，寻找容量最接近的教室
        if not suitable_rooms:
            activity_type = course_row.get('Activity Type Name', 'Unknown')
            print(f"警告: 课程 {course_code} ({activity_type}, 人数: {course_size}) 没有足够大的教室，尝试最接近的教室")

            # 按容量差异排序找出最接近的教室（优先选择容量略大于课程人数的）
            sorted_rooms = sorted(rooms_df.to_dict('records'),
                                  key=lambda x: (0 if x['CAP'] >= course_size else 1, abs(x['CAP'] - course_size)))
            if sorted_rooms:
                closest_room_idx = room_to_index[sorted_rooms[0]['ROOM NAME']]
                suitable_rooms = [closest_room_idx]
                print(f"  选择容量为 {sorted_rooms[0]['CAP']} 的最接近教室给课程 {course_code} (人数: {course_size})")
            else:
                print(f"  错误: 找不到任何教室给课程 {course_code}")
                skipped_courses['room_capacity'].append(course_code)
                continue

        total_week_count += len(teaching_weeks)
        week_success = False

        # 记录该课程的固定时间槽（确保每周相同时间上课）
        fixed_day = random.randint(1, 5)

        # 为这门课程选择一个固定的时间槽
        valid_fixed_slots = []
        for potential_start in range(1, 19 - duration_slots + 1):
            # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
            if potential_start <= 6 and potential_start + duration_slots > 6:
                # 修改：增加允许跨越午休的概率到75%
                if random.random() < 0.75:
                    valid_fixed_slots.append(potential_start)
                continue
            if potential_start <= 9 and potential_start + duration_slots > 9:
                # 修改：增加允许跨越午休的概率到75%
                if random.random() < 0.75:
                    valid_fixed_slots.append(potential_start)
                continue
            valid_fixed_slots.append(potential_start)

        # 如果找不到不跨午休的时间槽，使用所有可能的时间槽
        if not valid_fixed_slots:
            for potential_start in range(1, 19 - duration_slots + 1):
                valid_fixed_slots.append(potential_start)

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # 尝试多次寻找合适的时间和教室
            max_attempts = 500  # 增加尝试次数
            success = False

            # 使用固定的日期和时间槽
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            for attempt in range(max_attempts):
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
                    # 优先选择最合适的教室
                    chosen_room = available_rooms[0]

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
                    successful_week_count += 1
                    break

                # 如果当前日期和时间槽不可用，尝试更换教室或重新选择时间槽
                if attempt > max_attempts // 2:
                    # 尝试其他时间槽
                    day = random.randint(1, 5)

                    valid_start_slots = []
                    for potential_start in range(1, 19 - duration_slots + 1):
                        # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
                        if potential_start <= 6 and potential_start + duration_slots > 6:
                            # 修改：增加允许跨越午休的概率到75%
                            if random.random() < 0.75:
                                valid_start_slots.append(potential_start)
                            continue
                        if potential_start <= 9 and potential_start + duration_slots > 9:
                            # 修改：增加允许跨越午休的概率到75%
                            if random.random() < 0.75:
                                valid_start_slots.append(potential_start)
                            continue
                        valid_start_slots.append(potential_start)

                    if not valid_start_slots:
                        for potential_start in range(1, 19 - duration_slots + 1):
                            valid_start_slots.append(potential_start)

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # 如果尝试多次仍找不到合适的时间和教室，则记录警告
            if success:
                week_success = True
            else:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(f"警告: 无法为课程 {course_code} ({activity_type}) 在第{week}周找到合适的时间和教室")
                if course_code not in skipped_courses['time_conflict']:
                    skipped_courses['time_conflict'].append(course_code)

        if week_success:
            course_success_count += 1
        else:
            # 如果所有教学周都无法安排，添加到其他原因
            if course_code not in skipped_courses['room_capacity'] and course_code not in skipped_courses[
                'time_conflict']:
                skipped_courses['other'].append(course_code)

    print(f"\n初始解生成完成:")
    print(f"成功排课的课程数: {course_success_count}/{len(regular_course_indices)}")
    print(f"成功排课的教学周数: {successful_week_count}/{total_week_count}")
    print(f"初始解中的安排数: {len(solution)}")

    # 输出被跳过课程的统计
    print("\n被跳过课程统计:")
    print(f"因教室容量不足被跳过的课程: {len(skipped_courses['room_capacity'])} 门")
    print(f"因时间冲突被跳过的课程: {len(skipped_courses['time_conflict'])} 门")
    print(f"因其他原因被跳过的课程: {len(skipped_courses['other'])} 门")

    # 如果数量不多，列出被跳过的课程
    for reason, courses in skipped_courses.items():
        if courses and len(courses) <= 10:
            print(f"\n因{reason}被跳过的课程:")
            for course in courses:
                print(f"  - {course}")

    return solution


def generate_neighbor_with_activities(solution, regular_course_indices, index_to_course_row,
                                      rooms_df, room_to_index, index_to_course,
                                      blocked_slots, course_students):
    """
    生成邻域解，支持不同活动类型和合并的workshop：
    优先选择有冲突的课程进行调整，修改其时间和/或教室，
    确保新选择的教室不被大课程占用。
    考虑课程的持续时间。
    放宽午休时间约束。
    优化教室分配策略，提高教室利用率。

    修改：过滤掉lecture类型课程，确保它们不会被模拟退火过程改变。
    """
    # 创建当前解的深拷贝，确保修改不会影响原解
    new_solution = copy.deepcopy(solution)

    if not solution:
        return new_solution

    # 过滤掉lecture课程，确保它们不会被修改
    non_lecture_keys = []
    for key in solution.keys():
        course_idx = key[0]
        if course_idx in regular_course_indices:
            course_row = index_to_course_row.get(course_idx)
            if course_row and not course_row.get('Is_Lecture', False):
                non_lecture_keys.append(key)

    # 如果没有非lecture的课程，直接返回原解
    if not non_lecture_keys:
        return new_solution  # 没有可修改的非lecture课程

    # 筛选出常规课程的键，确保这些键在当前解中存在
    regular_keys = [key for key in non_lecture_keys if key[0] in regular_course_indices]

    if not regular_keys:
        return new_solution

    # 识别有冲突的课程键
    conflict_keys, _ = identify_conflict_courses_with_activities(solution, course_students,
                                                                 index_to_course, index_to_course_row)
    # 筛选只属于常规课程的冲突键，并且不是lecture
    valid_conflict_keys = [key for key in conflict_keys if key in regular_keys]

    # 优先选择有冲突的课程进行调整
    if valid_conflict_keys and random.random() < 0.95:  # 95%概率选择冲突课程
        key = random.choice(valid_conflict_keys)
    else:
        # 随机选择任意常规非lecture课程
        key = random.choice(regular_keys)

    # 验证键是否存在于解决方案中
    if key not in new_solution:
        print(f"警告: 键 {key} 不存在于解决方案中。跳过此邻域生成。")
        return solution  # 返回原始解决方案

    course_idx, weekday_idx, time_slot = key
    room_idx, duration_slots = new_solution[key]

    # 计算周和日
    week, day = convert_weekday_to_week_and_day(weekday_idx)

    # 获取课程信息
    course_row = index_to_course_row.get(course_idx)
    if course_row is None:
        print(f"警告: 找不到课程索引 {course_idx} 的信息，跳过邻域生成")
        return new_solution

    course_size = course_row['Real Size']

    # 跟踪教室使用频率和适配度
    room_usage_count = {}
    for _, (r_idx, _) in solution.items():
        room_usage_count[r_idx] = room_usage_count.get(r_idx, 0) + 1

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

    # 寻找这门课程在其他周的排课情况
    course_day_times = {}
    for (c_idx, w_idx, t_slot), _ in solution.items():
        if c_idx == course_idx and w_idx != weekday_idx:
            _, d = convert_weekday_to_week_and_day(w_idx)
            if d not in course_day_times:
                course_day_times[d] = set()
            course_day_times[d].add(t_slot)

    if operation in (1, 3):  # 改变时间 或 同时改变时间和教室
        # 如果课程在其他周已有固定时间，优先使用该时间
        fixed_times = []
        for d, slots in course_day_times.items():
            for t in slots:
                fixed_times.append((d, t))

        if fixed_times and random.random() < 0.8:  # 80%概率使用已有的时间
            new_day, new_time_slot = random.choice(fixed_times)
        else:
            # 生成新的时间
            new_day = random.randint(1, 5)

            # 找出所有有效的开始时间槽，放宽午休限制
            valid_start_slots = []
            for potential_start in range(1, 19 - duration_slots + 1):
                # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
                if potential_start <= 6 and potential_start + duration_slots > 6:
                    # 放宽约束，有75%的概率允许跨越午休
                    if random.random() < 0.75:
                        valid_start_slots.append(potential_start)
                    continue
                if potential_start <= 9 and potential_start + duration_slots > 9:
                    # 放宽约束，有75%的概率允许跨越午休
                    if random.random() < 0.75:
                        valid_start_slots.append(potential_start)
                    continue
                valid_start_slots.append(potential_start)

            if not valid_start_slots:
                # 没有有效的开始时间，尝试所有可能的时间槽
                for potential_start in range(1, 19 - duration_slots + 1):
                    valid_start_slots.append(potential_start)

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
            # 修改：允许使用容量略小于课程人数的教室(最多小20%)，并计算适配度
            capacity_sufficient = room['CAP'] >= course_size * 0.8  # 允许容量达到课程人数的80%

            if capacity_sufficient and r_idx != room_idx:  # 排除当前教室
                # 计算教室适配度（教室容量与课程人数的比例，越接近1越好）
                fit_ratio = course_size / room['CAP']

                # 检查教室在所有时间槽是否都可用
                all_slots_available = True
                for slot_offset in range(duration_slots):
                    curr_slot = check_time_slot + slot_offset
                    block_key = (check_weekday_idx, curr_slot)
                    if block_key in current_blocked_slots and r_idx in current_blocked_slots[block_key]:
                        all_slots_available = False
                        break

                if all_slots_available:
                    # 添加教室索引、适配度和使用频率
                    suitable_rooms.append((r_idx, fit_ratio, room_usage_count.get(r_idx, 0)))

        # 首先按适配度排序（优先选择最合适的），然后按使用频率排序（优先选择使用较少的）
        suitable_rooms.sort(key=lambda x: (abs(x[1] - 0.8), x[2]))
        suitable_rooms = [r_idx for r_idx, _, _ in suitable_rooms]

        if suitable_rooms:
            new_room_idx = suitable_rooms[0]  # 选择最合适的教室

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
    修正：考虑课程持续时间导致的部分时间重叠冲突。

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

    # 按照学期、周、日分组课程
    semester_week_day_courses = {}

    # 收集每门课程占用的所有时间槽
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # 提取周和日信息
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 获取学期信息
        semester = get_normalized_semester(course_idx)

        # 使用三元组作为键：(学期, 周, 日)
        key = (semester, week, day)

        if key not in semester_week_day_courses:
            semester_week_day_courses[key] = []

        # 存储课程信息，包括持续时间
        semester_week_day_courses[key].append({
            'course_key': (course_idx, weekday_idx, time_slot),
            'course_idx': course_idx,
            'start_slot': time_slot,
            'end_slot': time_slot + duration_slots - 1,  # 最后占用的时间槽
            'duration': duration_slots
        })

    # 对于每个分组，检查课程时间重叠
    for key, courses in semester_week_day_courses.items():
        semester, week, day = key

        # 比较每对课程是否有时间重叠
        for i in range(len(courses)):
            course_i = courses[i]
            start_i = course_i['start_slot']
            end_i = course_i['end_slot']
            course_idx_i = course_i['course_idx']
            course_key_i = course_i['course_key']

            for j in range(i + 1, len(courses)):
                course_j = courses[j]
                start_j = course_j['start_slot']
                end_j = course_j['end_slot']
                course_idx_j = course_j['course_idx']
                course_key_j = course_j['course_key']

                # 检查时间是否重叠：一门课的开始≤另一门课的结束 且 一门课的结束≥另一门课的开始
                if start_i <= end_j and end_i >= start_j:
                    # 提取课程代码
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
    修改：输出的Course Code只包含基础课程代码，不包含活动类型。
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
        full_code = index_to_course[course_idx]
        # 只提取基础课程代码，不包含活动类型
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
        activity_type = course_row.get('Activity Type Name', 'Unknown')
        room_name = index_to_room[room_idx]

        # 找到对应的教室信息
        room_rows = rooms_df[rooms_df['ROOM NAME'] == room_name]
        if len(room_rows) == 0:
            continue
        room_row = room_rows.iloc[0]

        # 计算周和日
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # 计算结束时间 - 修正结束时间计算，确保与课程持续时间一致
        end_time_slot = time_slot + duration_slots
        end_time = convert_slot_to_time(end_time_slot)

        # 添加课程安排
        schedule.append({
            'Course Code': course_code,  # 只包含基础课程代码
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
        full_code = index_to_course[course_idx]
        # 只提取基础课程代码
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
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
                    'Course Code': course_code,  # 只包含基础课程代码
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
                'Course Code': course_code,  # 只包含基础课程代码
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

            # 创建DataFrame
        schedule_df = pd.DataFrame(schedule)

        # 添加教室利用率列
        if not schedule_df.empty:
            schedule_df['使用率'] = (schedule_df['Class Size'] / schedule_df['Room Capacity'] * 100).round(2)

        # 计算总体教室利用率
        avg_utilization = schedule_df['使用率'].mean() if not schedule_df.empty else 0
        print(f"排课表平均教室利用率: {avg_utilization:.2f}%")

        # 统计各教室的使用次数
        room_usage = schedule_df['Room'].value_counts() if not schedule_df.empty else pd.Series()
        print(f"使用的教室数量: {len(room_usage)} 个")

        # 将return语句移到函数最外层
        return schedule_df

def compute_student_conflict_with_activities(schedule_df, course_students):
    """
    计算排课表中学生课程冲突的总数与冲突率。
    新的逻辑：
    1. 检查每个学生在每个时间槽的课程数
    2. 同一门课的不同workshop不算冲突（学生只需要上一个）
    3. 超过1门不同课程的情况被视为冲突

    冲突率 = 冲突的时间槽总数 / 所有学生所有时间槽的课程总数
    """
    # 收集所有选课学生和统计总选课数
    total_students = set()
    total_course_selections = 0  # 所有学生选的所有课的门数

    # 计算总选课数和总学生数
    for course_code, students in course_students.items():
        total_students.update(students)
        total_course_selections += len(students)  # 每个课程的选课人数累加

    # 正则化学期信息
    def normalize_semester(semester_str):
        if not isinstance(semester_str, str):
            return "Unknown"
        semester_str = semester_str.lower()
        if 'semester1' in semester_str or 'semester 1' in semester_str:
            return "Semester1"
        elif 'semester2' in semester_str or 'semester 2' in semester_str:
            return "Semester2"
        else:
            return semester_str

    # 预处理：规范化学期信息
    schedule_df['Normalized_Semester'] = schedule_df['Delivery Semester'].apply(normalize_semester)

    # 创建学生-时间槽映射，记录每个学生在每个时间槽上课的课程和活动类型
    # 键格式：(学生ID, 学期, 周, 日, 时间槽)
    student_time_slot_courses = {}

    # 遍历排课表，为每个课程的每个时间槽记录选课学生
    for _, row in schedule_df.iterrows():
        # 提取基础课程代码，不包含活动类型
        full_code = row['Course Code']
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
        activity_type = row['Activity Type'].lower()  # 获取活动类型

        students = course_students.get(course_code, set())

        if not students:
            continue

        semester = row['Normalized_Semester']
        week = row['Week']
        day = row['Day']

        # 考虑课程持续时间，处理占用的所有时间槽
        start_slot = row['Start Time Slot']
        duration_slots = row['Duration Slots']

        for slot_offset in range(duration_slots):
            curr_slot = start_slot + slot_offset

            # 为该时间槽的每个学生记录课程信息，包括活动类型
            for student_id in students:
                key = (student_id, semester, week, day, curr_slot)
                if key not in student_time_slot_courses:
                    student_time_slot_courses[key] = []
                student_time_slot_courses[key].append((course_code, activity_type))

    # 计算冲突情况
    total_conflicts = 0  # 总冲突数
    conflict_students = set()  # 有冲突的学生集合
    conflict_details = []  # 冲突详情

    # 每个学生的冲突情况
    student_conflicts = {}

    # 统计每个时间槽的冲突，考虑到同一门课的不同workshop不计入冲突
    for (student_id, semester, week, day, time_slot), course_info_list in student_time_slot_courses.items():
        # 提取不同课程的基础代码（不考虑活动类型）
        unique_courses = set()
        workshop_courses = set()
        regular_courses = set()

        # 先区分workshop和其他类型的课程
        for course_code, activity_type in course_info_list:
            if 'workshop' in activity_type.lower():
                workshop_courses.add(course_code)
            else:
                regular_courses.add(course_code)
                unique_courses.add(course_code)

        # 将workshop课程添加到唯一课程集合中（每门课只计一次）
        unique_courses.update(workshop_courses)

        # 如果唯一课程数超过1，则存在冲突
        if len(unique_courses) > 1:
            conflicts_count = len(unique_courses) - 1  # 超出1门课的数量为冲突数
            total_conflicts += conflicts_count
            conflict_students.add(student_id)

            # 记录学生冲突次数
            if student_id not in student_conflicts:
                student_conflicts[student_id] = 0
            student_conflicts[student_id] += conflicts_count

            # 记录冲突详情
            start_time = convert_slot_to_time(time_slot)
            end_time = convert_slot_to_time(time_slot + 1)

            # 格式化冲突课程信息，分类显示
            conflict_course_str = []
            for course in regular_courses:
                conflict_course_str.append(f"{course}(常规)")
            if workshop_courses:
                conflict_course_str.append(f"{', '.join(workshop_courses)}(workshop)")

            conflict_details.append({
                "学生ID": student_id,
                "学期": semester,
                "教学周": week,
                "星期": day,
                "时间槽": time_slot,
                "时间": f"{start_time}-{end_time}",
                "冲突课程": ", ".join(conflict_course_str),
                "冲突数": conflicts_count
            })

    # 计算冲突率（冲突总数/总选课数）
    conflict_rate = 0
    if total_course_selections > 0:
        conflict_rate = (total_conflicts / total_course_selections) * 100

    # 输出详细的冲突信息
    if conflict_details:
        print("\n详细冲突信息:")
        # 按冲突数从大到小排序
        sorted_details = sorted(conflict_details, key=lambda x: x['冲突数'], reverse=True)
        for i, detail in enumerate(sorted_details[:10], 1):  # 只显示前10条
            print(f"{i}. 学生:{detail['学生ID']}, 学期:{detail['学期']}, 教学周:{detail['教学周']}, "
                  f"星期:{detail['星期']}, 时间:{detail['时间']}, "
                  f"冲突课程:{detail['冲突课程']}, 冲突数:{detail['冲突数']}")

        if len(sorted_details) > 10:
            print(f"... 共有 {len(sorted_details)} 条冲突记录")

    # 输出冲突学生统计
    if student_conflicts:
        # 找出冲突最多的前5位学生
        top_conflict_students = sorted(student_conflicts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n冲突最多的学生:")
        for student_id, count in top_conflict_students:
            print(f"  学生 {student_id}: {count} 个冲突")

    print("\n---------- 学生课程冲突统计 ----------")
    print(f"全体选课学生数: {len(total_students)}")
    print(f"有课程冲突的学生数: {len(conflict_students)}")
    print(f"所有学生选的所有课的门数: {total_course_selections}")
    print(f"总冲突数: {total_conflicts}")
    print(f"课程冲突率: {conflict_rate:.2f}%")

    return total_conflicts, conflict_rate, len(conflict_students), len(total_students), conflict_details

def simulated_annealing_scheduling(
        enrollment_file='math_student_enrollment.xlsx',
        courses_file='df_final_cleaned_1.xlsx',
        rooms_file='Timetabling_KB_Rooms.xlsx',
        max_iterations=300000,
        initial_temperature=10000,
        cooling_rate=0.997,
        utilization_weight=0.5,
        conflict_weight=100.0,
        use_room_optimization=True
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

        # 检查教学周是否为空的课程
        empty_weeks_courses = sum(1 for _, row in processed_courses_df.iterrows() if not row['Teaching_Weeks'])
        if empty_weeks_courses > 0:
            print(f"警告: 发现 {empty_weeks_courses} 门课程没有有效的教学周")

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

        # 打印教室容量分布
        print("\n教室容量分布:")
        capacity_bins = [0, 30, 60, 90, 120, 150, 200, 300, 500, 1000]
        capacity_counts = pd.cut(rooms_df['CAP'], bins=capacity_bins).value_counts().sort_index()
        for bin_range, count in capacity_counts.items():
            print(f"  {bin_range}: {count}个教室")

    except Exception as e:
        print("数据加载失败: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return None

    # 构建课程与教室映射
    # 修改映射逻辑：使用课程索引作为键，但保存完整信息
    course_to_index = {}
    index_to_course = {}
    index_to_course_row = {}  # 新增：索引到课程行的映射
    regular_course_indices = set()
    large_course_indices = set()
    course_index_map = {}  # 增加映射来保存所有索引信息

    # 为每行课程创建唯一索引
    for i, row in all_courses_df.iterrows():
        course_code = row['Course Code']

        # 如果有活动类型，将其添加到课程代码中以区分不同活动
        if 'Activity Type Name' in row:
            activity_type = row['Activity Type Name']
            full_code = f"{course_code}_{activity_type}"  # 完整代码包含活动类型
        else:
            full_code = course_code

        # 加上行号确保完全唯一
        unique_id = f"{full_code}_{i}"

        # 课程索引从1开始
        idx_val = len(course_to_index) + 1

        # 保存完整映射
        course_to_index[unique_id] = idx_val
        index_to_course[idx_val] = full_code  # 保存完整的课程代码和活动类型
        index_to_course_row[idx_val] = row.to_dict()  # 存储整行数据
        course_index_map[idx_val] = {'code': course_code, 'full_code': full_code, 'row_idx': i}

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
    large_course_failures = 0  # 记录失败数
    large_course_success = 0  # 记录成功数

    for unique_id, idx_val in course_to_index.items():
        if idx_val not in large_course_indices:
            continue

        course_row = index_to_course_row[idx_val]
        course_code = course_row['Course Code']
        normalized_room_name = course_row['Normalized_Room_Name']
        course_size = course_row['Real Size']

        # 获取课程持续时间
        duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(duration)

        print(f"大课程 {unique_id} 持续时间: {duration} ({duration_slots} 个时间槽)")

        # 确定使用的教室
        designated_room_name = None

        # 1. 尝试标准化名称匹配
        if normalized_room_name in room_name_map:
            designated_room_name = room_name_map[normalized_room_name]

        # 2. 尝试直接匹配
        elif course_row['ROOM NAME'] in room_to_index:
            designated_room_name = course_row['ROOM NAME']

        # 3. 寻找容量足够的替代教室
        else:
            print(f"警告：大课程 {course_code} 的指定教室 {course_row['ROOM NAME']} 不存在，尝试寻找替代教室")

            alternative_rooms = []
            for _, room in rooms_df.iterrows():
                if room['CAP'] >= course_size:
                    alternative_rooms.append(room['ROOM NAME'])

            if alternative_rooms:
                designated_room_name = random.choice(alternative_rooms)
                print(f"  为大课程 {course_code} 选择替代教室: {designated_room_name}")

        # 4. 如果仍找不到，尝试使用最大容量教室
        if not designated_room_name:
            largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
            if largest_rooms:
                designated_room_name = largest_rooms[0]['ROOM NAME']
                print(
                    f"  警告：大课程 {course_code} (人数:{course_size}) 使用容量为 {largest_rooms[0]['CAP']} 的最大教室")
            else:
                print(f"  错误：大课程 {course_code} 没有可用的教室，跳过此课程")
                large_course_failures += 1
                continue

        designated_room_idx = room_to_index[designated_room_name]

        # 为每个教学周安排一个时间
        teaching_weeks = course_row['Teaching_Weeks']
        if not teaching_weeks:
            print(f"  警告：大课程 {course_code} 没有有效的教学周，使用默认教学周(1-12)")
            teaching_weeks = list(range(1, 13))

        week_success_count = 0

        for week in teaching_weeks:
            # 尝试多次寻找合适的时间
            max_attempts = 500  # 增加尝试次数
            success = False

            for attempt in range(max_attempts):
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

                # 如果找不到不跨午休的时间槽，放宽这个限制（尝试任意时间槽）
                if not valid_start_slots and attempt >= max_attempts // 2:
                    for potential_start in range(1, 19 - duration_slots + 1):
                        valid_start_slots.append(potential_start)
                    print(f"  警告: 大课程 {course_code} 允许跨越午休时间安排")

                if not valid_start_slots:
                    continue

                time_slot = random.choice(valid_start_slots)
                weekday_idx = 5 * (week - 1) + day

                # 使用元组(room_idx, duration_slots)而不是单个整数
                large_courses_solution[(idx_val, weekday_idx, time_slot)] = (designated_room_idx, duration_slots)
                week_success_count += 1
                success = True
                break

            if not success:
                print(f"  警告: 无法为大课程 {course_code} 在第 {week} 周找到合适的时间（尝试 {max_attempts} 次）")

        if week_success_count > 0:
            large_course_success += 1
            print(f"  大课程 {course_code} 成功排课: {week_success_count}/{len(teaching_weeks)} 个教学周")

    print(f"\n大课程排课完成: {large_course_success}/{len(large_course_indices)} 门课程成功排课")
    if large_course_failures > 0:
        print(f"有 {large_course_failures} 门大课程因找不到合适教室而被跳过")

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

    # 修改为新的初始解生成函数
    def generate_initial_solution_improved(regular_course_indices, index_to_course_row,
                                           rooms_df, room_to_index, blocked_slots):
        """改进的初始解生成函数，更灵活地处理约束条件"""
        solution = {}
        course_success_count = 0  # 成功排课的课程数
        total_week_count = 0  # 总教学周数
        successful_week_count = 0  # 成功排课的教学周数

        for course_idx in regular_course_indices:
            course_row = index_to_course_row[course_idx]
            course_size = course_row['Real Size']
            course_code = course_row.get('Course Code', f"未知_{course_idx}")
            teaching_weeks = course_row['Teaching_Weeks']

            # 检查教学周是否为空
            if not teaching_weeks:
                print(f"警告: 课程 {course_code} 没有有效的教学周，使用默认教学周(1-12)")
                teaching_weeks = list(range(1, 13))

            # 获取课程持续时间
            original_duration = course_row.get('Duration', '1:00')
            duration_slots = parse_duration(original_duration)

            # 筛选容量足够的教室
            suitable_rooms = []
            for _, room in rooms_df.iterrows():
                room_idx = room_to_index[room['ROOM NAME']]
                if room['CAP'] >= course_size:
                    suitable_rooms.append(room_idx)

            # 如果找不到容量足够的教室，使用容量最大的教室
            if not suitable_rooms:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(
                    f"警告: 课程 {course_code} ({activity_type}, 人数: {course_size}) 没有足够大的教室，尝试使用最大教室")

                # 按容量排序找出最大教室
                largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
                if largest_rooms:
                    largest_room_idx = room_to_index[largest_rooms[0]['ROOM NAME']]
                    suitable_rooms = [largest_room_idx]
                    print(
                        f"  选择容量为 {largest_rooms[0]['CAP']} 的最大教室给课程 {course_code} (人数: {course_size})")
                else:
                    print(f"  错误: 找不到任何教室给课程 {course_code}")
                    continue

            total_week_count += len(teaching_weeks)
            week_success = False

            for week in teaching_weeks:
                # 尝试多次寻找合适的时间和教室
                max_attempts = 500  # 增加尝试次数
                success = False

                for attempt in range(max_attempts):
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

                    # 如果找不到不跨午休的时间槽，第二阶段尝试放宽这个限制
                    if not valid_start_slots and attempt >= max_attempts // 2:
                        for potential_start in range(1, 19 - duration_slots + 1):
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
                        successful_week_count += 1
                        break

                # 如果尝试多次仍找不到合适的时间和教室，则记录警告
                if success:
                    week_success = True
                else:
                    activity_type = course_row.get('Activity Type Name', 'Unknown')
                    print(f"警告: 无法为课程 {course_code} ({activity_type}) 在第{week}周找到合适的时间和教室")

            if week_success:
                course_success_count += 1

        print(f"\n初始解生成完成:")
        print(f"成功排课的课程数: {course_success_count}/{len(regular_course_indices)}")
        print(f"成功排课的教学周数: {successful_week_count}/{total_week_count}")
        print(f"初始解中的安排数: {len(solution)}")

        return solution

    # 使用改进的初始解生成函数
    current_solution = generate_initial_solution_improved(regular_course_indices, index_to_course_row,
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
        if energy_delta < 0:
            acceptance_probability = 1.0
        elif temperature > 0:
            try:
                exp_term = -energy_delta / temperature
                # 防止溢出，限制exp_term的最大值
                if exp_term < -700:  # Python的exp函数大约在exp(-710)时溢出
                    acceptance_probability = 0.0
                else:
                    acceptance_probability = math.exp(exp_term)
            except OverflowError:
                acceptance_probability = 0.0
        else:
            acceptance_probability = 0.0
        if energy_delta < 0 or random.random() < acceptance_probability:
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

    # 增加额外的统计分析
    course_stats = schedule_df['Course Code'].value_counts()
    room_stats = schedule_df['Room'].value_counts()
    activity_type_stats = schedule_df['Activity Type'].value_counts()

    print("\n----------- 排课结果详细统计 -----------")
    print(f"排课表中的条目总数: {len(schedule_df)}")
    print(f"不同课程代码数量: {len(course_stats)}")
    print(f"不同教室数量: {len(room_stats)}")
    print(f"不同活动类型数量: {len(activity_type_stats)}")

    # 输出前10个最常用教室
    print("\n最常用的教室:")
    for room, count in room_stats.head(10).items():
        print(f"  {room}: {count}个安排")

    # 检查原始数据中有多少课程没有被排课
    all_course_codes = set(all_courses_df['Course Code'].unique())
    scheduled_course_codes = set(schedule_df['Course Code'].unique())
    missed_courses = all_course_codes - scheduled_course_codes

    print(f"\n未被排课的课程数量: {len(missed_courses)}")
    if missed_courses:
        print("未被排课的部分课程示例:")
        for code in list(missed_courses)[:10]:
            print(f"  {code}")
        # 在函数结束前添加教室使用情况的统计
        used_rooms = set(schedule_df['Room'].unique())
        all_rooms = set(rooms_df['ROOM NAME'])
        unused_rooms = all_rooms - used_rooms

        print("\n----------- 教室使用情况统计 -----------")
        print(f"总共教室数: {len(all_rooms)}")
        print(f"已使用教室数: {len(used_rooms)}")
        print(f"未使用教室数: {len(unused_rooms)}")

        # 输出未使用的教室列表
        if unused_rooms:
            print("\n未被使用的教室:")
            for room in sorted(unused_rooms):
                if room in rooms_df['ROOM NAME'].values:
                    room_cap = rooms_df[rooms_df['ROOM NAME'] == room]['CAP'].values[0]
                    print(f"  {room} (容量: {room_cap})")

        # 输出每个教室的使用次数
        print("\n教室使用频率:")
        room_usage = schedule_df['Room'].value_counts()
        for room, count in room_usage.nlargest(20).items():
            if room in rooms_df['ROOM NAME'].values:
                room_cap = rooms_df[rooms_df['ROOM NAME'] == room]['CAP'].values[0]
                print(f"  {room} (容量: {room_cap}): 使用 {count} 次")

    if len(schedule_df) < len(all_courses_df) * 0.8:  # 少于80%的课程被安排
        print("警告: 排课数量过少。使用放宽的约束重试...")
        return simulated_annealing_scheduling(
            enrollment_file=enrollment_file,
            courses_file=courses_file,
            rooms_file=rooms_file,
            max_iterations=max_iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            conflict_weight=50.0,  # 降低冲突惩罚
            utilization_weight=0.5,  # 减少对利用率的强调
        )

    return schedule_df, course_students, conflict_details


def generate_initial_solution_improved(regular_course_indices, index_to_course_row,
                                       rooms_df, room_to_index, blocked_slots):
    """改进的初始解生成函数，更灵活地处理约束条件"""
    solution = {}
    course_success_count = 0  # 成功排课的课程数
    total_week_count = 0  # 总教学周数
    successful_week_count = 0  # 成功排课的教学周数

    for course_idx in regular_course_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"未知_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # 检查教学周是否为空
        if not teaching_weeks:
            print(f"警告: 课程 {course_code} 没有有效的教学周，使用默认教学周(1-12)")
            teaching_weeks = list(range(1, 13))

        # 获取课程持续时间
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # 跟踪教室使用频率
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # 筛选容量足够的教室并按使用频率排序
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size:
                suitable_rooms.append((room_idx, room_usage_count.get(room_idx, 0)))

        # 按使用频率排序，优先选择使用较少的教室
        suitable_rooms.sort(key=lambda x: x[1])
        suitable_rooms = [room_idx for room_idx, _ in suitable_rooms]

        # 如果找不到容量足够的教室，使用容量最大的教室
        if not suitable_rooms:
            activity_type = course_row.get('Activity Type Name', 'Unknown')
            print(f"警告: 课程 {course_code} ({activity_type}, 人数: {course_size}) 没有足够大的教室，尝试使用最大教室")

            # 按容量排序找出最大教室
            largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
            if largest_rooms:
                largest_room_idx = room_to_index[largest_rooms[0]['ROOM NAME']]
                suitable_rooms = [largest_room_idx]
                print(f"  选择容量为 {largest_rooms[0]['CAP']} 的最大教室给课程 {course_code} (人数: {course_size})")
            else:
                print(f"  错误: 找不到任何教室给课程 {course_code}")
                continue

        total_week_count += len(teaching_weeks)
        week_success = False

        # 为这门课选择固定的星期几和时间
        fixed_day = random.randint(1, 5)

        # 找出所有有效的开始时间槽（避开午休时间或有一定概率允许）
        valid_fixed_slots = []
        for potential_start in range(1, 19 - duration_slots + 1):
            # 检查是否跨越午休时间 (假设午休时间为12:00-13:30，对应时间槽7-9)
            if potential_start <= 6 and potential_start + duration_slots > 6:
                # 放宽约束，有50%的概率允许跨越午休
                if random.random() < 0.5:
                    valid_fixed_slots.append(potential_start)
                continue
            if potential_start <= 9 and potential_start + duration_slots > 9:
                # 放宽约束，有50%的概率允许跨越午休
                if random.random() < 0.5:
                    valid_fixed_slots.append(potential_start)
                continue
            valid_fixed_slots.append(potential_start)

        # 如果找不到不跨午休的时间槽，使用所有可能的时间槽
        if not valid_fixed_slots:
            for potential_start in range(1, 19 - duration_slots + 1):
                valid_fixed_slots.append(potential_start)

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # 使用固定的星期几和时间
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            # 尝试多次寻找合适的教室
            max_attempts = 500
            success = False

            for attempt in range(max_attempts):
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
                    # 优先选择使用较少的教室
                    chosen_room = available_rooms[0]

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
                    successful_week_count += 1
                    break

                # 如果当前时间槽不可用，尝试其他教室或时间
                if attempt > max_attempts // 2:
                    # 在后半段尝试中，尝试更换时间
                    day = random.randint(1, 5)

                    valid_start_slots = []
                    for potential_start in range(1, 19 - duration_slots + 1):
                        # 检查是否跨越午休时间
                        if potential_start <= 6 and potential_start + duration_slots > 6:
                            if random.random() < 0.5:
                                valid_start_slots.append(potential_start)
                            continue
                        if potential_start <= 9 and potential_start + duration_slots > 9:
                            if random.random() < 0.5:
                                valid_start_slots.append(potential_start)
                            continue
                        valid_start_slots.append(potential_start)

                    if not valid_start_slots:
                        for potential_start in range(1, 19 - duration_slots + 1):
                            valid_start_slots.append(potential_start)

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # 如果尝试多次仍找不到合适的时间和教室，则记录警告
            if success:
                week_success = True
            else:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(f"警告: 无法为课程 {course_code} ({activity_type}) 在第{week}周找到合适的时间和教室")

        if week_success:
            course_success_count += 1

    print(f"\n初始解生成完成:")
    print(f"成功排课的课程数: {course_success_count}/{len(regular_course_indices)}")
    print(f"成功排课的教学周数: {successful_week_count}/{total_week_count}")
    print(f"初始解中的安排数: {len(solution)}")

    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="课程排课系统（支持不同活动类型和合并workshop）")
    parser.add_argument('--enrollment_file', type=str, default='math_student_enrollment.xlsx',
                        help='选课数据文件路径')
    parser.add_argument('--courses_file', type=str, default='df_final_cleaned_1.xlsx',
                        help='课程数据文件路径')
    parser.add_argument('--rooms_file', type=str, default='Timetabling_KB_Rooms.xlsx',
                        help='教室数据文件路径')
    parser.add_argument('--max_iterations', type=int, default=100000,
                        help='最大迭代次数')
    parser.add_argument('--initial_temperature', type=float, default=5000,
                        help='初始温度')
    parser.add_argument('--cooling_rate', type=float, default=0.997,
                        help='冷却率')
    parser.add_argument('--utilization_weight', type=float, default=0.5,
                        help='教室利用率权重')
    parser.add_argument('--conflict_weight', type=float, default=100,
                        help='学生课程冲突权重')
    args = parser.parse_args()

    result_schedule, course_students, conflict_details = simulated_annealing_scheduling(
        enrollment_file=args.enrollment_file,
        courses_file=args.courses_file,
        rooms_file=args.rooms_file,
        max_iterations=100000,  # 增加迭代次数
        initial_temperature=5000,  # 提高初始温度
        cooling_rate=0.997,  # 更慢的冷却速率
        utilization_weight=0.5,  # 增加教室利用率权重
        conflict_weight=100  # 保持学生冲突权重为软约束
    )

    conflict_count, conflict_rate, num_conflict_students, total_students, _ = compute_student_conflict_with_activities(
        result_schedule, course_students)

    print("全体学生数:", total_students)
    print("冲突学生数:", num_conflict_students)
    print("累计冲突计数:", conflict_count)
    print("学生课程冲突率: {:.2f}%".format(conflict_rate))