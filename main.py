# 数据处理：从原始Anon Enrollment Data_new.xlsx中提取数院学生选课情况的数据，并输出为math_student_enrollment.xlsx
# 每个学生和每个课都有其对应的index，若学生s选择了i，则Y_si=1。(s和i都是index)
import pandas as pd
import numpy as np
import time
import re
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_math_student_enrollment(file_path):
    """
    处理数学院学生的选课数据（仅处理'Course Enrollments'工作表）

    参数:
        file_path (str): 选课数据Excel文件的路径

    返回:
        pandas.DataFrame: 数学院学生选课数据
        dict: 学生ID到索引的映射
        dict: 课程ID到索引的映射
        dict: 学生-课程映射
    """
    print(f"正在处理数学院学生选课数据: {file_path}")

    try:
        # 仅读取'Course Enrollments'工作表
        df = pd.read_excel(file_path, sheet_name='Course Enrollments')

        # 打印原始数据的基本信息
        print(f"原始数据形状: {df.shape}")
        print(f"原始数据列: {df.columns.tolist()}")

        # 筛选数学院的学生数据和MATH开头的课程
        math_students = df[
            (df['Programme School Name'] == 'School of Mathematics') &
            (df['Course Code'].str.startswith('MATH'))
            ].copy()
        print(f"数学院学生MATH课程数据形状: {math_students.shape}")

        # 提取所需的列: UNN、Course Code、Programme Of Study Sought Title等
        enrollment_data = math_students[['UNN', 'Course Code', 'Programme Of Study Sought Title']].copy()

        # 重命名列为标准格式
        enrollment_data = enrollment_data.rename(columns={
            'UNN': 'student_id',
            'Course Code': 'course_id',
            'Programme Of Study Sought Title': 'programme'
        })

        # 确保没有重复记录
        enrollment_data = enrollment_data.drop_duplicates()

        # 创建学生ID到索引的映射
        student_id_to_index = {student_id: i + 1 for i, student_id in enumerate(enrollment_data['student_id'].unique())}

        # 创建课程ID到索引的映射
        course_id_to_index = {course_id: i + 1 for i, course_id in enumerate(enrollment_data['course_id'].unique())}

        # 添加索引列
        enrollment_data['student_index'] = enrollment_data['student_id'].map(student_id_to_index)
        enrollment_data['course_index'] = enrollment_data['course_id'].map(course_id_to_index)

        print(f"处理后的数据形状: {enrollment_data.shape}")
        print(f"数学院学生人数: {len(student_id_to_index)}")
        print(f"MATH课程数量: {len(course_id_to_index)}")

        # 将结果保存为Excel文件，以便于后续使用
        output_file = 'math_student_enrollment.xlsx'
        enrollment_data.to_excel(output_file, index=False)
        print(f"数学院学生MATH课程选课数据已保存到: {output_file}")

        # 创建学生-课程映射
        student_courses = {}
        for _, row in enrollment_data.iterrows():
            student_idx = row['student_index']
            course_idx = row['course_index']
            if student_idx not in student_courses:
                student_courses[student_idx] = []
            student_courses[student_idx].append(course_idx)

        return enrollment_data, student_id_to_index, course_id_to_index, student_courses

    except Exception as e:
        print(f"处理数学院学生选课数据时出错: {str(e)}")
        return None, None, None, None


# 示例用法
if __name__ == "__main__":
    file_path = "/Users/ashley/Desktop/Topics in Applied OR/code_qp/topics/Anon Enrollment Data_new.xlsx"
    enrollment_data, student_id_to_index, course_id_to_index, student_courses = process_math_student_enrollment(
        file_path)

    if enrollment_data is not None:
        # 打印部分学生选课信息作为示例
        print("\n学生选课示例:")
        for student_idx, courses in list(student_courses.items())[:5]:  # 显示前5个学生的选课
            # 找回学生ID
            student_id = [sid for sid, idx in student_id_to_index.items() if idx == student_idx][0]
            # 找回课程ID
            course_ids = [cid for cid, idx in course_id_to_index.items() if idx in courses]
            print(f"学生 {student_id} 选了以下MATH课程: {course_ids}")


# 数据处理：处理教学周的问题
def preprocess_course_data(courses_df):
    """
    预处理课程数据

    参数:
    courses_df (pandas.DataFrame): 原始课程数据

    返回:
    pandas.DataFrame: 预处理后的课程数据
    """

    # 解析教学周模式
    def parse_week_pattern(pattern):
        if not isinstance(pattern, str):
            return []

        weeks = []
        parts = pattern.split(',')

        for part in parts:
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

    # 确定周期性
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

    # 添加解析后的教学周列表
    courses_df['Teaching_Weeks'] = courses_df['Teaching Week Pattern'].apply(parse_week_pattern)
    courses_df['Week_Parity'] = courses_df['Delivery Semester'].apply(determine_week_parity)

    # 应用单双周筛选到教学周
    for idx, row in courses_df.iterrows():
        if row['Week_Parity'] == 'ODD':
            courses_df.at[idx, 'Teaching_Weeks'] = [w for w in row['Teaching_Weeks'] if w % 2 == 1]
        elif row['Week_Parity'] == 'EVEN':
            courses_df.at[idx, 'Teaching_Weeks'] = [w for w in row['Teaching_Weeks'] if w % 2 == 0]

    return courses_df


def identify_workshop_merge_opportunities(courses_df):
    """
    识别潜在的workshop合并机会，特别关注 Real Size 的处理

    参数:
    courses_df (pandas.DataFrame): 预处理后的课程数据

    返回:
    list: 潜在合并的workshop组
    """
    # 创建时间标识
    courses_df['Day_Time'] = courses_df['Scheduled Days'] + '_' + courses_df['Scheduled Start Time']

    potential_merges = []
    course_codes = courses_df['Course Code'].unique()

    for course_code in course_codes:
        course_subset = courses_df[courses_df['Course Code'] == course_code]

        day_times = course_subset['Day_Time'].unique()
        for day_time in day_times:
            workshops = course_subset[course_subset['Day_Time'] == day_time]
            if len(workshops) > 1:
                week_patterns = workshops['Teaching Week Pattern'].unique()
                parities = workshops['Week_Parity'].unique()

                if len(week_patterns) == 1 and len(parities) == 1:
                    potential_merges.append({
                        'course_code': course_code,
                        'day_time': day_time,
                        'workshop_indices': workshops.index.tolist(),
                        'total_size': workshops['Real Size'].sum(),  # 使用 Real Size 求和
                        'merged_name': f"{course_code}_{day_time}_MERGED",
                        'week_pattern': week_patterns[0],
                        'parity': parities[0],
                        'teaching_weeks': workshops.iloc[0]['Teaching_Weeks']
                    })

    return potential_merges


def convert_weekday_to_week_and_day(weekday_index):
    """
    根据数学模型的weekday索引逻辑转换
    例如:
    11 -> (2, 1)  # 第2周的第1天
    12 -> (2, 2)  # 第2周的第2天
    """
    week = (weekday_index - 1) // 5 + 1
    day_in_week = (weekday_index - 1) % 5 + 1
    return week, day_in_week


# 数据处理：处理slot对应具体哪周哪天的问题
def convert_day_to_index(day_str):
    """将星期几字符串转换为索引(0-4)"""
    if not isinstance(day_str, str):
        return None

    day_str = day_str.lower()
    if 'mon' in day_str:
        return 0
    elif 'tue' in day_str:
        return 1
    elif 'wed' in day_str:
        return 2
    elif 'thu' in day_str:
        return 3
    elif 'fri' in day_str:
        return 4
    return None


def convert_time_to_slot(time_str):
    """将时间字符串(如'09:00')转换为时间段索引(1-18)"""
    if not isinstance(time_str, str):
        return None

    # 尝试解析"HH:MM"格式
    match = re.match(r'(\d{1,2}):(\d{2})', time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        # 计算时间段索引 (9:00为1, 9:30为2, 以此类推)
        if 9 <= hour < 18:
            slot = (hour - 9) * 2 + 1
            if minute >= 30:
                slot += 1
            return slot

    return None


def convert_slot_to_time(time_slot):
    """将时间段索引(1-18)转换为时间字符串(如'09:00')"""
    if not isinstance(time_slot, int) or time_slot < 1 or time_slot > 18:
        return None

    hour = 9 + ((time_slot - 1) // 2)
    minute = 30 if (time_slot - 1) % 2 == 1 else 0
    return f"{hour:02d}:{minute:02d}"


def multi_objective_course_scheduling_incremental(
        enrollment_file='math_student_enrollment.xlsx',
        courses_file='df_final_cleaned_1.xlsx',
        rooms_file='Timetabling_KB_Rooms.xlsx'
):
    """
    增量式课程排课优化模型：先找可行解，再优化
    """
    # 1. 数据加载与预处理（与原函数相同）
    try:
        # 加载数据
        courses_df = pd.read_excel(courses_file)
        rooms_df = pd.read_excel(rooms_file)
        enrollment_df = pd.read_excel(enrollment_file)

        # 预处理课程数据
        courses_df = preprocess_course_data(courses_df)

        print(f"课程数量: {len(courses_df)}")
        print(f"教室数量: {len(rooms_df)}")
        print(f"学生选课记录: {len(enrollment_df)}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None

    # 2. 创建优化模型
    model = cp_model.CpModel()

    # 3. 定义决策变量 Xijkt 与原函数相同
    X = {}

    # 4. 准备数据映射 与原函数相同
    course_to_index = {course: idx + 1 for idx, course in enumerate(courses_df['Course Code'])}
    room_to_index = {room: idx + 1 for idx, room in enumerate(rooms_df['ROOM NAME'])}

    # 5. 创建决策变量 与原函数相同
    for i, course in courses_df.iterrows():
        course_code = course['Course Code']
        course_idx = course_to_index[course_code]

        # 从第9周开始，即j从41开始（因为第9周第1天是 5*(9-1)+1 = 41）
        for j in range(41, 186):  # 从第9周到第37周的weekdays
            week, day_in_week = convert_weekday_to_week_and_day(j)

            if week not in course['Teaching_Weeks']:
                continue

            for k, room in rooms_df.iterrows():
                room_idx = room_to_index[room['ROOM NAME']]

                for t in range(1, 19):  # 时间槽
                    X[course_idx, j, room_idx, t] = model.NewBoolVar(
                        f'X_{course_idx}_{j}_{room_idx}_{t}'
                    )

    print(f"决策变量数量: {len(X)}")
    print(f"开始寻找可行解...")


    # 6. 添加所有约束条件（与原函数相同）
    # 约束1: 一个教室同一时间只能安排一门课程
    for j in range(41, 186):
        for room_idx in range(1, len(rooms_df) + 1):
            for t in range(1, 19):
                model.Add(
                    sum(X[course_idx, j, room_idx, t]
                        for course_idx in range(1, len(courses_df) + 1)
                        if (course_idx, j, room_idx, t) in X) <= 1
                )

    # 约束2: 一门课程同一时间不能在多个教室
    for course_idx in range(1, len(courses_df) + 1):
        for j in range(41, 186):
            model.Add(
                sum(X[course_idx, j, room_idx, t]
                    for room_idx in range(1, len(rooms_df) + 1)
                    for t in range(1, 19)
                    if (course_idx, j, room_idx, t) in X) <= 1
            )

    # 约束3: 教室容量约束
    for i, course in courses_df.iterrows():
        course_idx = course_to_index[course['Course Code']]
        course_real_size = course['Real Size']

        for j in range(41, 186):
            for k, room in rooms_df.iterrows():
                room_idx = room_to_index[room['ROOM NAME']]
                room_capacity = room['CAP']

                for t in range(1, 19):
                    if (course_idx, j, room_idx, t) in X:
                        model.Add(
                            course_real_size <= room_capacity + (1 - X[course_idx, j, room_idx, t]) * 10000
                        )

    # 约束5: 每周最多安排一次课
    for course_idx in range(1, len(courses_df) + 1):
        # 创建一个反向查找，找出course_idx对应的课程代码
        matching_courses = [k for k, v in course_to_index.items() if v == course_idx]

        # 如果找不到对应的课程代码，跳过这次循环
        if not matching_courses:
            continue

        course_code = matching_courses[0]

        # 查找该课程的教学周
        course_rows = courses_df[courses_df['Course Code'] == course_code]

        # 如果找不到对应的课程行，跳过这次循环
        if len(course_rows) == 0:
            continue

        course_teaching_weeks = course_rows['Teaching_Weeks'].values[0]

        for week in range(9, 38):  # 考虑9-37周
            week_start = 5 * (week - 1) + 1
            week_end = 5 * week

            # 如果该周不是教学周，则不应该安排课程
            if week not in course_teaching_weeks:
                model.Add(
                    sum(X[course_idx, j, room_idx, t]
                        for j in range(week_start, week_end + 1)
                        for room_idx in range(1, len(rooms_df) + 1)
                        for t in range(1, 19)
                        if (course_idx, j, room_idx, t) in X) == 0
                )

    # 7. 每门课程必须在其要求的教学周内安排一次，且每周仅安排一次
    for i, course in courses_df.iterrows():
        course_idx = course_to_index[course['Course Code']]
        teaching_weeks = course['Teaching_Weeks']

        # 对每个教学周添加恰好安排一次课的约束
        for week in teaching_weeks:
            week_start = 5 * (week - 1) + 1
            week_end = 5 * week

            # 每周恰好安排一次课（不是至少一次，而是恰好一次）
            model.Add(
                sum(X[course_idx, j, room_idx, t]
                    for j in range(week_start, week_end + 1)
                    for room_idx in range(1, len(rooms_df) + 1)
                    for t in range(1, 19)
                    if (course_idx, j, room_idx, t) in X) == 1
            )

    # 8. 先求可行解，不设置优化目标
    print("步骤1: 寻找可行解...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300  # 修改为5分钟，原为600
    solver.parameters.log_search_progress = True  # 开启日志

    # 定义回调函数，找到解后就停止
    class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__variables = variables
            self.__solution_count = 0
            self.__start_time = time.time()

        def on_solution_callback(self):
            self.__solution_count += 1
            current_time = time.time()
            print(f'找到第 {self.__solution_count} 个解！用时: {current_time - self.__start_time:.2f}秒')
            # 只找一个解就停止
            if self.__solution_count >= 1:
                self.StopSearch()

        def solution_count(self):
            return self.__solution_count

    # 执行第一步求解：只找可行解
    solution_printer = VarArraySolutionPrinter([X[key] for key in X])
    status = solver.Solve(model, solution_printer)

    # 如果找到可行解，保存结果
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print(f"找到可行解，状态: {status}")

        # 保存可行解变量值
        feasible_solution = {}
        for key in X:
            if solver.Value(X[key]) == 1:
                feasible_solution[key] = 1

        print(f"可行解包含 {len(feasible_solution)} 个安排")

        # 创建第二个模型，加入目标函数并使用可行解作为起点
        print("\n步骤2: 从可行解开始优化...")
        optimization_model = cp_model.CpModel()

        # 重新创建变量
        X_opt = {}
        for key in X:
            X_opt[key] = optimization_model.NewBoolVar(f'X_opt_{key}')

            # 设置可行解中的值作为提示
            if key in feasible_solution:
                optimization_model.AddHint(X_opt[key], 1)
            else:
                optimization_model.AddHint(X_opt[key], 0)

        # 添加所有约束（同上）
        # 约束1: 一个教室同一时间只能安排一门课程
        for j in range(1, 186):
            for room_idx in range(1, len(rooms_df) + 1):
                for t in range(1, 19):
                    optimization_model.Add(
                        sum(X_opt[course_idx, j, room_idx, t]
                            for course_idx in range(1, len(courses_df) + 1)
                            if (course_idx, j, room_idx, t) in X_opt) <= 1
                    )

        # 约束2: 一门课程同一时间不能在多个教室
        for course_idx in range(1, len(courses_df) + 1):
            for j in range(1, 186):
                optimization_model.Add(
                    sum(X_opt[course_idx, j, room_idx, t]
                        for room_idx in range(1, len(rooms_df) + 1)
                        for t in range(1, 19)
                        if (course_idx, j, room_idx, t) in X_opt) <= 1
                )

        # 约束3: 教室容量约束
        for i, course in courses_df.iterrows():
            course_idx = course_to_index[course['Course Code']]
            course_real_size = course['Real Size']

            for j in range(1, 186):
                for k, room in rooms_df.iterrows():
                    room_idx = room_to_index[room['ROOM NAME']]
                    room_capacity = room['CAP']

                    for t in range(1, 19):
                        if (course_idx, j, room_idx, t) in X_opt:
                            optimization_model.Add(
                                course_real_size <= room_capacity + (1 - X_opt[course_idx, j, room_idx, t]) * 10000
                            )

        # 约束5: 每周最多安排一次课
        for course_idx in range(1, len(courses_df) + 1):
            for week in range(1, 38):  # 37周
                week_start = 5 * (week - 1) + 1
                week_end = 5 * week

                optimization_model.Add(
                    sum(X_opt[course_idx, j, room_idx, t]
                        for j in range(week_start, week_end + 1)
                        for room_idx in range(1, len(rooms_df) + 1)
                        for t in range(1, 19)
                        if (course_idx, j, room_idx, t) in X_opt) <= 1
                )

        # 约束6: 每个课程至少安排一次
        for course_idx in range(1, len(courses_df) + 1):
            optimization_model.Add(
                sum(X_opt[course_idx, j, room_idx, t]
                    for j in range(1, 186)
                    for room_idx in range(1, len(rooms_df) + 1)
                    for t in range(1, 19)
                    if (course_idx, j, room_idx, t) in X_opt) >= 1
            )

        # 添加目标函数（简化版，只关注教室利用率）
        f1_terms = []
        for i, course in courses_df.iterrows():
            course_idx = course_to_index[course['Course Code']]
            course_real_size = course['Real Size']

            for j in range(1, 186):
                for k, room in rooms_df.iterrows():
                    room_idx = room_to_index[room['ROOM NAME']]
                    room_capacity = room['CAP']

                    for t in range(1, 19):
                        if (course_idx, j, room_idx, t) in X_opt:
                            # 使用辅助变量来表示利用率
                            utilization_var = optimization_model.NewIntVar(0, 100,
                                                                           f'util_{course_idx}_{j}_{room_idx}_{t}')

                            # 当X_opt为1时计算利用率
                            scaled_size = optimization_model.NewIntVar(0, course_real_size * 100,
                                                                       f'scaled_{course_idx}_{j}_{room_idx}_{t}')
                            optimization_model.Add(scaled_size == course_real_size * 100).OnlyEnforceIf(
                                X_opt[course_idx, j, room_idx, t])
                            optimization_model.Add(scaled_size == 0).OnlyEnforceIf(
                                X_opt[course_idx, j, room_idx, t].Not())

                            # 使用AddDivisionEquality而不是直接除法
                            optimization_model.AddDivisionEquality(utilization_var, scaled_size, room_capacity)

                            # 添加到目标函数项，利用率越高越好（使用负号因为是最小化问题）
                            f1_terms.append(-utilization_var)

        # 设置最小化目标
        if f1_terms:
            optimization_model.Minimize(sum(f1_terms))

        # 优化阶段的求解器
        opt_solver = cp_model.CpSolver()
        opt_solver.parameters.max_time_in_seconds = 1800  # 30分钟优化
        opt_solver.parameters.log_search_progress = True

        print("开始优化...")
        opt_status = opt_solver.Solve(optimization_model)

        # 处理优化结果
        if opt_status == cp_model.OPTIMAL or opt_status == cp_model.FEASIBLE:
            print(f"优化完成，状态: {opt_status}")

            # 提取排课结果
            schedule = []
            for i, course in courses_df.iterrows():
                course_idx = course_to_index[course['Course Code']]

                for j in range(1, 186):
                    for k, room in rooms_df.iterrows():
                        room_idx = room_to_index[room['ROOM NAME']]

                        for t in range(1, 19):
                            if (course_idx, j, room_idx, t) in X_opt and opt_solver.Value(
                                    X_opt[course_idx, j, room_idx, t]) == 1:
                                # 计算周和日期
                                week, day = convert_weekday_to_week_and_day(j)

                                schedule.append({
                                    'Course Code': course['Course Code'],
                                    'Course Name': course['Course Name'] if 'Course Name' in course else '',
                                    'Week': week,
                                    'Day': day,
                                    'Weekday Index': j,
                                    'Room': room['ROOM NAME'],
                                    'Room Capacity': room['CAP'],
                                    'Time Slot': t,
                                    'Time': convert_slot_to_time(t),
                                    'Class Size': course['Real Size']
                                })

            # 保存结果
            schedule_df = pd.DataFrame(schedule)
            output_file = 'incremental_course_schedule.xlsx'
            schedule_df.to_excel(output_file, index=False)

            print(f"排课完成，共安排 {len(schedule)} 个课程时段")
            print(f"结果已保存到 {output_file}")

            return schedule_df
        else:
            print(f"优化阶段未找到解，状态: {opt_status}")
            return None
    else:
        print(f"无法找到可行解，状态: {status}")
        return None


# 调用增量式排课函数
result = multi_objective_course_scheduling_incremental(
    enrollment_file='math_student_enrollment.xlsx',
    courses_file='df_final_cleaned_1.xlsx',
    rooms_file='Timetabling_KB_Rooms.xlsx'
)

# 处理结果
if result is not None:
    # 打印前几行排课结果
    print("\n排课结果示例:")
    print(result.head())

    # 基本统计信息
    print("\n排课统计信息:")
    print(f"总排课数量: {len(result)}")
    print(f"涉及课程数: {result['Course Code'].nunique()}")
    print(f"使用教室数: {result['Room'].nunique()}")

    # 按课程分组查看排课情况
    course_distribution = result.groupby('Course Code').size().reset_index(name='排课次数')
    print("\n各课程排课数量:")
    print(course_distribution.head(10))  # 显示前10个课程

    # 教室利用率分析
    result['使用率'] = (result['Class Size'] / result['Room Capacity'] * 100).round(2)
    avg_utilization = result['使用率'].mean()
    print(f"\n平均教室利用率: {avg_utilization:.2f}%")
    print(f"最高教室利用率: {result['使用率'].max():.2f}%")
    print(f"最低教室利用率: {result['使用率'].min():.2f}%")

    # 按周和日分布统计
    week_distribution = result.groupby('Week').size()
    day_distribution = result.groupby('Day').size()
    print("\n按周分布:")
    print(week_distribution)
    print("\n按天分布 (1=周一, 2=周二, ...):")
    print(day_distribution)

    # 按时间段分布
    time_distribution = result.groupby('Time').size().reset_index(name='数量')
    print("\n按时间段分布:")
    print(time_distribution)

    # 保存详细结果到Excel
    detailed_file = 'course_schedule_analysis.xlsx'

    # 创建Excel写入器
    with pd.ExcelWriter(detailed_file) as writer:
        # 保存主排课表
        result.to_excel(writer, sheet_name='完整排课表', index=False)

        # 保存课程分布
        course_distribution.to_excel(writer, sheet_name='课程分布', index=False)

        # 保存利用率分析
        utilization_analysis = result.groupby('Room')['使用率'].agg(['mean', 'min', 'max']).reset_index()
        utilization_analysis.columns = ['教室', '平均利用率', '最低利用率', '最高利用率']
        utilization_analysis.to_excel(writer, sheet_name='教室利用率', index=False)

        # 保存时间分布
        pivot_table = pd.pivot_table(result,
                                     values='Course Code',
                                     index='Week',
                                     columns='Day',
                                     aggfunc='count',
                                     fill_value=0)
        pivot_table.to_excel(writer, sheet_name='周日分布')

    print(f"\n详细分析已保存到 {detailed_file}")
else:
    print("排课失败，未找到可行解")