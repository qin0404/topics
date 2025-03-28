import pandas as pd
import time
import random
import math
import argparse
import copy
import concurrent.futures


# Data processing: Handle teaching week issues
def preprocess_course_data(courses_df):
    """
    Preprocess course data, parse teaching week patterns and odd/even week information
    """

    def parse_week_pattern(pattern):
        """
        Preprocess course data, parse teaching week pattern, supporting complex non-contiguous week patterns.
        Improvement: Better handle whitespace, mixed single week and range week patterns, and add more error handling.
        """
        if not isinstance(pattern, str):
            # Default to using all teaching weeks (1-12) to avoid completely skipping the course
            print(f"Warning: Teaching week pattern is not a string '{pattern}', using default teaching weeks (1-12)")
            return list(range(1, 13))

        pattern = pattern.strip()  # Remove leading and trailing spaces
        if not pattern:  # If empty string
            print(f"Warning: Teaching week pattern is empty, using default teaching weeks (1-12)")
            return list(range(1, 13))

        weeks = []
        try:
            # Split by commas and remove spaces from each part
            for part in [p.strip() for p in pattern.split(',')]:
                if not part:  # Skip empty parts
                    continue

                if '-' in part:
                    # Process range notation (e.g., "18-19")
                    try:
                        start_str, end_str = [s.strip() for s in part.split('-')]
                        start, end = int(start_str), int(end_str)
                        if start > end:
                            print(f"Warning: Teaching week range invalid '{part}' (start greater than end), adjusted")
                            start, end = end, start
                        weeks.extend(range(start, end + 1))
                    except ValueError:
                        print(f"Warning: Unable to parse teaching week range '{part}', skipping this item")
                        continue
                else:
                    # Process individual week number (e.g., "10")
                    try:
                        weeks.append(int(part))
                    except ValueError:
                        print(f"Warning: Unable to parse teaching week '{part}', skipping this item")
                        continue
        except Exception as e:
            print(f"Warning: Error parsing teaching week pattern '{pattern}': {str(e)}, using default teaching weeks")
            return list(range(1, 13))

        # Check parsing result; if empty, use default value
        if not weeks:
            print(f"Warning: Teaching week parsing result is empty, using default teaching weeks (1-12)")
            return list(range(1, 13))

        # Remove duplicates and sort
        return sorted(set(weeks))

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


# Parse duration and convert to number of time slots
def parse_duration(duration_str):
    """Convert duration string (e.g., '1:00' or '2:00') to number of time slots"""
    if not isinstance(duration_str, str):
        return 2  # Default: 1 hour occupies 2 time slots

    try:
        # Parse string formatted as 'hours:minutes'
        if ':' in duration_str:
            hours, minutes = map(int, duration_str.split(':'))
            # Calculate total minutes and then divide by 30 minutes to get number of time slots
            slots = (hours * 60 + minutes) / 30
            # Ensure to return an integer number of time slots
            return int(round(slots))

        # It might also be a pure number representing hours
        elif duration_str.isdigit():
            return int(duration_str) * 2  # Each hour equals 2 time slots
    except:
        pass

    return 2  # Default value


# Process different activity types and merge workshops
def preprocess_course_with_activities(courses_df):
    """
    Preprocess course data, handle different activity types (lecture, workshop, etc.)
    and merge workshops that meet the criteria:
    - Same course code
    - Same Activity Type Name (workshop or computer workshop)
    - Merged total size between 30 and 120

    Added: Lecture priority flag to ensure lectures are processed first.
    New: Add designated room course flag, giving high priority.
    """
    # First, filter out courses with Real Size of 0
    original_count = len(courses_df)
    courses_df = courses_df[courses_df['Real Size'] > 0].copy()
    filtered_count = len(courses_df)
    if original_count > filtered_count:
        print(f"Filtered out {original_count - filtered_count} courses with Real Size of 0")

    # First, use the existing preprocess_course_data to process teaching week information
    courses_df = preprocess_course_data(courses_df)

    # Ensure that the 'Activity Type Name' column exists
    if 'Activity Type Name' not in courses_df.columns:
        print("Warning: 'Activity Type Name' column is missing in the data, cannot process different activity types")
        return courses_df

    # Create a new list to store processed courses
    processed_courses = []

    # First process lecture type courses (directly add and flag them)
    lecture_df = courses_df[courses_df['Activity Type Name'].str.lower().str.contains('lecture', case=False)]
    for _, row in lecture_df.iterrows():
        course_data = row.to_dict()
        course_data['Is_Lecture'] = True  # Add flag indicating this is a lecture
        course_data['Priority'] = 1  # High priority

        # Check if there is a designated room
        if pd.notna(row.get('ROOM NAME')) and row['ROOM NAME'] != '':
            course_data['Has_Designated_Room'] = True
        else:
            course_data['Has_Designated_Room'] = False

        processed_courses.append(course_data)

    print(f"Found {len(lecture_df)} lecture type courses, set to highest priority")

    # Process other non-workshop and non-lecture courses
    other_df = courses_df[~(courses_df['Activity Type Name'].str.lower().str.contains('lecture', case=False) |
                            courses_df['Activity Type Name'].str.lower().str.contains('workshop', case=False))]

    for _, row in other_df.iterrows():
        course_data = row.to_dict()
        course_data['Is_Lecture'] = False

        # New: Set higher priority for courses with a designated room
        if pd.notna(row.get('ROOM NAME')) and row['ROOM NAME'] != '':
            course_data['Priority'] = 1.5  # Priority between lecture (1) and others (2)
            course_data['Has_Designated_Room'] = True  # Flag designated room
        else:
            course_data['Priority'] = 2  # Default medium priority
            course_data['Has_Designated_Room'] = False

        processed_courses.append(course_data)

    print(f"Found {len(other_df)} other type courses, set to medium priority")

    # Process workshop type courses
    workshop_df = courses_df[courses_df['Activity Type Name'].str.lower().str.contains('workshop', case=False)]

    # Modification: Group only by course code and activity type, ignore teaching week pattern
    workshop_groups = []
    for (course_code, activity_type), group in workshop_df.groupby(['Course Code', 'Activity Type Name']):
        workshop_groups.append((course_code, None, activity_type, group))

    print(f"Found {len(workshop_groups)} groups of workshops that may be merged")

    # Process each group
    for group_info in workshop_groups:
        course_code, week_pattern, activity_type, group = group_info

        # Convert to list for processing
        workshops = group.to_dict('records')

        # If there is only one workshop, add directly
        if len(workshops) == 1:
            workshop_entry = workshops[0].copy()
            workshop_entry['Merged_ID'] = None
            workshop_entry['Is_Merged'] = False
            workshop_entry['Merged_Count'] = 1
            workshop_entry['Group_Index'] = 0
            workshop_entry['Is_Lecture'] = False
            workshop_entry['Priority'] = 3  # Lowest priority

            # Check if there is a designated room
            if pd.notna(workshop_entry.get('ROOM NAME')) and workshop_entry['ROOM NAME'] != '':
                workshop_entry['Has_Designated_Room'] = True
                workshop_entry['Priority'] = 1.5  # Increase priority
            else:
                workshop_entry['Has_Designated_Room'] = False

            processed_courses.append(workshop_entry)
            continue

        print(f"Processing group {course_code} {activity_type}, total {len(workshops)} workshops")

        # Try to merge workshops
        merged_workshops = []
        current_group = []
        current_size = 0

        # Sort by size to better group them
        sorted_workshops = sorted(workshops, key=lambda x: x['Real Size'])

        for workshop in sorted_workshops:
            # More flexible merging condition: allow merging for total size between 30 and 120
            if current_size + workshop['Real Size'] <= 120 and (
                    current_size + workshop['Real Size'] >= 30 or not current_group):
                current_group.append(workshop)
                current_size += workshop['Real Size']
            else:
                # Current group is full, save it and start a new group
                if current_group:
                    merged_workshops.append(current_group)
                current_group = [workshop]
                current_size = workshop['Real Size']

        # Add the last group
        if current_group:
            merged_workshops.append(current_group)

        print(f"  Merged into {len(merged_workshops)} groups")

        # Add merged workshops to the result
        for group_idx, group in enumerate(merged_workshops):
            if len(group) == 1:
                # Only one workshop, add directly
                workshop_entry = group[0].copy()
                workshop_entry['Merged_ID'] = None
                workshop_entry['Is_Merged'] = False
                workshop_entry['Merged_Count'] = 1
                workshop_entry['Group_Index'] = group_idx
                workshop_entry['Is_Lecture'] = False
                workshop_entry['Priority'] = 3  # Lowest priority

                # Check if there is a designated room
                if pd.notna(workshop_entry.get('ROOM NAME')) and workshop_entry['ROOM NAME'] != '':
                    workshop_entry['Has_Designated_Room'] = True
                    workshop_entry['Priority'] = 1.5  # Increase priority
                else:
                    workshop_entry['Has_Designated_Room'] = False

                processed_courses.append(workshop_entry)
            else:
                # Merge multiple workshops
                main_workshop = group[0].copy()  # Use the first workshop as the base

                # Record original IDs for later expansion
                original_ids = []
                for w in group:
                    if 'ID' in w:
                        original_ids.append(w['ID'])
                    else:
                        # If no ID, use row index
                        original_ids.append(f"{course_code}_{activity_type}_{w.get('Row_ID', 'unknown')}")

                # Update merge information
                main_workshop['Real Size'] = sum(w['Real Size'] for w in group)
                main_workshop['Merged_IDs'] = original_ids
                main_workshop['Is_Merged'] = True
                main_workshop['Merged_Count'] = len(group)
                main_workshop['Group_Index'] = group_idx
                main_workshop['Is_Lecture'] = False
                main_workshop['Priority'] = 3  # Lowest priority

                # Check if there is a designated room (if any merged workshop has one)
                has_designated_room = False
                designated_room = None
                for w in group:
                    if pd.notna(w.get('ROOM NAME')) and w['ROOM NAME'] != '':
                        has_designated_room = True
                        designated_room = w['ROOM NAME']
                        break

                if has_designated_room:
                    main_workshop['Has_Designated_Room'] = True
                    main_workshop['ROOM NAME'] = designated_room  # Keep the first found designated room
                    main_workshop['Priority'] = 1.5  # Increase priority
                else:
                    main_workshop['Has_Designated_Room'] = False

                # Record merged workshop details (for later expansion)
                merged_details = []
                for w in group:
                    merged_details.append({
                        'ID': w.get('ID', f"{course_code}_{activity_type}_{w.get('Row_ID', 'unknown')}"),
                        'Real Size': w['Real Size']
                    })
                main_workshop['Merged_Details'] = merged_details

                processed_courses.append(main_workshop)

    # Convert the processed results back to a DataFrame
    result_df = pd.DataFrame(processed_courses)

    # Calculate the number of courses with designated rooms
    designated_room_count = sum(1 for _, row in result_df.iterrows() if row.get('Has_Designated_Room', False))

    print(f"Total courses before processing: {len(courses_df)}")
    print(f"Total courses after processing: {len(result_df)}")
    print(f"Number of courses with designated rooms: {designated_room_count}")
    print(
        f"Number of merged workshop groups: {sum(1 for _, row in result_df.iterrows() if row.get('Is_Merged', False))}")
    print(f"Number of lecture courses: {sum(1 for _, row in result_df.iterrows() if row.get('Is_Lecture', False))}")

    return result_df


def convert_weekday_to_week_and_day(weekday_index):
    """
    Convert weekday index according to the mathematical model.
    For example: 11 -> (2, 1)
    """
    week = (weekday_index - 1) // 5 + 1
    day_in_week = (weekday_index - 1) % 5 + 1
    return week, day_in_week


def convert_slot_to_time(time_slot):
    """
    Convert time slot index (1-18) to a time string, e.g., '09:00'
    """
    if not isinstance(time_slot, int) or time_slot < 1 or time_slot > 19:
        return None
    hour = 9 + ((time_slot - 1) // 2)
    minute = 30 if (time_slot - 1) % 2 == 1 else 0
    return "{:02d}:{:02d}".format(hour, minute)


def normalize_room_name(room_name):
    """
    Normalize room name for matching:
    - Convert to lowercase
    - Remove extra spaces
    - Replace common separator variants
    """
    if not isinstance(room_name, str):
        return ""

    # Convert to lowercase
    normalized = room_name.lower()

    # Replace common separator variants
    normalized = normalized.replace(" - ", " ")
    normalized = normalized.replace("_", " ")

    # Remove extra spaces
    normalized = " ".join(normalized.split())

    return normalized


def calculate_energy(solution, course_code_to_row, room_name_to_row, course_to_index, room_to_index,
                     index_to_course, index_to_room, course_students, utilization_weight=2.0, conflict_weight=50.0):
    """
    Calculate the energy value of the solution:
      - Hard constraints (e.g., only one course per room at the same time, insufficient room capacity, each course scheduled only once per week, etc.) incur high penalties;
      - Soft constraints include: encouraging high room utilization and penalizing student course conflicts (multiple courses selected by a student scheduled at the same time).

    Update: Support courses with different durations and change student conflicts to a soft constraint.
    """
    energy = 0
    # Modification: Reduce hard constraint weight
    hard_constraint_weight = 9000

    # Hard Constraint 1: Only one course per room at the same time
    # Update to consider course duration
    room_time_conflicts = {}

    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # Check each time slot occupied by the course
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            key = (weekday_idx, curr_slot, room_idx)
            room_time_conflicts[key] = room_time_conflicts.get(key, 0) + 1

    for count in room_time_conflicts.values():
        if count > 1:
            energy += (count - 1) * hard_constraint_weight

    # Constraint 2: The same course cannot be scheduled in multiple rooms at the same time
    # This constraint does not need to change because the unique course index ensures it

    # Constraint 3: Insufficient room capacity
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # Check if course index and room index are valid
        if course_idx not in index_to_course or room_idx not in index_to_room:
            energy += hard_constraint_weight  # Invalid index, add penalty
            continue

        course_code = index_to_course[course_idx]
        course_row = course_code_to_row.get(course_idx)  # Using course index rather than course code
        if course_row is None:
            energy += hard_constraint_weight  # Course information not found, add penalty
            continue

        course_size = course_row['Real Size']
        room_name = index_to_room[room_idx]
        room_row = room_name_to_row.get(room_name)
        if room_row is None:
            energy += hard_constraint_weight  # Room information not found, add penalty
            continue

        room_capacity = room_row['CAP']
        if course_size > room_capacity:
            # Modification: Reduce penalty for insufficient capacity, calculate proportionally
            capacity_ratio = course_size / room_capacity
            if capacity_ratio < 1.2:  # Less than 20% over
                energy += (course_size - room_capacity) * hard_constraint_weight * 0.5
            else:
                energy += (course_size - room_capacity) * hard_constraint_weight

    # Constraint 4: Each course is scheduled only once per teaching week

    # Constraint 5: Ensure long courses do not span lunch break or the end of the day
    # Lunch break is from 12:00 to 13:30 (corresponding to time slots 7-9)

    # Soft Constraint 1: Room utilization (using provided weight)
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
        total_utilization += utilization * duration_slots  # Consider course duration
        count += duration_slots  # Count total time slots rather than courses

    if count > 0:
        avg_utilization = total_utilization / count
        # Encourage high utilization: (100% - average utilization) * weight
        energy += (100 - avg_utilization) * utilization_weight

    # Soft Constraint 2: Student course conflict rate
    # Create a detailed mapping from time slot to courses
    time_slot_courses = {}
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        course_code = index_to_course.get(course_idx)
        if not course_code:
            continue

        # Get the course's semester and week information
        course_row = course_code_to_row.get(course_idx)
        if not course_row:
            continue

        semester_str = course_row.get('Delivery Semester', 'Unknown')
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # Standardize semester information
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

        # Check each time slot occupied by the course
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            # Use a tuple (semester, week, day, time slot) as the key to ensure courses are compared in the same context
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
                    # Extract base course code (remove possible activity type suffix)
                    base_course_i = course_codes[i].split('_')[0] if '_' in course_codes[i] else course_codes[i]
                    base_course_j = course_codes[j].split('_')[0] if '_' in course_codes[j] else course_codes[j]

                    s1 = course_students.get(base_course_i, set())
                    s2 = course_students.get(base_course_j, set())
                    common_students = len(s1.intersection(s2))
                    conflict_count += common_students

    # Modification: Change student conflicts to a soft constraint, penalize proportionally to the number of conflicts
    energy += conflict_count * conflict_weight

    # New hard constraint: Ensure the course is scheduled at the same time each week - relaxed constraint
    course_day_time = {}
    for (course_idx, weekday_idx, time_slot), _ in solution.items():
        # Get week and day information
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        scheduled_time = (day, time_slot)
        if course_idx not in course_day_time:
            course_day_time[course_idx] = scheduled_time
        else:
            if course_day_time[course_idx] != scheduled_time:
                # Violation of hard constraint: the same course is scheduled at different times in different weeks.
                # Assign an extremely high penalty to ensure such a solution is not accepted.
                energy += hard_constraint_weight * 10 ** 9

    return energy


def generate_initial_solution_with_activities(regular_course_indices, index_to_course_row,
                                              rooms_df, room_to_index, blocked_slots):
    """
    Generate an initial solution for regular courses, supporting different activity types and merged workshops.
    Exclude rooms already occupied by large courses in blocked_slots during generation.
    Optimize room allocation strategy to improve overall room utilization.
    Added lecture priority scheduling logic to ensure all lectures are scheduled.
    New: Prioritize courses with designated rooms to ensure they use their designated room.

    Return format: {(course_idx, weekday_idx, time_slot): (room_idx, duration_slots)}
    """
    solution = {}
    course_success_count = 0  # Number of courses successfully scheduled
    total_week_count = 0  # Total number of teaching weeks
    successful_week_count = 0  # Number of teaching weeks successfully scheduled

    # New: Track skipped courses and reasons
    skipped_courses = {'room_capacity': [], 'time_conflict': [], 'other': []}

    # Sort course indices by priority (ensure lectures and non-lecture courses with designated rooms are processed first)
    lecture_indices = []
    designated_room_indices = []  # New: Non-lecture courses with designated rooms
    other_indices = []  # Other regular courses

    for course_idx in regular_course_indices:
        course_row = index_to_course_row.get(course_idx)
        if course_row and course_row.get('Is_Lecture', False):
            lecture_indices.append(course_idx)
        elif course_row and course_row.get('Has_Designated_Room', False):
            designated_room_indices.append(course_idx)
        else:
            other_indices.append(course_idx)

    print(f"Prioritizing processing of {len(lecture_indices)} lecture courses...")

    # Create a dedicated copy of blocked_slots for lectures
    lecture_blocked_slots = copy.deepcopy(blocked_slots)

    # First, process all lecture courses to ensure they are scheduled
    for course_idx in lecture_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"Unknown_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # Check if teaching weeks is empty
        if not teaching_weeks:
            print(
                f"Warning: Lecture course {course_code} does not have valid teaching weeks, using default teaching weeks (1-12)")
            teaching_weeks = list(range(1, 13))

        # Get course duration
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # Log original duration and parsed time slots for debugging
        print(
            f"Lecture course {course_code} original duration: {original_duration}, parsed as {duration_slots} time slots")

        # If the lecture has a designated room, use it preferentially
        designated_room_idx = None
        if course_row.get('Has_Designated_Room', False) and pd.notna(course_row.get('ROOM NAME')):
            room_name = course_row['ROOM NAME']
            if room_name in room_to_index:
                designated_room_idx = room_to_index[room_name]
                print(f"Lecture course {course_code} uses designated room: {room_name}")
            else:
                print(f"Warning: Designated room {room_name} for lecture course {course_code} does not exist")
        # Track room usage frequency
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # Filter rooms with sufficient capacity and sort by fit ratio (optimize room selection)
        suitable_rooms = []

        # If designated room exists, only use that room
        if designated_room_idx is not None:
            suitable_rooms = [designated_room_idx]
        else:
            for _, room in rooms_df.iterrows():
                room_idx = room_to_index[room['ROOM NAME']]
                if room['CAP'] >= course_size * 0.9:  # Allow using a room with slightly smaller capacity
                    # Calculate fit ratio (closer to 1 is better)
                    fit_ratio = course_size / room['CAP']
                    usage_count = room_usage_count.get(room_idx, 0)
                    # Consider both fit ratio and usage frequency
                    suitable_rooms.append((room_idx, fit_ratio, usage_count))

            # First sort by fit ratio (prefer best fit), then by usage frequency (prefer less used)
            suitable_rooms.sort(key=lambda x: (abs(x[1] - 0.8), x[2]))
            suitable_rooms = [room_idx for room_idx, _, _ in suitable_rooms]

        # If no room with sufficient capacity is found, try to find the room with closest capacity
        if not suitable_rooms:
            print(
                f"Warning: Lecture course {course_code} (size: {course_size}) does not have a large enough room, trying the closest room")
            # Sort by capacity difference to find the closest room (prefer rooms slightly larger than course size)
            sorted_rooms = sorted(rooms_df.to_dict('records'),
                                  key=lambda x: (
                                  0 if x['CAP'] >= course_size * 0.7 else 1, abs(x['CAP'] - course_size)))
            if sorted_rooms:
                closest_room_idx = room_to_index[sorted_rooms[0]['ROOM NAME']]
                suitable_rooms = [closest_room_idx]
                print(
                    f"  Selected closest room with capacity {sorted_rooms[0]['CAP']} for lecture course {course_code} (size: {course_size})")
            else:
                print(f"  Error: No room found for lecture course {course_code}, will use the largest room")
                # Find the room with the largest capacity
                largest_room = max(rooms_df.to_dict('records'), key=lambda x: x['CAP'])
                largest_room_idx = room_to_index[largest_room['ROOM NAME']]
                suitable_rooms = [largest_room_idx]
                print(
                    f"  Selected largest room {largest_room['ROOM NAME']} (capacity: {largest_room['CAP']}) for lecture course {course_code}")

        total_week_count += len(teaching_weeks)
        week_success = False

        # Record a fixed time slot for this course (to ensure it is scheduled at the same time each week)
        fixed_day = random.randint(1, 5)

        # Choose a fixed time slot for this course (ignoring lunch break issues)
        valid_fixed_slots = list(range(1, 19 - duration_slots + 1))

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # Try multiple times to find a suitable time and room
            max_attempts = 1000  # Greatly increased attempt count for lectures
            success = False

            # Use fixed day and time slot
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            for attempt in range(max_attempts):
                # Check if all time slots occupied by the course are not blocked
                all_slots_available = True
                available_rooms = list(suitable_rooms)  # Make a copy of available rooms list

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in lecture_blocked_slots:
                        # Remove rooms already occupied in this time slot
                        available_rooms = [r for r in available_rooms if r not in lecture_blocked_slots[key]]
                        if not available_rooms:
                            all_slots_available = False
                            break

                # If all time slots have an available room, schedule the course
                if all_slots_available and available_rooms:
                    # Prefer the most suitable room
                    chosen_room = available_rooms[0]

                    # Schedule the course in continuous time slots, saving the starting slot and duration
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # Update blocked_slots, mark all time slots occupied by this course as blocked
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in lecture_blocked_slots:
                            lecture_blocked_slots[key] = set()
                        lecture_blocked_slots[key].add(chosen_room)

                    success = True
                    successful_week_count += 1
                    break

                # If the current day and time slot are unavailable, try another day or time slot
                if attempt > max_attempts // 2:
                    # Try a different time slot
                    day = random.randint(1, 5)
                    valid_start_slots = list(range(1, 19 - duration_slots + 1))  # Ignore lunch break

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # If after many attempts no suitable time and room is found, force schedule
            if not success:
                print(
                    f"Warning: Unable to find a suitable time and room for lecture course {course_code} in week {week}, forcing schedule")

                # Try a different day
                day = random.randint(1, 5)
                time_slot = random.randint(1, 15)  # Choose an earlier time slot to avoid exceeding range
                weekday_idx = 5 * (week - 1) + day

                # Find the largest room
                if rooms_df.empty:
                    print("Severe error: No room information!")
                    continue

                # If designated room exists, force use that room
                if designated_room_idx is not None:
                    chosen_room = designated_room_idx
                else:
                    largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
                    chosen_room = room_to_index[largest_rooms[0]['ROOM NAME']]

                # Force schedule (may lead to conflicts but ensures the lecture is scheduled)
                solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                # Update blocked_slots
                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key not in lecture_blocked_slots:
                        lecture_blocked_slots[key] = set()
                    lecture_blocked_slots[key].add(chosen_room)

                success = True
                successful_week_count += 1
                print(
                    f"  Forced schedule for lecture course {course_code} in week {week}, day {day}, time slot {time_slot}, using room index {chosen_room}")

            if success:
                week_success = True

        if week_success:
            course_success_count += 1
        else:
            print(f"Severe error: Lecture course {course_code} could not be scheduled at all")

    # Update the main blocked_slots to include all scheduled lectures
    for key, rooms in lecture_blocked_slots.items():
        if key not in blocked_slots:
            blocked_slots[key] = set()
        blocked_slots[key].update(rooms)

    # Process non-lecture courses with designated rooms
    print(f"Processing {len(designated_room_indices)} non-lecture courses with designated rooms...")
    designated_room_success = 0

    for course_idx in designated_room_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"Unknown_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # Get the designated room
        designated_room_idx = None
        if pd.notna(course_row.get('ROOM NAME')):
            room_name = course_row['ROOM NAME']
            if room_name in room_to_index:
                designated_room_idx = room_to_index[room_name]
                print(f"Course {course_code} uses designated room: {room_name}")
            else:
                # Try to find a room matching after normalization
                normalized_name = normalize_room_name(room_name)
                for room, idx in room_to_index.items():
                    if normalize_room_name(room) == normalized_name:
                        designated_room_idx = idx
                        print(f"Course {course_code} uses normalized matched room: {room}")
                        break

                if designated_room_idx is None:
                    print(
                        f"Warning: Designated room {room_name} for course {course_code} does not exist, trying the closest room")
                    # Find the room with the closest capacity
                    sorted_rooms = sorted(rooms_df.to_dict('records'),
                                          key=lambda x: abs(x['CAP'] - course_size))
                    if sorted_rooms:
                        designated_room_idx = room_to_index[sorted_rooms[0]['ROOM NAME']]
                        print(f"  Selected closest room with capacity {sorted_rooms[0]['CAP']}")
                    else:
                        print(f"  Error: No available room found")
                        continue
        else:
            print(f"Error: Course {course_code} is marked as having a designated room, but no room name found")
            continue

        # Check if teaching weeks is empty
        if not teaching_weeks:
            print(
                f"Warning: Course {course_code} does not have valid teaching weeks, using default teaching weeks (1-12)")
            teaching_weeks = list(range(1, 13))

        # Get course duration
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # Log original duration and parsed time slots for debugging
        print(f"Course {course_code} original duration: {original_duration}, parsed as {duration_slots} time slots")

        total_week_count += len(teaching_weeks)
        week_success = False

        # Record a fixed time slot for this course (to ensure it is scheduled at the same time each week)
        fixed_day = random.randint(1, 5)

        # Choose a fixed time slot for this course, ignoring lunch break
        valid_fixed_slots = list(range(1, 19 - duration_slots + 1))

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # Try multiple times to find a suitable time
            max_attempts = 1000  # Greatly increased attempt count
            success = False

            # Use fixed day and time slot
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            for attempt in range(max_attempts):
                # Check if all time slots occupied by the course are not blocked
                all_slots_available = True

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in blocked_slots and designated_room_idx in blocked_slots[key]:
                        all_slots_available = False
                        break

                # If all time slots have an available room, schedule the course
                if all_slots_available:
                    # Schedule the course in continuous time slots, saving the starting slot and duration
                    solution[(course_idx, weekday_idx, time_slot)] = (designated_room_idx, duration_slots)

                    # Update blocked_slots, mark all time slots occupied by the course as blocked
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in blocked_slots:
                            blocked_slots[key] = set()
                        blocked_slots[key].add(designated_room_idx)

                    success = True
                    successful_week_count += 1
                    break

                # If the current day and time slot are unavailable, try another time
                if attempt > max_attempts // 2:
                    # Try a different time slot
                    day = random.randint(1, 5)
                    valid_start_slots = list(range(1, 19 - duration_slots + 1))

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            fixed_day = random.randint(1, 5)
            valid_fixed_slots = list(range(1, 19 - duration_slots + 1))
            fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

            for week in teaching_weeks:
                # (This loop block appears to be a placeholder for additional scheduling logic)
                max_attempts = 1000  # Greatly increased attempt count
                success = False

                # Use fixed day and time slot without re-randomizing each loop
                day = fixed_day
                time_slot = fixed_time_slot
                weekday_idx = 5 * (week - 1) + day

            # If after many attempts no suitable time is found, force schedule
            if not success:
                print(
                    f"Warning: Unable to find a suitable time for designated room course {course_code} in week {week}, forcing schedule")

                # Try a different day
                day = random.randint(1, 5)
                time_slot = random.randint(1, 15)  # Choose an earlier time slot
                weekday_idx = 5 * (week - 1) + day

                # Force schedule (may lead to conflicts but ensures the course is scheduled)
                solution[(course_idx, weekday_idx, time_slot)] = (designated_room_idx, duration_slots)

                # Update blocked_slots
                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key not in blocked_slots:
                        blocked_slots[key] = set()
                    blocked_slots[key].add(designated_room_idx)

                success = True
                successful_week_count += 1
                print(
                    f"  Forced schedule for designated room course {course_code} in week {week}, day {day}, time slot {time_slot}")

            if success:
                week_success = True

        if week_success:
            course_success_count += 1
            designated_room_success += 1
        else:
            print(f"Severe error: Designated room course {course_code} could not be scheduled at all")

        print(
            f"Successfully scheduled {designated_room_success}/{len(designated_room_indices)} courses with designated rooms")

    # Now process other regular courses
    print(f"Processing {len(other_indices)} regular courses...")

    for course_idx in other_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"Unknown_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # Check if teaching weeks is empty
        if not teaching_weeks:
            print(
                f"Warning: Course {course_code} does not have valid teaching weeks, using default teaching weeks (1-12)")
            teaching_weeks = list(range(1, 13))

        # Get course duration
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # Log original duration and parsed time slots for debugging
        print(f"Course {course_code} original duration: {original_duration}, parsed as {duration_slots} time slots")

        # Track room usage frequency
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # Filter rooms with sufficient capacity and sort by fit ratio (optimize room selection)
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size:
                # Calculate fit ratio (closer to 1 is better)
                fit_ratio = course_size / room['CAP']
                usage_count = room_usage_count.get(room_idx, 0)
                # Consider both fit ratio and usage frequency
                suitable_rooms.append((room_idx, fit_ratio, usage_count))

        # First sort by fit ratio then by usage frequency
        suitable_rooms.sort(key=lambda x: (abs(x[1] - 0.8), x[2]))
        suitable_rooms = [room_idx for room_idx, _, _ in suitable_rooms]

        # Modification: If no room with sufficient capacity is found, try to find the room with closest capacity
        if not suitable_rooms:
            activity_type = course_row.get('Activity Type Name', 'Unknown')
            print(
                f"Warning: Course {course_code} ({activity_type}, size: {course_size}) does not have a large enough room, trying the closest room")
            # Sort by capacity difference to find the closest room (prefer rooms slightly larger than course size)
            sorted_rooms = sorted(rooms_df.to_dict('records'),
                                  key=lambda x: (0 if x['CAP'] >= course_size else 1, abs(x['CAP'] - course_size)))
            if sorted_rooms:
                closest_room_idx = room_to_index[sorted_rooms[0]['ROOM NAME']]
                suitable_rooms = [closest_room_idx]
                print(
                    f"  Selected closest room with capacity {sorted_rooms[0]['CAP']} for course {course_code} (size: {course_size})")
            else:
                print(f"  Error: No room found for course {course_code}")
                skipped_courses['room_capacity'].append(course_code)
                continue

        total_week_count += len(teaching_weeks)
        week_success = False

        # Record a fixed time slot for this course (to ensure it is scheduled at the same time each week)
        fixed_day = random.randint(1, 5)

        # Choose a fixed time slot for this course
        valid_fixed_slots = list(range(1, 19 - duration_slots + 1))

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # Try multiple times to find a suitable time and room
            max_attempts = 500  # Increased attempt count
            success = False

            # Use fixed day and time slot
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            for attempt in range(max_attempts):
                # Check if all time slots occupied by the course are not blocked
                all_slots_available = True
                available_rooms = list(suitable_rooms)  # Make a copy of available rooms list

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in blocked_slots:
                        # Remove rooms already occupied in this time slot
                        available_rooms = [r for r in available_rooms if r not in blocked_slots[key]]
                        if not available_rooms:
                            all_slots_available = False
                            break

                # If all time slots have an available room, schedule the course
                if all_slots_available and available_rooms:
                    # Prefer the most suitable room
                    chosen_room = available_rooms[0]

                    # Schedule the course in continuous time slots, saving the starting slot and duration
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # Update blocked_slots, mark all time slots occupied by the course as blocked
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in blocked_slots:
                            blocked_slots[key] = set()
                        blocked_slots[key].add(chosen_room)

                    success = True
                    successful_week_count += 1
                    break

                # If the current day and time slot are unavailable, try another time slot
                if attempt > max_attempts // 2:
                    # Try a different time slot
                    day = random.randint(1, 5)
                    valid_start_slots = list(range(1, 19 - duration_slots + 1))

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # If after many attempts no suitable time and room is found, force schedule
            if not success:
                # New: Force schedule regular course
                print(
                    f"Warning: Unable to find a suitable time and room for course {course_code} in week {week}, forcing schedule")

                # Choose a random day and time
                day = random.randint(1, 5)
                time_slot = random.randint(1, 15)  # Earlier time slot
                weekday_idx = 5 * (week - 1) + day

                # Choose the available room with the largest capacity
                largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
                if largest_rooms:
                    chosen_room = room_to_index[largest_rooms[0]['ROOM NAME']]

                    # Force schedule
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # Update blocked_slots
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in blocked_slots:
                            blocked_slots[key] = set()
                        blocked_slots[key].add(chosen_room)

                    success = True
                    successful_week_count += 1
                    print(
                        f"  Forced schedule for course {course_code} in week {week}, day {day}, time slot {time_slot}")
                else:
                    print(f"  Severe error: No room found for course {course_code}")
                    if course_code not in skipped_courses['time_conflict']:
                        skipped_courses['time_conflict'].append(course_code)

            if success:
                week_success = True
            else:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(
                    f"Warning: Unable to find a suitable time and room for course {course_code} ({activity_type}) in week {week}")
                if course_code not in skipped_courses['time_conflict']:
                    skipped_courses['time_conflict'].append(course_code)

        if week_success:
            course_success_count += 1
        else:
            # If none of the teaching weeks could be scheduled, add to other reasons
            if course_code not in skipped_courses['room_capacity'] and course_code not in skipped_courses[
                'time_conflict']:
                skipped_courses['other'].append(course_code)

        print(f"\nInitial solution generation complete:")
        print(f"Number of courses successfully scheduled: {course_success_count}/{len(regular_course_indices)}")
        print(f"Number of teaching weeks successfully scheduled: {successful_week_count}/{total_week_count}")
        print(f"Number of assignments in the initial solution: {len(solution)}")

        # Output statistics of skipped courses
        print("\nSkipped courses statistics:")
        print(f"Courses skipped due to insufficient room capacity: {len(skipped_courses['room_capacity'])}")
        print(f"Courses skipped due to time conflicts: {len(skipped_courses['time_conflict'])}")
        print(f"Courses skipped due to other reasons: {len(skipped_courses['other'])}")

        # If the number is small, list the skipped courses
        for reason, courses in skipped_courses.items():
            if courses and len(courses) <= 10:
                print(f"\nCourses skipped due to {reason}:")
                for course in courses:
                    print(f"  - {course}")

        return solution


def generate_neighbor_with_activities(solution, regular_course_indices, index_to_course_row,
                                      rooms_df, room_to_index, index_to_course,
                                      blocked_slots, course_students):
    """
    Generate a neighboring solution, supporting different activity types and merged workshops:
    Prefer to adjust courses with conflicts by modifying their time and/or room,
    ensuring the newly chosen room is not occupied by a large course.
    Consider the course duration.

    Fixes:
    1. Ensure the same course is scheduled on the same day and time slot across different weeks.
    2. Add robust error handling to avoid KeyError.
    """
    # Create a deep copy of the current solution to ensure modifications do not affect the original
    new_solution = copy.deepcopy(solution)

    if not solution:
        return new_solution

    # Filter keys of regular courses, ensuring these keys exist in the current solution
    regular_keys = [key for key in solution.keys() if key[0] in regular_course_indices]

    if not regular_keys:
        return new_solution

    # Identify keys of courses with conflicts
    try:
        conflict_keys, _ = identify_conflict_courses_with_activities(solution, course_students,
                                                                     index_to_course, index_to_course_row)
        # Filter conflict keys that belong only to regular courses
        valid_conflict_keys = [key for key in conflict_keys if key in solution and key[0] in regular_course_indices]
    except Exception as e:
        print(f"Error identifying conflict courses: {str(e)}")
        valid_conflict_keys = []

    # Prefer to select a course with conflict for adjustment
    try:
        if valid_conflict_keys and random.random() < 0.95:  # 95% probability to choose a conflict course
            key = random.choice(valid_conflict_keys)
        else:
            # Randomly choose any regular course
            key = random.choice(regular_keys)
    except Exception as e:
        print(f"Error selecting course: {str(e)}")
        return new_solution  # Return the original solution on error

    # Verify that the key exists in the solution
    if key not in new_solution:
        print(f"Warning: Key {key} does not exist in the solution. Skipping this neighbor generation.")
        return solution  # Return the original solution

    try:
        course_idx, weekday_idx, time_slot = key
        room_idx, duration_slots = new_solution[key]

        # Calculate week and day
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # Get course information
        course_row = index_to_course_row.get(course_idx)
        if course_row is None:
            print(f"Warning: Could not find information for course index {course_idx}, skipping neighbor generation")
            return new_solution

        course_size = course_row['Real Size']
    except Exception as e:
        print(f"Error retrieving course information: {str(e)}")
        return new_solution

    # First, find all schedules for the same course in different weeks
    same_course_keys = []
    for k in list(new_solution.keys()):  # Use list() to avoid modifying while iterating
        if k[0] == course_idx and k != key:
            same_course_keys.append(k)

    # Determine operation type: 1 = change time, 2 = change room, 3 = change both time and room
    operation = random.randint(1, 3)

    # Create a copy of the current blocked_slots
    try:
        current_blocked_slots = copy.deepcopy(blocked_slots)

        # First, remove the current course's occupied time slots from blocked_slots
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            block_key = (weekday_idx, curr_slot)
            if block_key in current_blocked_slots and room_idx in current_blocked_slots[block_key]:
                current_blocked_slots[block_key].remove(room_idx)

        # Also remove the time slots for the same course in other weeks
        for course_key in same_course_keys:
            if course_key not in new_solution:
                continue  # Skip keys that no longer exist
            c_idx, w_idx, t_slot = course_key
            r_idx, d_slots = new_solution[course_key]
            for slot_offset in range(d_slots):
                curr_slot = t_slot + slot_offset
                block_key = (w_idx, curr_slot)
                if block_key in current_blocked_slots and r_idx in current_blocked_slots[block_key]:
                    current_blocked_slots[block_key].remove(r_idx)
    except Exception as e:
        print(f"Error processing blocked time slots: {str(e)}")
        return new_solution

    if operation in (1, 3):  # Change time or change both time and room
        try:
            # Generate new time, ensuring it does not span the lunch break
            new_day = random.randint(1, 5)

            # Find all valid starting time slots
            valid_start_slots = [potential_start for potential_start in range(1, 19 - duration_slots + 1)]

            if not valid_start_slots:
                # No valid starting time, return original solution
                return new_solution

            new_time_slot = random.choice(valid_start_slots)

            # Get all teaching weeks for the course
            teaching_weeks = course_row.get('Teaching_Weeks', [])
            if not teaching_weeks:
                teaching_weeks = list(range(1, 13))  # Default weeks 1-12

            # Check if the new time is feasible for all teaching weeks
            all_weeks_available = True
            for check_week in teaching_weeks:
                check_weekday_idx = 5 * (check_week - 1) + new_day
                for slot_offset in range(duration_slots):
                    curr_slot = new_time_slot + slot_offset
                    block_key = (check_weekday_idx, curr_slot)
                    if block_key in current_blocked_slots and room_idx in current_blocked_slots[block_key]:
                        all_weeks_available = False
                        break
                if not all_weeks_available:
                    break

            if all_weeks_available:
                # Safely delete all schedules for the original course
                if key in new_solution:
                    del new_solution[key]

                for k in same_course_keys:
                    if k in new_solution:  # Ensure key exists
                        del new_solution[k]

                # Reschedule the course for all teaching weeks
                for course_week in teaching_weeks:
                    new_weekday_idx = 5 * (course_week - 1) + new_day
                    # For operation 1, keep the original room; for operation 3, handle later
                    if operation == 1:
                        new_solution[(course_idx, new_weekday_idx, new_time_slot)] = (room_idx, duration_slots)

                        # Update blocked_slots
                        for slot_offset in range(duration_slots):
                            curr_slot = new_time_slot + slot_offset
                            block_key = (new_weekday_idx, curr_slot)
                            if block_key not in blocked_slots:
                                blocked_slots[block_key] = set()
                            blocked_slots[block_key].add(room_idx)

                if operation == 1:
                    return new_solution
            else:
                # If new time is not available, return original solution
                if operation == 1:
                    return new_solution
        except Exception as e:
            print(f"Error during time change operation: {str(e)}")
            return new_solution

    if operation in (2, 3):  # Change room or change both time and room
        try:
            # Filter rooms that meet the criteria: sufficient capacity and not in blocked_slots
            suitable_rooms = []

            # For operation 2, check the original time; for operation 3 (if new time is available), check new time
            use_new_time = operation == 3 and all_weeks_available
            check_day = new_day if use_new_time else day
            check_time_slot = new_time_slot if use_new_time else time_slot

            # Get all teaching weeks for the course
            teaching_weeks = course_row.get('Teaching_Weeks', [])
            if not teaching_weeks:
                teaching_weeks = list(range(1, 13))  # Default weeks 1-12

            for _, room in rooms_df.iterrows():
                r_idx = room_to_index[room['ROOM NAME']]
                if room['CAP'] >= course_size and r_idx != room_idx:  # Exclude current room
                    # Check if all time slots in all teaching weeks are available
                    all_rooms_available = True
                    for check_week in teaching_weeks:
                        check_weekday_idx = 5 * (check_week - 1) + check_day
                        for slot_offset in range(duration_slots):
                            curr_slot = check_time_slot + slot_offset
                            block_key = (check_weekday_idx, curr_slot)
                            if block_key in current_blocked_slots and r_idx in current_blocked_slots[block_key]:
                                all_rooms_available = False
                                break
                        if not all_rooms_available:
                            break

                    if all_rooms_available:
                        suitable_rooms.append(r_idx)

            if suitable_rooms:
                new_room_idx = random.choice(suitable_rooms)

                # Safely delete all schedules for the original course
                if key in new_solution:
                    del new_solution[key]

                for k in same_course_keys:
                    if k in new_solution:  # Ensure key exists
                        del new_solution[k]

                # Reschedule the course for all teaching weeks
                for course_week in teaching_weeks:
                    if operation == 2:
                        # Only change room, keep original time
                        new_weekday_idx = 5 * (course_week - 1) + day
                        new_solution[(course_idx, new_weekday_idx, time_slot)] = (new_room_idx, duration_slots)

                        # Update blocked_slots
                        for slot_offset in range(duration_slots):
                            curr_slot = time_slot + slot_offset
                            block_key = (new_weekday_idx, curr_slot)
                            if block_key not in blocked_slots:
                                blocked_slots[block_key] = set()
                            blocked_slots[block_key].add(new_room_idx)
                    else:
                        # Change both time and room
                        new_weekday_idx = 5 * (course_week - 1) + new_day
                        new_solution[(course_idx, new_weekday_idx, new_time_slot)] = (new_room_idx, duration_slots)

                        # Update blocked_slots
                        for slot_offset in range(duration_slots):
                            curr_slot = new_time_slot + slot_offset
                            block_key = (new_weekday_idx, curr_slot)
                            if block_key not in blocked_slots:
                                blocked_slots[block_key] = set()
                            blocked_slots[block_key].add(new_room_idx)
        except Exception as e:
            print(f"Error during room change operation: {str(e)}")
            return new_solution

    # Additional check: Ensure the same course has the same day and time slot in different weeks
    try:
        # Verify and fix consistency issues, although the previous code should have ensured consistency
        new_solution = fix_schedule_consistency(new_solution)
    except Exception as e:
        print(f"Error fixing schedule consistency: {str(e)}")
        return solution  # Return original solution on error

    new_solution = fix_schedule_consistency(new_solution)
    return new_solution


# Retain the original fix function as backup
def fix_schedule_consistency(sol):
    """
    Fix schedule consistency issues, ensuring the same course has the same class day and time across different weeks.
    """
    fixed_sol = {}
    course_reference = {}  # Store each course's reference class time (day, time_slot)

    for key, value in sol.items():
        try:
            course_idx, weekday_idx, time_slot = key
            week, day = convert_weekday_to_week_and_day(weekday_idx)

            if course_idx not in course_reference:
                course_reference[course_idx] = (day, time_slot)

            ref_day, ref_time_slot = course_reference[course_idx]
            # Keep week unchanged; update weekday_idx to 5*(week-1)+ref_day
            new_weekday_idx = 5 * (week - 1) + ref_day
            fixed_sol[(course_idx, new_weekday_idx, ref_time_slot)] = value
        except Exception as e:
            print(f"Error processing key {key} in fix_schedule_consistency: {str(e)}")
            # In case of error, keep the original key-value pair
            fixed_sol[key] = value

    return fixed_sol


def identify_conflict_courses_with_activities(solution, course_students, index_to_course, index_to_course_row):
    """
    Identify conflicting courses in the current solution, supporting different activity types and merged workshops.
    Considers course duration and strictly defines a conflict as courses in the same semester, same teaching week, and same weekday with overlapping time slots.
    Correction: Accounts for partial time overlaps due to course duration.

    Returns: A set of conflicting course keys (course_idx, weekday_idx, time_slot).
    """
    conflict_keys = set()  # Now store complete keys

    # Get the semester information for a course
    def get_normalized_semester(course_idx):
        course_row = index_to_course_row.get(course_idx)
        if not course_row:
            return "Unknown"

        semester_str = course_row.get('Delivery Semester', 'Unknown')
        if not isinstance(semester_str, str):
            return "Unknown"

        # Convert to lowercase for a standardized format
        semester_str = semester_str.lower()

        # Extract semester information (e.g., semester1 or semester2)
        if 'semester1' in semester_str or 'semester 1' in semester_str:
            return "Semester1"
        elif 'semester2' in semester_str or 'semester 2' in semester_str:
            return "Semester2"
        else:
            return semester_str  # Keep as is

    # Store a set of conflicting course indices
    conflict_course_indices = set()

    # Group courses by semester, week, and day
    semester_week_day_courses = {}

    # Collect all time slots occupied by each course
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # Extract week and day information
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # Get semester information
        semester = get_normalized_semester(course_idx)

        # Use a tuple (semester, week, day) as the key
        key = (semester, week, day)

        if key not in semester_week_day_courses:
            semester_week_day_courses[key] = []

        # Store course information including duration
        semester_week_day_courses[key].append({
            'course_key': (course_idx, weekday_idx, time_slot),
            'course_idx': course_idx,
            'start_slot': time_slot,
            'end_slot': time_slot + duration_slots - 1,  # Last occupied time slot
            'duration': duration_slots
        })

    # For each group, check for overlapping course times
    for key, courses in semester_week_day_courses.items():
        semester, week, day = key

        # Compare each pair of courses for overlapping times
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

                # Check if the times overlap: one course's start <= the other course's end and its end >= the other course's start
                if start_i <= end_j and end_i >= start_j:
                    # Extract course codes
                    course_code_i = index_to_course.get(course_idx_i)
                    course_code_j = index_to_course.get(course_idx_j)

                    if not course_code_i or not course_code_j:
                        continue

                    # Extract base course codes (removing possible activity type suffix)
                    base_course_i = course_code_i.split('_')[0] if '_' in course_code_i else course_code_i
                    base_course_j = course_code_j.split('_')[0] if '_' in course_code_j else course_code_j

                    students_i = course_students.get(base_course_i, set())
                    students_j = course_students.get(base_course_j, set())

                    # If there are common enrolled students, mark these courses as conflicting
                    if students_i.intersection(students_j):
                        conflict_keys.add(course_key_i)
                        conflict_keys.add(course_key_j)
                        conflict_course_indices.add(course_idx_i)
                        conflict_course_indices.add(course_idx_j)

    # Additional check: Ensure all returned keys are present in the current solution
    valid_conflict_keys = {key for key in conflict_keys if key in solution}

    # Optionally, return the set of conflicting course indices for compatibility
    return valid_conflict_keys, conflict_course_indices


def convert_solution_to_schedule_with_activities(solution, all_courses_df, rooms_df,
                                                 index_to_course, index_to_room, index_to_course_row):
    """
    Convert the solution to a timetable format, supporting different activity types and merged workshops.
    For merged workshops, they will be expanded into multiple records, each using the same time and room.
    Considers course duration and ensures the correct end time.
    Modification: The output 'Course Code' only contains the base course code, without the activity type.
    """
    schedule = []

    # Store merged workshop information for later expansion
    merged_workshops = {}

    # First process all non-merged courses
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in solution.items():
        # Get course information
        course_row = index_to_course_row.get(course_idx)
        if not course_row:
            continue

        # If it is a merged workshop, store it for later expansion
        if course_row.get('Is_Merged', False):
            merged_workshops[(course_idx, weekday_idx, time_slot, room_idx, duration_slots)] = course_row
            continue

        # Process non-merged courses
        full_code = index_to_course[course_idx]
        # Extract only the base course code without the activity type
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
        activity_type = course_row.get('Activity Type Name', 'Unknown')
        room_name = index_to_room[room_idx]

        # Find the corresponding room information
        room_rows = rooms_df[rooms_df['ROOM NAME'] == room_name]
        if len(room_rows) == 0:
            continue
        room_row = room_rows.iloc[0]

        # Compute week and day
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # Calculate the end time - adjust end time calculation to be consistent with course duration
        end_time_slot = time_slot + duration_slots
        end_time = convert_slot_to_time(end_time_slot)

        # Add course assignment
        schedule.append({
            'Course Code': course_code,  # Contains only the base course code
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
            'Duration': course_row.get('Duration', '1:00'),  # Save original duration
            'Class Size': course_row['Real Size'],
            'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
            'Is Merged Workshop': False
        })

    # Process merged workshops and expand into multiple records
    for (course_idx, weekday_idx, time_slot, room_idx, duration_slots), course_row in merged_workshops.items():
        full_code = index_to_course[course_idx]
        # Extract only the base course code
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
        room_name = index_to_room[room_idx]

        # Find the corresponding room information
        room_rows = rooms_df[rooms_df['ROOM NAME'] == room_name]
        if len(room_rows) == 0:
            continue
        room_row = room_rows.iloc[0]

        # Compute week and day
        week, day = convert_weekday_to_week_and_day(weekday_idx)

        # Get activity type and merged details
        activity_type = course_row.get('Activity Type Name', 'Unknown')
        merged_details = course_row.get('Merged_Details', [])

        # Calculate the end time - adjust end time calculation
        end_time_slot = time_slot + duration_slots
        end_time = convert_slot_to_time(end_time_slot)

        # Type check to ensure merged_details is iterable
        if isinstance(merged_details, list) and merged_details:
            # Create a record for each merged workshop
            for detail in merged_details:
                workshop_id = detail.get('ID', 'Unknown')
                workshop_size = detail.get('Real Size', 0)

                schedule.append({
                    'Course Code': course_code,  # Contains only the base course code
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
                    'Duration': course_row.get('Duration', '1:00'),  # Save original duration
                    'Class Size': workshop_size,  # Use the original workshop size
                    'Total Merged Size': course_row['Real Size'],  # Total merged size
                    'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
                    'Is Merged Workshop': True,
                    'Workshop ID': workshop_id
                })
        else:
            # If no detailed information, create a single record
            print(f"Warning: merged_details is not a list or is empty, type is {type(merged_details)}")
            schedule.append({
                'Course Code': course_code,  # Contains only the base course code
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
                'Duration': course_row.get('Duration', '1:00'),  # Save original duration
                'Class Size': course_row['Real Size'],
                'Is Large Course': course_row.get('Real Size', 0) > course_row.get('Planned Size', 0),
                'Is Merged Workshop': True,
                'Merged Count': course_row.get('Merged_Count', 0)
            })

    # Create DataFrame from schedule list
    schedule_df = pd.DataFrame(schedule)

    # Add room utilization column
    if not schedule_df.empty:
        schedule_df['Utilization'] = (schedule_df['Class Size'] / schedule_df['Room Capacity'] * 100).round(2)

        # Calculate overall room utilization
    avg_utilization = schedule_df['Utilization'].mean() if not schedule_df.empty else 0
    print(f"Timetable average room utilization: {avg_utilization:.2f}%")

    # Count the number of rooms used
    room_usage = schedule_df['Room'].value_counts() if not schedule_df.empty else pd.Series()
    print(f"Number of rooms used: {len(room_usage)}")

    # Return the final schedule DataFrame
    return schedule_df


def compute_student_conflict_with_activities(schedule_df, course_students):
    """
    Compute the total number of student course conflicts and the conflict rate in the timetable.
    New logic:
      1. For each student, check how many different courses are selected in each time slot;
      2. Different workshops of the same course are counted only once (each course is counted only once);
      3. If more than one distinct course is in a time slot, the conflict count is (number of different courses - 1).

    Conflict rate = (total conflict time slots) / (total number of course selections across all time slots for all students)
    """
    # ------------------ Count Total Course Selections and Total Students ------------------
    total_students = set()
    total_course_selections = 0  # Sum of enrollment numbers for all courses
    for course_code, students in course_students.items():
        total_students.update(students)
        total_course_selections += len(students)

    # ------------------ Preprocess Semester Information ------------------
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

    schedule_df['Normalized_Semester'] = schedule_df['Delivery Semester'].apply(normalize_semester)

    # ------------------ Build Student-Time Slot Mapping ------------------
    # Key format: (student_id, semester, teaching week, day, time_slot)
    student_time_slot_courses = {}

    # Iterate over the timetable; each record may occupy multiple consecutive time slots
    for _, row in schedule_df.iterrows():
        # Extract the base course code (without activity type suffix)
        full_code = row['Course Code']
        course_code = full_code.split('_')[0] if '_' in full_code else full_code
        activity_type = row['Activity Type'].lower()
        # Get the set of students enrolled in this course from course_students
        students = course_students.get(course_code, set())
        if not students:
            continue

        semester = row['Normalized_Semester']
        week = row['Week']
        day = row['Day']
        start_slot = row['Start Time Slot']
        duration_slots = row['Duration Slots']

        # For all time slots occupied by this record, record the course information
        for slot_offset in range(duration_slots):
            curr_slot = start_slot + slot_offset
            # For each student enrolled, record the course info for the corresponding time slot
            for student_id in students:
                key = (student_id, semester, week, day, curr_slot)
                if key not in student_time_slot_courses:
                    student_time_slot_courses[key] = []
                student_time_slot_courses[key].append((course_code, activity_type))

    # ------------------ Count Conflict Cases ------------------
    total_conflicts = 0       # Total number of conflicts (at time slot level)
    conflict_students = set() # Set of students with conflicts
    conflict_details = []     # Record conflict details for later analysis
    student_conflicts = {}    # Cumulative conflict count for each student

    for (student_id, semester, week, day, time_slot), course_info_list in student_time_slot_courses.items():
        # For the same time slot, count different workshops of the same course only once
        unique_courses = set()
        workshop_courses = set()
        regular_courses = set()

        for course_code, activity_type in course_info_list:
            if 'workshop' in activity_type:
                workshop_courses.add(course_code)
            else:
                regular_courses.add(course_code)
                unique_courses.add(course_code)
        # Merge workshop courses into the unique courses set (each course is counted only once)
        unique_courses.update(workshop_courses)

        # If there is more than one distinct course, count the conflict
        if len(unique_courses) > 1:
            conflicts_count = len(unique_courses) - 1  # Count conflicts as number of courses beyond one
            total_conflicts += conflicts_count
            conflict_students.add(student_id)
            student_conflicts[student_id] = student_conflicts.get(student_id, 0) + conflicts_count

            # Convert time slot to a time display (using the external function convert_slot_to_time)
            start_time = convert_slot_to_time(time_slot)
            end_time = convert_slot_to_time(time_slot + 1)
            # Form a conflict course info string (display regular and workshop courses separately)
            conflict_course_str = []
            for course in regular_courses:
                conflict_course_str.append(f"{course}(Regular)")
            if workshop_courses:
                conflict_course_str.append(f"{', '.join(workshop_courses)}(Workshop)")
            conflict_details.append({
                "Student ID": student_id,
                "Semester": semester,
                "Teaching Week": week,
                "Day": day,
                "Time Slot": time_slot,
                "Time": f"{start_time}-{end_time}",
                "Conflicting Courses": ", ".join(conflict_course_str),
                "Conflict Count": conflicts_count
            })

    # ------------------ Calculate Conflict Rate ------------------
    conflict_rate = 0
    if total_course_selections > 0:
        conflict_rate = (total_conflicts / total_course_selections) * 100

    # ------------------ Output Conflict Details and Statistics ------------------
    if conflict_details:
        print("\nDetailed Conflict Information:")
        sorted_details = sorted(conflict_details, key=lambda x: x['Conflict Count'], reverse=True)
        for i, detail in enumerate(sorted_details[:10], 1):  # Only show the first 10 entries
            print(f"{i}. Student: {detail['Student ID']}, Semester: {detail['Semester']}, "
                  f"Teaching Week: {detail['Teaching Week']}, Day: {detail['Day']}, Time: {detail['Time']}, "
                  f"Conflicting Courses: {detail['Conflicting Courses']}, Conflict Count: {detail['Conflict Count']}")
        if len(sorted_details) > 10:
            print(f"... A total of {len(sorted_details)} conflict records found")

    if student_conflicts:
        top_conflict_students = sorted(student_conflicts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nStudents with the Most Conflicts:")
        for student_id, count in top_conflict_students:
            print(f"  Student {student_id}: {count} conflicts")

    print("\n---------- Student Course Conflict Statistics ----------")
    print(f"Total number of enrolled students: {len(total_students)}")
    print(f"Number of students with course conflicts: {len(conflict_students)}")

    return total_conflicts, conflict_rate, len(conflict_students), len(total_students), conflict_details


def simulated_annealing_scheduling(
        enrollment_file='math_student_enrollment.xlsx',
        courses_file='df_final_cleaned_1.xlsx',
        rooms_file='Timetabling_KB_Rooms.xlsx',
        max_iterations=100000,
        initial_temperature=5000,
        cooling_rate=0.997,
        utilization_weight=2.0,
        conflict_weight=40.0,
        use_room_optimization=True
):
    """
    Solve the course scheduling problem using simulated annealing, supporting different activity types and merged workshops:
      1. Large courses are defined as: Real Size > Planned Size; these courses are forced to be scheduled in their designated room;
      2. Handle different activity types (lecture, workshop, etc.); different activities with the same course code are scheduled separately;
      3. Merge workshops that meet the criteria (same course code, teaching week pattern, and activity type), with a merged total not exceeding 120;
      4. In the output, merged workshops will be expanded so that all merged workshops use the same time and room;
      5. Ensure that course duration remains consistent with the original data and is not split into multiple smaller time segments.
    """
    try:
        courses_df = pd.read_excel(courses_file)
        rooms_df = pd.read_excel(rooms_file)
        enrollment_df = pd.read_excel(enrollment_file)

        # Add a row index to the course data for identification
        courses_df['Row_ID'] = courses_df.index

        # Display the number of courses with Real Size 0
        zero_size_courses = courses_df[courses_df['Real Size'] == 0]
        if not zero_size_courses.empty:
            print(f"There are {len(zero_size_courses)} courses with Real Size 0 that will be automatically ignored")

        # Create a standardized mapping for room names
        rooms_df['Normalized_Name'] = rooms_df['ROOM NAME'].apply(normalize_room_name)

        # Create a mapping from the original name to the standardized name
        room_name_map = {}
        for _, row in rooms_df.iterrows():
            room_name_map[normalize_room_name(row['ROOM NAME'])] = row['ROOM NAME']

        # Create standardized room names in the course data
        courses_df['Normalized_Room_Name'] = courses_df['ROOM NAME'].apply(normalize_room_name)

        # Display statistics on course durations
        print("\n---------- Course Duration Statistics ----------")
        duration_counts = courses_df['Duration'].value_counts()
        for duration, count in duration_counts.items():
            slots = parse_duration(duration)
            print(f"Duration {duration} ({slots} time slots): {count} courses")

        # Preprocess course data, including handling different activity types and merging workshops
        processed_courses_df = preprocess_course_with_activities(courses_df)

        # Check for courses with empty teaching weeks
        empty_weeks_courses = sum(1 for _, row in processed_courses_df.iterrows() if not row['Teaching_Weeks'])
        if empty_weeks_courses > 0:
            print(f"Warning: Found {empty_weeks_courses} courses with no valid teaching weeks")

        # New: Divide courses into large courses and regular courses based on conditions
        # Large courses are defined as: Real Size > Planned Size
        large_course_df = processed_courses_df[processed_courses_df['Real Size'] > processed_courses_df['Planned Size']]
        regular_courses_df = processed_courses_df[
            processed_courses_df['Real Size'] <= processed_courses_df['Planned Size']]

        all_courses_df = processed_courses_df.copy()  # Save all course data for final energy calculation

        print("\n---------- Course and Room Statistics ----------")
        print("Total courses after processing: {}".format(len(processed_courses_df)))
        print("Number of large courses: {}".format(len(large_course_df)))
        print("Number of regular courses: {}".format(len(regular_courses_df)))
        print("Number of rooms: {}".format(len(rooms_df)))
        print("Student enrollment records: {}".format(len(enrollment_df)))

        # Print room name matching status
        distinct_room_names = processed_courses_df['Normalized_Room_Name'].unique()
        matched_rooms = 0
        for room_name in distinct_room_names:
            if room_name in room_name_map:
                matched_rooms += 1

        print("Designated room match rate for large courses: {}/{} ({:.2f}%)".format(
            matched_rooms, len(distinct_room_names),
            (matched_rooms / len(distinct_room_names)) * 100 if len(distinct_room_names) > 0 else 0
        ))

        # Print room capacity distribution
        print("\nRoom Capacity Distribution:")
        capacity_bins = [0, 30, 60, 90, 120, 150, 200, 300, 500, 1000]
        capacity_counts = pd.cut(rooms_df['CAP'], bins=capacity_bins).value_counts().sort_index()
        for bin_range, count in capacity_counts.items():
            print(f"  {bin_range}: {count} rooms")

    except Exception as e:
        print("Data loading failed: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return None

    # Build course-to-index mapping using the course index as key and saving full information
    course_to_index = {}
    index_to_course = {}
    index_to_course_row = {}  # New: Mapping from index to course row
    regular_course_indices = set()
    large_course_indices = set()
    course_index_map = {}  # Additional mapping to store all index information

    # Create a unique index for each course row
    for i, row in all_courses_df.iterrows():
        course_code = row['Course Code']

        # If there is an activity type, append it to distinguish different activities
        if 'Activity Type Name' in row:
            activity_type = row['Activity Type Name']
            full_code = f"{course_code}_{activity_type}"  # Full code includes activity type
        else:
            full_code = course_code

        # Append row number to ensure complete uniqueness
        unique_id = f"{full_code}_{i}"

        # Course indices start from 1
        idx_val = len(course_to_index) + 1

        # Save the complete mapping
        course_to_index[unique_id] = idx_val
        index_to_course[idx_val] = full_code  # Save the full course code including activity type
        index_to_course_row[idx_val] = row.to_dict()  # Store the entire row as a dictionary
        course_index_map[idx_val] = {'code': course_code, 'full_code': full_code, 'row_idx': i}

        # Determine if it is a large course or a regular course
        if row['Real Size'] > row.get('Planned Size', 0):
            large_course_indices.add(idx_val)
        else:
            regular_course_indices.add(idx_val)

    room_to_index = {room: idx + 1 for idx, room in enumerate(rooms_df['ROOM NAME'])}
    index_to_room = {idx + 1: room for idx, room in enumerate(rooms_df['ROOM NAME'])}

    # Build a mapping from courses to enrolled students based on the enrollment file
    course_students = {}
    try:
        if 'course_id' in enrollment_df.columns and 'student_id' in enrollment_df.columns:
            # For math_student_enrollment.xlsx format
            for _, row in enrollment_df.iterrows():
                course_code = row['course_id']
                student_id = row['student_id']
                if course_code not in course_students:
                    course_students[course_code] = set()
                course_students[course_code].add(student_id)
        elif 'Course ID' in enrollment_df.columns and 'Student ID' in enrollment_df.columns:
            # For Anon Enrollment Data_new.xlsx format
            for _, row in enrollment_df.iterrows():
                course_code = row['Course ID']
                student_id = row['Student ID']
                if course_code not in course_students:
                    course_students[course_code] = set()
                course_students[course_code].add(student_id)
        else:
            print("Warning: Unable to recognize enrollment data format, student course conflict rate cannot be calculated")
    except Exception as e:
        print(f"Error constructing course-to-student mapping: {str(e)}")
        course_students = {}

    # --- Process Large Courses ---
    # For large courses, force schedule them in their designated room
    large_courses_solution = {}
    large_course_failures = 0  # Count failures
    large_course_success = 0  # Count successes

    for unique_id, idx_val in course_to_index.items():
        if idx_val not in large_course_indices:
            continue

        course_row = index_to_course_row[idx_val]
        course_code = course_row['Course Code']
        normalized_room_name = course_row['Normalized_Room_Name']
        course_size = course_row['Real Size']

        # Get course duration
        duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(duration)

        print(f"Large course {unique_id} duration: {duration} ({duration_slots} time slots)")

        # Determine the room to use
        designated_room_name = None

        # 1. Try matching using the standardized name
        if normalized_room_name in room_name_map:
            designated_room_name = room_name_map[normalized_room_name]

        # 2. Try direct matching
        elif course_row['ROOM NAME'] in room_to_index:
            designated_room_name = course_row['ROOM NAME']

        # 3. Look for an alternative room with sufficient capacity
        else:
            print(f"Warning: The designated room {course_row['ROOM NAME']} for large course {course_code} does not exist, trying to find an alternative room")

            alternative_rooms = []
            for _, room in rooms_df.iterrows():
                if room['CAP'] >= course_size:
                    alternative_rooms.append(room['ROOM NAME'])

            if alternative_rooms:
                designated_room_name = random.choice(alternative_rooms)
                print(f"  For large course {course_code}, selected alternative room: {designated_room_name}")

        # 4. If still not found, try using the room with the maximum capacity
        if not designated_room_name:
            largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
            if largest_rooms:
                designated_room_name = largest_rooms[0]['ROOM NAME']
                print(
                    f"  Warning: Large course {course_code} (size: {course_size}) will use the largest room with capacity {largest_rooms[0]['CAP']}")
            else:
                print(f"  Error: No available room found for large course {course_code}, skipping this course")
                large_course_failures += 1
                continue

        designated_room_idx = room_to_index[designated_room_name]

        # Schedule a time for each teaching week
        teaching_weeks = course_row['Teaching_Weeks']
        if not teaching_weeks:
            print(f"  Warning: Large course {course_code} has no valid teaching weeks, using default teaching weeks (1-12)")
            teaching_weeks = list(range(1, 13))

        week_success_count = 0

        for week in teaching_weeks:
            # Try multiple times to find a suitable time
            max_attempts = 500  # Increase attempt count
            success = False

            for attempt in range(max_attempts):
                day = random.randint(1, 5)

                # Get all valid starting time slots
                valid_start_slots = list(range(1, 19 - duration_slots + 1))

                time_slot = random.choice(valid_start_slots)
                weekday_idx = 5 * (week - 1) + day

                # Use tuple (room_idx, duration_slots) instead of a single integer
                large_courses_solution[(idx_val, weekday_idx, time_slot)] = (designated_room_idx, duration_slots)
                week_success_count += 1
                success = True
                break

            if not success:
                print(f"  Warning: Unable to find a suitable time for large course {course_code} in week {week} after {max_attempts} attempts")

        if week_success_count > 0:
            large_course_success += 1
            print(f"  Large course {course_code} successfully scheduled: {week_success_count}/{len(teaching_weeks)} teaching weeks")

    print(f"\nLarge courses scheduling completed: {large_course_success}/{len(large_course_indices)} courses successfully scheduled")
    if large_course_failures > 0:
        print(f"{large_course_failures} large courses were skipped due to not finding a suitable room")

    # Construct blocked_slots: record the rooms occupied by large courses for each (weekday_idx, time_slot)
    blocked_slots = {}
    for (course_idx, weekday_idx, time_slot), (room_idx, duration_slots) in large_courses_solution.items():
        # Mark every time slot occupied by the course
        for slot_offset in range(duration_slots):
            curr_slot = time_slot + slot_offset
            key = (weekday_idx, curr_slot)
            if key not in blocked_slots:
                blocked_slots[key] = set()
            blocked_slots[key].add(room_idx)

    # --- Process Regular Courses ---
    print("\nGenerating initial solution for regular courses...")

    # Improved initial solution generation function (defined below)
    def generate_initial_solution_improved(regular_course_indices, index_to_course_row,
                                           rooms_df, room_to_index, blocked_slots):
        """Improved initial solution generation function, handling constraints more flexibly"""
        solution = {}
        course_success_count = 0  # Number of courses successfully scheduled
        total_week_count = 0  # Total number of teaching weeks
        successful_week_count = 0  # Number of teaching weeks successfully scheduled

        for course_idx in regular_course_indices:
            course_row = index_to_course_row[course_idx]
            course_size = course_row['Real Size']
            course_code = course_row.get('Course Code', f"Unknown_{course_idx}")
            teaching_weeks = course_row['Teaching_Weeks']

            # Check if teaching weeks is empty
            if not teaching_weeks:
                print(f"Warning: Course {course_code} has no valid teaching weeks, using default teaching weeks (1-12)")
                teaching_weeks = list(range(1, 13))

            # Get course duration
            original_duration = course_row.get('Duration', '1:00')
            duration_slots = parse_duration(original_duration)

            # Track room usage frequency
            room_usage_count = {}
            for key, (room_idx, _) in solution.items():
                room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

            # Filter rooms with sufficient capacity and sort by usage frequency
            suitable_rooms = []
            for _, room in rooms_df.iterrows():
                room_idx = room_to_index[room['ROOM NAME']]
                if room['CAP'] >= course_size:
                    suitable_rooms.append((room_idx, room_usage_count.get(room_idx, 0)))

            # Sort by usage frequency, preferring rooms used less frequently
            suitable_rooms.sort(key=lambda x: x[1])
            suitable_rooms = [room_idx for room_idx, _ in suitable_rooms]

            # If no room with sufficient capacity is found, use the room with the maximum capacity
            if not suitable_rooms:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(f"Warning: Course {course_code} ({activity_type}, size: {course_size}) does not have a large enough room, trying the largest room")

                # Sort to find the largest room by capacity
                largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
                if largest_rooms:
                    largest_room_idx = room_to_index[largest_rooms[0]['ROOM NAME']]
                    suitable_rooms = [largest_room_idx]
                    print(f"  Selected largest room with capacity {largest_rooms[0]['CAP']} for course {course_code} (size: {course_size})")
                else:
                    print(f"  Error: No room found for course {course_code}")
                    continue

            total_week_count += len(teaching_weeks)
            week_success = False

            for week in teaching_weeks:
                # Try multiple times to find a suitable time and room
                max_attempts = 500  # Increase attempt count
                success = False

                for attempt in range(max_attempts):
                    day = random.randint(1, 5)

                    # Ensure the course does not exceed the day's time range or cross the lunch break
                    valid_start_slots = list(range(1, 19 - duration_slots + 1))
                    time_slot = random.choice(valid_start_slots)
                    weekday_idx = 5 * (week - 1) + day

                    # Check if all time slots occupied by the course are not blocked
                    all_slots_available = True
                    available_rooms = list(suitable_rooms)  # Copy the list of available rooms

                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key in blocked_slots:
                            # Remove rooms that are blocked in this time slot
                            available_rooms = [r for r in available_rooms if r not in blocked_slots[key]]
                            if not available_rooms:
                                all_slots_available = False
                                break

                    # If all time slots have an available room, schedule the course
                    if all_slots_available and available_rooms:
                        chosen_room = random.choice(available_rooms)

                        # Schedule the course in consecutive time slots; save the starting slot and duration
                        solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                        # Update blocked_slots to mark all time slots occupied by the course as blocked
                        for slot_offset in range(duration_slots):
                            curr_slot = time_slot + slot_offset
                            key = (weekday_idx, curr_slot)
                            if key not in blocked_slots:
                                blocked_slots[key] = set()
                            blocked_slots[key].add(chosen_room)

                        success = True
                        successful_week_count += 1
                        break

                # If after many attempts no suitable time and room are found, record a warning
                if success:
                    week_success = True
                else:
                    activity_type = course_row.get('Activity Type Name', 'Unknown')
                    print(f"Warning: Unable to find a suitable time and room for course {course_code} ({activity_type}) in week {week}")

            if week_success:
                course_success_count += 1

        print(f"\nInitial solution generation complete:")
        print(f"Number of courses successfully scheduled: {course_success_count}/{len(regular_course_indices)}")
        print(f"Number of teaching weeks successfully scheduled: {successful_week_count}/{total_week_count}")
        print(f"Number of assignments in the initial solution: {len(solution)}")

        return solution

    # Use the improved initial solution generation function
    current_solution = generate_initial_solution_improved(regular_course_indices, index_to_course_row,
                                                          rooms_df, room_to_index, blocked_slots)
    if current_solution is None:
        print("Unable to generate an initial solution, please check constraint conditions")
        return None

    # Merge the large course solution and the regular course solution
    current_solution.update(large_courses_solution)

    # Create index mappings and lookup tables
    room_name_to_row = {}
    for _, row in rooms_df.iterrows():
        room_name = row['ROOM NAME']
        room_name_to_row[room_name] = row

    # Calculate the energy of the initial solution
    current_energy = calculate_energy(current_solution, index_to_course_row, room_name_to_row,
                                      course_to_index, room_to_index, index_to_course, index_to_room,
                                      course_students, utilization_weight, conflict_weight)

    best_solution = current_solution.copy()
    best_energy = current_energy

    print("Initial solution energy value: {}".format(current_energy))

    # ------------------------- MODIFIED: Use Multithreading for Simulated Annealing -------------------------
    # Define the simulated annealing process for each thread
    def sa_worker(worker_id, init_solution, init_energy, init_blocked_slots):
        current_solution_worker = copy.deepcopy(init_solution)
        current_energy_worker = init_energy
        best_solution_worker = current_solution_worker.copy()
        best_energy_worker = current_energy_worker
        temperature_worker = initial_temperature
        iteration = 0
        no_improvement = 0
        local_blocked_slots = copy.deepcopy(init_blocked_slots)
        start_time_worker = time.time()
        while iteration < max_iterations and temperature_worker > 0.1 and no_improvement < 10000:
            temp_blocked_slots = copy.deepcopy(local_blocked_slots)
            new_solution = generate_neighbor_with_activities(current_solution_worker, regular_course_indices,
                                                             index_to_course_row, rooms_df, room_to_index,
                                                             index_to_course, temp_blocked_slots, course_students)
            new_solution.update(large_courses_solution)
            new_energy = calculate_energy(new_solution, index_to_course_row, room_name_to_row,
                                          course_to_index, room_to_index, index_to_course, index_to_room,
                                          course_students, utilization_weight, conflict_weight)
            energy_delta = new_energy - current_energy_worker
            if energy_delta < 0:
                acceptance_probability = 1.0
            elif temperature_worker > 0:
                try:
                    exp_term = -energy_delta / temperature_worker
                    if exp_term < -700:
                        acceptance_probability = 0.0
                    else:
                        acceptance_probability = math.exp(exp_term)
                except OverflowError:
                    acceptance_probability = 0.0
            else:
                acceptance_probability = 0.0
            if energy_delta < 0 or random.random() < acceptance_probability:
                current_solution_worker = new_solution
                current_energy_worker = new_energy
                local_blocked_slots = temp_blocked_slots
                if current_energy_worker < best_energy_worker:
                    best_solution_worker = copy.deepcopy(current_solution_worker)
                    best_energy_worker = current_energy_worker
                    no_improvement = 0
                    print("Worker {}: Iteration {}, Temperature {:.2f}, Found better solution: {}".format(worker_id, iteration, temperature_worker, best_energy_worker))
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            temperature_worker *= cooling_rate
            iteration += 1
            if iteration % 1000 == 0:
                elapsed_time = time.time() - start_time_worker
                print("Worker {}: Iteration {}, Temperature {:.2f}, Current Energy {}, Best Energy {}, Time elapsed {:.2f} seconds".format(worker_id, iteration, temperature_worker, current_energy_worker, best_energy_worker, elapsed_time))
        print("Worker {}: Simulated annealing completed, total iterations: {}, Final Temperature: {:.2f}".format(worker_id, iteration, temperature_worker))
        print("Worker {}: Best solution energy value: {}".format(worker_id, best_energy_worker))
        return best_solution_worker, best_energy_worker

    # Run multiple simulated annealing processes in parallel using multithreading
    num_workers = 8  # Adjust number of threads as needed
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sa_worker, i, current_solution, current_energy, blocked_slots) for i in range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Select the solution with the lowest energy from all thread results
    best_result = min(results, key=lambda x: x[1])
    best_solution, best_energy = best_result
    final_temperature = 0.1  # Default value
    final_iteration = max_iterations  # Default value

    # --- Generate Final Results ---
    print("\nSimulated annealing completed, total iterations: {}, Final Temperature: {:.2f}".format(final_iteration, final_temperature))
    print("Best solution energy value: {}".format(best_energy))

    # Generate the timetable using the modified function, supporting activity types and merged workshops
    schedule_df = convert_solution_to_schedule_with_activities(best_solution, all_courses_df, rooms_df,
                                                               index_to_course, index_to_room, index_to_course_row)

    output_file = 'sa_course_schedule_with_activities.xlsx'
    schedule_df.to_excel(output_file, index=False)
    print("Scheduling completed, a total of {} course time slots scheduled".format(len(schedule_df)))
    print("Results saved to {}".format(output_file))

    # --- Evaluate Optimization Metrics ---
    # Calculate average room utilization
    schedule_df['Utilization'] = (schedule_df['Class Size'] / schedule_df['Room Capacity'] * 100).round(2)
    avg_utilization = schedule_df['Utilization'].mean()

    # Verify that course durations are consistent
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
        print("\nWarning: Inconsistent durations found for the following courses:")
        for course, duration, slots in duration_errors:
            print(f"  Course {course}, Original Duration {duration}, parsed into different time slots: {slots}")
    else:
        print("\nAll courses have consistent durations")

    # Compute student course conflict metrics and get additional conflict details
    conflict_count, conflict_rate, conflict_students, total_students, conflict_details = compute_student_conflict_with_activities(
        schedule_df, course_students)

    print("\nOptimization Metrics:")
    print("Average Room Utilization: {:.2f}%".format(avg_utilization))
    print("Number of Students with Conflicts: {} (out of {} students)".format(conflict_students, total_students))

    # Create a DataFrame for conflict details
    if conflict_details:
        conflict_df = pd.DataFrame(conflict_details)

        # Use ExcelWriter to append to the existing file
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
            # Write conflict details to a new sheet
            conflict_df.to_excel(writer, sheet_name='Course Conflict Details', index=False)

            # Create a worksheet for duration check if there are errors
            if duration_errors:
                duration_error_data = []
                for course, duration, slots in duration_errors:
                    duration_error_data.append({
                        'Course Code': course,
                        'Original Duration': duration,
                        'Parsed Time Slots': ', '.join(map(str, slots))
                    })
                pd.DataFrame(duration_error_data).to_excel(writer, sheet_name='Duration Check', index=False)

            # Create a summary worksheet for optimization metrics
            summary_data = {
                'Metric': ['Average Room Utilization', 'Total Student Conflicts', 'Number of Students with Conflicts', 'Total Students', 'Student Conflict Rate'],
                'Value': [
                    f"{avg_utilization:.2f}%",
                    conflict_count,
                    conflict_students,
                    total_students,
                    f"{conflict_rate:.2f}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Optimization Summary', index=False)

        print(f"Conflict details and optimization metrics have been written to {output_file}")
    else:
        print("No course conflicts were found")

    # Additional statistical analysis
    course_stats = schedule_df['Course Code'].value_counts()
    room_stats = schedule_df['Room'].value_counts()
    activity_type_stats = schedule_df['Activity Type'].value_counts()

    print("\n----------- Detailed Scheduling Statistics -----------")
    print(f"Total number of entries in the timetable: {len(schedule_df)}")
    print(f"Number of distinct course codes: {len(course_stats)}")
    print(f"Number of distinct rooms: {len(room_stats)}")
    print(f"Number of distinct activity types: {len(activity_type_stats)}")

    # Output room usage details
    print("\nRoom Usage:")
    for room, count in room_stats.head(58).items():
        print(f"  {room}: {count} assignments")

    # Check how many courses in the original data were not scheduled
    all_course_codes = set(all_courses_df['Course Code'].unique())
    scheduled_course_codes = set(schedule_df['Course Code'].unique())
    missed_courses = all_course_codes - scheduled_course_codes

    print(f"\nNumber of courses not scheduled: {len(missed_courses)}")
    if missed_courses:
        print("Examples of courses not scheduled:")
        for code in list(missed_courses)[:10]:
            print(f"  {code}")
        # Before ending, add room usage statistics
        used_rooms = set(schedule_df['Room'].unique())
        all_rooms = set(rooms_df['ROOM NAME'])
        unused_rooms = all_rooms - used_rooms

        print("\n----------- Room Usage Statistics -----------")
        print(f"Total number of rooms: {len(all_rooms)}")
        print(f"Number of rooms used: {len(used_rooms)}")
        print(f"Number of unused rooms: {len(unused_rooms)}")

        # Output the list of unused rooms
        if unused_rooms:
            print("\nUnused Rooms:")
            for room in sorted(unused_rooms):
                if room in rooms_df['ROOM NAME'].values:
                    room_cap = rooms_df[rooms_df['ROOM NAME'] == room]['CAP'].values[0]
                    print(f"  {room} (Capacity: {room_cap})")

        # Output usage frequency for each room
        print("\nRoom Usage Frequency:")
        room_usage = schedule_df['Room'].value_counts()
        for room, count in room_usage.nlargest(20).items():
            if room in rooms_df['ROOM NAME'].values:
                room_cap = rooms_df[rooms_df['ROOM NAME'] == room]['CAP'].values[0]
                print(f"  {room} (Capacity: {room_cap}): used {count} times")

    if len(schedule_df) < len(all_courses_df) * 0.8:  # Less than 80% of courses scheduled
        print("Warning: Too few courses were scheduled. Retrying with relaxed constraints...")
        return simulated_annealing_scheduling(
            enrollment_file=enrollment_file,
            courses_file=courses_file,
            rooms_file=rooms_file,
            max_iterations=max_iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            conflict_weight=30.0,  # Lower conflict penalty
            utilization_weight=1.0,  # Reduce emphasis on utilization
        )

    return schedule_df, course_students, conflict_details


def generate_initial_solution_improved(regular_course_indices, index_to_course_row,
                                       rooms_df, room_to_index, blocked_slots):
    """Improved initial solution generation function, handling constraints more flexibly"""
    solution = {}
    course_success_count = 0  # Number of courses successfully scheduled
    total_week_count = 0  # Total number of teaching weeks
    successful_week_count = 0  # Number of teaching weeks successfully scheduled

    for course_idx in regular_course_indices:
        course_row = index_to_course_row[course_idx]
        course_size = course_row['Real Size']
        course_code = course_row.get('Course Code', f"Unknown_{course_idx}")
        teaching_weeks = course_row['Teaching_Weeks']

        # Check if teaching weeks is empty
        if not teaching_weeks:
            print(f"Warning: Course {course_code} has no valid teaching weeks, using default teaching weeks (1-12)")
            teaching_weeks = list(range(1, 13))

        # Get course duration
        original_duration = course_row.get('Duration', '1:00')
        duration_slots = parse_duration(original_duration)

        # Track room usage frequency
        room_usage_count = {}
        for key, (room_idx, _) in solution.items():
            room_usage_count[room_idx] = room_usage_count.get(room_idx, 0) + 1

        # Filter rooms with sufficient capacity and sort by usage frequency
        suitable_rooms = []
        for _, room in rooms_df.iterrows():
            room_idx = room_to_index[room['ROOM NAME']]
            if room['CAP'] >= course_size:
                suitable_rooms.append((room_idx, room_usage_count.get(room_idx, 0)))

        # Sort by usage frequency, preferring rooms used less frequently
        suitable_rooms.sort(key=lambda x: x[1])
        suitable_rooms = [room_idx for room_idx, _ in suitable_rooms]

        # If no room with sufficient capacity is found, use the room with the maximum capacity
        if not suitable_rooms:
            activity_type = course_row.get('Activity Type Name', 'Unknown')
            print(f"Warning: Course {course_code} ({activity_type}, size: {course_size}) does not have a large enough room, trying the largest room")

            # Sort to find the largest room by capacity
            largest_rooms = sorted(rooms_df.to_dict('records'), key=lambda x: x['CAP'], reverse=True)
            if largest_rooms:
                largest_room_idx = room_to_index[largest_rooms[0]['ROOM NAME']]
                suitable_rooms = [largest_room_idx]
                print(f"  Selected largest room with capacity {largest_rooms[0]['CAP']} for course {course_code} (size: {course_size})")
            else:
                print(f"  Error: No room found for course {course_code}")
                continue

        total_week_count += len(teaching_weeks)
        week_success = False

        # Choose a fixed day and time for this course
        fixed_day = random.randint(1, 5)

        # Get all valid starting time slots (avoiding lunch break or sometimes allowing it)
        valid_fixed_slots = list(range(1, 19 - duration_slots + 1))

        fixed_time_slot = random.choice(valid_fixed_slots) if valid_fixed_slots else 1

        for week in teaching_weeks:
            # Use the fixed day and time
            day = fixed_day
            time_slot = fixed_time_slot
            weekday_idx = 5 * (week - 1) + day

            # Try multiple times to find a suitable room
            max_attempts = 500
            success = False

            for attempt in range(max_attempts):
                # Check if all time slots occupied by the course are not blocked
                all_slots_available = True
                available_rooms = list(suitable_rooms)  # Copy the list of available rooms

                for slot_offset in range(duration_slots):
                    curr_slot = time_slot + slot_offset
                    key = (weekday_idx, curr_slot)
                    if key in blocked_slots:
                        # Remove rooms that are blocked in this time slot
                        available_rooms = [r for r in available_rooms if r not in blocked_slots[key]]
                        if not available_rooms:
                            all_slots_available = False
                            break

                # If all time slots have an available room, schedule the course
                if all_slots_available and available_rooms:
                    # Prefer the room used less frequently
                    chosen_room = available_rooms[0]

                    # Schedule the course in consecutive time slots; save the starting time slot and duration
                    solution[(course_idx, weekday_idx, time_slot)] = (chosen_room, duration_slots)

                    # Update blocked_slots to mark all time slots occupied by the course as blocked
                    for slot_offset in range(duration_slots):
                        curr_slot = time_slot + slot_offset
                        key = (weekday_idx, curr_slot)
                        if key not in blocked_slots:
                            blocked_slots[key] = set()
                        blocked_slots[key].add(chosen_room)

                    success = True
                    successful_week_count += 1
                    break

                # If the current time slot is not available, try other rooms or times
                if attempt > max_attempts // 2:
                    # In later attempts, try changing the time
                    day = random.randint(1, 5)

                    valid_start_slots = list(range(1, 19 - duration_slots + 1))

                    time_slot = random.choice(valid_start_slots) if valid_start_slots else 1
                    weekday_idx = 5 * (week - 1) + day

            # If after many attempts no suitable time and room are found, record a warning
            if success:
                week_success = True
            else:
                activity_type = course_row.get('Activity Type Name', 'Unknown')
                print(f"Warning: Unable to find a suitable time and room for course {course_code} ({activity_type}) in week {week}")

        if week_success:
            course_success_count += 1

    print(f"\nInitial solution generation complete:")
    print(f"Number of courses successfully scheduled: {course_success_count}/{len(regular_course_indices)}")
    print(f"Number of teaching weeks successfully scheduled: {successful_week_count}/{total_week_count}")
    print(f"Number of assignments in the initial solution: {len(solution)}")

    return solution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Course Scheduling System (supports different activity types and merged workshops)")
    parser.add_argument('--enrollment_file', type=str, default='math_student_enrollment.xlsx',
                        help='Path to enrollment data file')
    parser.add_argument('--courses_file', type=str, default='df_final_cleaned_1.xlsx',
                        help='Path to course data file')
    parser.add_argument('--rooms_file', type=str, default='Timetabling_KB_Rooms.xlsx',
                        help='Path to room data file')
    parser.add_argument('--max_iterations', type=int, default=200000,
                        help='Maximum number of iterations')
    parser.add_argument('--initial_temperature', type=float, default=1000,
                        help='Initial temperature')
    parser.add_argument('--cooling_rate', type=float, default=0.995,
                        help='Cooling rate')
    parser.add_argument('--utilization_weight', type=float, default=1,
                        help='Room utilization weight')
    parser.add_argument('--conflict_weight', type=float, default=50000,
                        help='Student course conflict weight')
    args = parser.parse_args()

    result_schedule, course_students, conflict_details = simulated_annealing_scheduling(
        enrollment_file=args.enrollment_file,
        courses_file=args.courses_file,
        rooms_file=args.rooms_file,
        max_iterations=100000,  # Increased iterations
        initial_temperature=10000,  # Higher initial temperature
        cooling_rate=0.998,  # Slower cooling rate
        utilization_weight=80,  # Increased room utilization weight
        conflict_weight=100  # Keep student conflict weight as soft constraint
    )

    conflict_count, conflict_rate, num_conflict_students, total_students, _ = compute_student_conflict_with_activities(
        result_schedule, course_students)

    print("Total number of students:", total_students)
    print("Number of students with conflicts:", num_conflict_students)
