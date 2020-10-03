# Data import
import numpy as np
import pandas as pd

# Whether or not planned (future) courses and masters should be included
includeCoursePlan = True
includeMasters = False

# Load the datasets
df_students = pd.read_csv('../data/students.csv', delimiter=";", decimal=",", encoding="utf-8-sig")
df_students_course_history = pd.read_csv('../data/students_course_history.csv', delimiter=";", encoding="utf-8-sig")
df_students_course_plan = pd.read_csv('../data/students_course_plan.csv', delimiter=";", encoding="utf-8-sig")
df_students_courses_results = pd.read_csv('../data/student_courses_results.csv', delimiter=",", encoding="utf-8-sig")

# Remove masters
if not includeMasters:
    df_students = df_students[df_students['degree'] == 'BSc']

# Append course plan (if selected)
df_students_courses = df_students_course_history
if includeCoursePlan:
    df_students_courses = df_students_courses.append(df_students_course_plan)
    
# Calculate student age
df_students['age'] = df_students['start_age'] + ((pd.Timestamp('2020-10-01T12') - pd.to_datetime(df_students['start_date'], format="%d-%m-%Y")) / np.timedelta64(1, 'D') / 365)


# K-out-of-1 encode the courses
courses_encoding = pd.crosstab(df_students_courses['student_id'],df_students_courses['course'],margins = False)
courses_encoding[courses_encoding != 0] = 1
courses_encoding = courses_encoding.reset_index()

# K/D calculator
KD_courses = pd.crosstab(df_students_courses_results['student_id'],df_students_courses_results['fortolket_res'])
data_div = KD_courses['bestået'] / (KD_courses['bestået'] + KD_courses['ikke bestået'] + KD_courses['forsøg'])
df_data_div = data_div.to_frame().reset_index()
df_data_div.columns = ['student_id','kd'] 

# Combine students with K-out-of-1 encoded courses
df_students = pd.merge(df_students, df_data_div, on='student_id', how='left').fillna(1)
df_students = pd.merge(df_students, courses_encoding, on='student_id',how='left')

# Extract matrix
X = df_students.get_values()
attributeNames = df_students.columns
N = len(X)
M = len(attributeNames)

