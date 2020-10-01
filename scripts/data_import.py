# Data import
import numpy as np
import pandas as pd

# Whether or not planned (future) courses and masters should be included
includeCoursePlan = True
includeMasters = False

# Load the datasets
df_students = pd.read_csv('../data/students.csv', delimiter=";", decimal=",")
df_students_course_history = pd.read_csv('../data/students_course_history.csv', delimiter=";")
df_students_course_plan = pd.read_csv('../data/students_course_plan.csv', delimiter=";")
df_students_courses_results = pd.read_csv('../data/student_courses_results.csv', delimiter=",")

# Remove masters
if not includeMasters:
    df_students = df_students[df_students['degree'] == 'BSc']

# Append course plan (if selected)
df_students_courses = df_students_course_history
if includeCoursePlan:
    df_students_courses = df_students_courses.append(df_students_course_plan)

# K-out-of-1 encode the courses
courses_encoding = pd.crosstab(df_students_courses['student_id'], 
                            df_students_courses['course'],  
                               margins = False)  

courses_encoding[courses_encoding != 0] = 1


# K/D calculator
KD_courses = pd.crosstab(df_students_courses_results['student_id'],df_students_courses_results['fortolket_res'])
data_div = KD_courses['bestået']/(KD_courses['bestået'] + KD_courses['ikke bestået'] + KD_courses['forsøg'])
df_data_div = data_div.to_frame()
df_data_div.columns=['kd'] 

# Combine students with K-out-of-1 encoded courses
df_students = df_students.set_index('student_id')
df_students = pd.merge(df_students, df_data_div, on='student_id', how='left').fillna(1)
df_students = pd.merge(df_students, courses_encoding, on='student_id',how='left')
KD_courses = pd.merge(KD_courses, df_data_div, left_index=True, right_index=True)


# Extract matrix
df_students = df_students.reset_index()
#X = df_students.get_values()
attributeNames = df_students.columns
#N = len(X)
M = len(attributeNames)

