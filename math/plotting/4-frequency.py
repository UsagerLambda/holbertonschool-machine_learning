#!/usr/bin/env python3
"""
Histogram of student grade
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """_summary_
    Generates and displays a histogram of student grades.
    Shows grade distribution for 50 students with labeled axes and title.
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
