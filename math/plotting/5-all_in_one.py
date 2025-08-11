#!/usr/bin/env python3
"""
Affiche plusieurs graphiques de données sur une seule figure.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Creates a figure with differents subplots for various data visualizations.
    Includes line, scatter, exponential decay, and histogram plots.
    Displays all plots in a single window with appropriate titles and labels.
    """

    y0 = np.arange(0, 11) ** 3  # 0-line

    mean = [69, 0]  # 1-scatter
    cov = [[15, 8], [8, 15]]  # 1-scatter
    np.random.seed(5)  # 1-scatter
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T  # 1-scatter
    y1 += 180  # 1-scatter

    x2 = np.arange(0, 28651, 5730)  # 2-change_scale
    r2 = np.log(0.5)  # 2-change_scale
    t2 = 5730  # 2-change_scale
    y2 = np.exp((r2 / t2) * x2)  # 2-change_scale

    x3 = np.arange(0, 21000, 1000)  # 3-two
    r3 = np.log(0.5)  # 3-two
    t31 = 5730  # 3-two
    t32 = 1600  # 3-two
    y31 = np.exp((r3 / t31) * x3)  # 3-two
    y32 = np.exp((r3 / t32) * x3)  # 3-two

    np.random.seed(5)  # 4-frequency
    student_grades = np.random.normal(68, 15, 50)  # 4-frequency

    plt.figure()

    g1 = plt.subplot2grid((3, 2), (0, 0))
    x = np.arange(0, 11)  # 0-line
    g1.plot(x, y0, 'r-')

    g2 = plt.subplot2grid((3, 2), (0, 1))
    g2.scatter(x1, y1, color='m')
    g2.set_title("Men's Height vs Weight", fontsize='x-small')

    g3 = plt.subplot2grid((3, 2), (1, 0))
    g3.plot(x2, y2)
    g3.set_yscale('log')
    g3.set_xlabel("Time (years)", fontsize='x-small')
    g3.set_ylabel("Fraction Remaining", fontsize='x-small')
    g3.set_title("Exponential Decay of C-14", fontsize='x-small')

    g4 = plt.subplot2grid((3, 2), (1, 1))
    g4.plot(x3, y31, c='r', ls="--", label="C-14")
    g4.plot(x3, y32, c='g', label="Ra-226")
    g4.set_ylabel("Fraction Remaining", fontsize='x-small')
    g4.set_title("Exponential Decay of Radioactive Elements",
                 fontsize='x-small')
    g4.legend(loc='upper right', fontsize='x-small')

    g5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    g5.hist(student_grades, bins=range(0, 101, 10), range=(0, 100),
            edgecolor='black')
    g5.set_xlabel("Grades", fontsize='x-small')
    g5.set_ylabel("Number of Students", fontsize='x-small')
    g5.set_title("Project A", fontsize='x-small')
    g5.set_xticks(range(0, 101, 10))
    g5.set_yticks(range(0, 31, 10))
    g5.set_ylim(0, 30)
    g5.set_xlim(0, 100)

    plt.suptitle("All in One")
    plt.tight_layout()
    plt.show()
