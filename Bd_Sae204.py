# Importing modules
from random import randint
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import psycopg
import math
from scipy.stats import binom


# Function to display a curve of tab values
def showGraph(tab: list, title="Sensor Measurement 0"):
    x = []
    y = []

    for i in range(len(tab)):
        x.append(i + 1)
        y.append(tab[i])

    # Calculating mean, standard deviation, and regression line
    reg_line = regression_line(x, y)
    y2 = [reg_line[1] + reg_line[0] * x[i] for i in range(len(x))]
    deviation = standard_deviation(tab)
    avg = mean(tab)

    plt.plot(x, y, 'c')  # Sensor results
    plt.plot([1, len(tab)], [-2 * deviation + avg, -2 * deviation + avg], 'r')  # Minimum acceptable line
    plt.plot([1, len(tab)], [2 * deviation + avg, 2 * deviation + avg], 'r')  # Maximum acceptable line
    plt.plot([1, len(tab)], [avg, avg], 'k')  # Mean line
    plt.plot(x, y2, 'y')  # Regression line

    plt.grid()
    plt.xlabel("Number")
    plt.ylabel("Variation")
    if title == "Sensor Measurement 0":
        plt.title("Sensor Measurements")
    else:
        plt.title(title)
    plt.show()


def showHistogram(tab: list, title="Sensor Measurement 0"):
    x = []
    for i in range(len(tab)):
        x.append(round(tab[i], 2))

    plt.hist(x, rwidth=0.95)

    plt.xlabel("Offsets")
    plt.ylabel("Counts")
    if title == "Sensor Measurement 0":
        plt.title("Sensor Measurements")
    else:
        plt.title(title)
    plt.show()


def showHistogram2(tab, title="Sensor Measurement 0"):
    gap = standard_deviation(tab)

    mini = -4 * gap
    maxi = 4 * gap
    bins = np.arange(mini, maxi + gap, gap)
    my_xsticks = ['-4σ', '-3σ', '-2σ', '-1σ', '0', '1σ', '2σ', '3σ', '4σ']
    plt.xticks(np.arange(mini, maxi + gap, gap), my_xsticks)
    counts, edges, patches = plt.hist(tab, bins=bins, align='mid', edgecolor='black')

    for count, edge in zip(counts, edges):
        height = count
        plt.text(edge + gap / 2, height, str(round((int(count) / len(tab)) * 100, 1)) + '%', ha='center', va='bottom')

    plt.ylabel("Number of Occurrences")

    if title == "Sensor Measurement 0":
        plt.title("Sensor Measurements")
    else:
        plt.title(title)

    plt.show()


# Function to connect to the database
def connect():
    return psycopg.connect(
        host="iutinfo-sgbd.uphf.fr",
        dbname="capteurs",
        user="iutinfo313",
        password="uJV5YZBr"
    )


def question1(conn):
    sql = """SELECT (controlvalue - sensorvalue) AS difference
             FROM controlmeasurement 
             JOIN sensormeasurement ON sensortimestamp = timestamp
             GROUP BY controlvalue, sensorvalue
             LIMIT 500;"""
    with conn.execute(sql) as cur:
        s = cur.fetchall()
        return s


def question2(conn):
    sql = """
    WITH value_date AS (
        SELECT (controlvalue - sensorvalue) AS difference, CAST(sensortimestamp AS DATE) AS d
        FROM controlmeasurement 
        JOIN sensormeasurement ON sensortimestamp = timestamp
        WHERE sensortimestamp >= '2024-03-23' AND sensortimestamp <= '2024-03-31'
        GROUP BY controlvalue, sensorvalue, d
        ORDER BY d ASC
    )
    SELECT AVG(value_date.difference)
    FROM value_date
    GROUP BY d;
    """
    with conn.execute(sql) as cur:
        s = cur.fetchall()
        return s


def question4(conn, num):
    sql = """SELECT (controlvalue - sensorvalue) AS difference
             FROM controlmeasurement 
             JOIN sensormeasurement ON sensortimestamp = timestamp
             WHERE sensormeasurement.sensorid = %s
             GROUP BY controlvalue, sensorvalue
             LIMIT 500;"""
    with conn.execute(sql, [num]) as cur:
        s1 = cur.fetchall()
        return s1


# Function to calculate the mean of a list
def mean(tab: list) -> float:
    return stat.mean(tab)


# Function to calculate the standard deviation of a list
def standard_deviation(tab: list) -> float:
    return stat.stdev(tab)


def extract(tab: list) -> list:
    new_tab = []
    for elt in tab:
        new_tab.append(elt[0])
    return new_tab


def extract_values(conn, rep) -> list:
    if rep == 0:
        tab = extract(question1(conn))
    else:
        tab = extract(question4(conn, rep))
    return tab


# Function to count the points within -σ and +σ
def qStat(tab: list, p) -> int:
    avg = mean(tab)
    deviation = p * standard_deviation(tab)
    count = 0
    for x in tab:
        if (avg - deviation) <= x <= (avg + deviation):
            count += 1
    return count


# Function to verify the percentage of points within σ +/-
def verify_percentage(tab: list, p, n):
    total_points = len(tab)
    pts_qStat = qStat(tab, n)
    percentage_qStat = (pts_qStat / total_points) * 100
    print(f"Number of points within ", n, "σ +/- : ", pts_qStat, "/", str(total_points))
    print(f"Percentage of points within the interval σ +/- : {percentage_qStat}%")
    result = percentage_qStat - p

    print("The difference with the interval of", p, "% is: ", round(result, 3), "% \n")


# Part 4 test
def part4(tab: list):
    print("")
    verify_percentage(tab, 68, 1)
    verify_percentage(tab, 95, 2)
    verify_percentage(tab, 99.7, 3)


def regression_line(x: list, y: list) -> list:
    b1 = round(covariance(x, y) / variance(x), 2)
    b0 = round(mean(y) - b1 * mean(x), 2)
    return [b1, b0]


def variance(x: list) -> float:
    tab = []
    avg = mean(x)
    for elt in x:
        tab.append((elt - avg) ** 2)
    return round(mean(tab), 2)


def covariance(x: list, y: list):
    return round(mean([(x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x))]), 2)


def prob(x: int, n: int, p: float) -> float:
    return 1 - binom.cdf(x - 1, n, p)


# Rule:

def binomial_prob(n, k, p):
    binomial_coefficient = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))


def calculate_p_value(n, k, p):
    probability = 0
    for i in range(k, n + 1):
        probability += binomial_prob(n, i, p)
    return probability


# Parameters for the rule for at least 8 out of the last 10 points on the same side of the mean:

def calculate_p_value1():
    # Calculating p-values:
    p_value8 = calculate_p_value(10, 8, 0.5)
    p_value5 = calculate_p_value(7, 5, 0.1)
    p_value7 = calculate_p_value(10, 7, 0.32)

    print(f"The p-value for the rule 'at least 8 out of the last 10 points on the same side of the mean' is: {p_value8:.4f}")
    print(f"The p-value for the rule 'at least 5 out of the last 7 points outside the interval [-2σ, +2σ]' is: {p_value5:.6f}")
    print(f"The p-value for the rule 'at least 7 out of the last 10 points outside the interval [-σ, +σ]' is: {p_value7:.5f}")

    # Condition to display the message in the Notebook
    if p_value8 < 0.05:
        print("Rule triggered: At least 8 out of the last 10 points on the same side of the mean.")
    elif p_value5 < 0.05:
        print("Rule triggered: At least 5 out of the last 7 points outside the interval [-2σ, +2σ].")
    elif p_value8 > 0.05:
        print("Rule triggered: At least 7 out of the last 10 points outside the interval [-σ, +σ].")
    else:
        print("Rule not triggered.")


# Attempt to rule

def test_pValueRule():
    # Total number of points
    n = 12
    # Minimum number of points outside the interval
    k = 6
    # Probability that a point is outside the interval [-1.5σ, +1.5σ]
    p = 0.1336

    # Cumulative probability calculation up to k-1 successes
    p_value_cumulative = binom.cdf(k - 1, n, p)
    # Complementary p-value calculation
    p_value_theoretical = 1 - p_value_cumulative
    # Displaying the theoretical p-value
    print("The theoretical p-value for the rule is: ", round(p_value_theoretical, 5))

    # Creating a random number generator
    rng = np.random.default_rng()
    # Normal distribution parameters
    mu = 0  # Mean
    sigma = 1  # Standard deviation

    # Number of simulations
    num_simulations = 100000
    # Counter for triggered rules
    count = 0

    # Performing the simulations
    for _ in range(num_simulations):
        # Generate 12 points following a normal distribution
        data = rng.normal(mu, sigma, n)

        # Count the points outside the interval [-1.5σ, +1.5σ]
        outside_points = 0
        for point in data:
            if point < -1.5 * sigma or point > 1.5 * sigma:
                outside_points += 1

        # Check if the rule is triggered
        if outside_points >= k:
            count += 1

    # Calculate the experimental p-value
    p_value_experimental = count / num_simulations
    # Displaying the experimental p-value
    print("The experimental p-value for the rule is: ", round(p_value_experimental, 5))


def prog():
    conn = connect()

    rep = int(input("Which sensor would you like to see? \nIf all, press 0: "))
    tab = extract_values(conn, rep)
    rep = "Sensor Measurement " + str(rep)

    finished = False
    while not finished:
        rep2 = int(input("1 - Display the Graph\n2 - Display the Histogram \n3 - View the theoretical distribution \n4 - Calculate the p-value \n5 - Test rule \n6 - Exit\nWhat is your choice: "))
        if rep2 == 1:
            showGraph(tab, rep)
        elif rep2 == 2:
            showHistogram2(tab, rep)
        elif rep2 == 3:
            part4(tab)
        elif rep2 == 4:
            calculate_p_value1()
        elif rep2 == 5:
            test_pValueRule()
        else:
            finished = True


if __name__ == "__main__":
    prog()
    test_pValueRule()
