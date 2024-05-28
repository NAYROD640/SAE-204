# Imporations des modules
from random import randint
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
import psycopg
import math
from scipy.stats import binom


# Fonction pour afficher une courbe des valeur de tab
def showGraph(tab: list, title="Mesure du capteur 0"):
    x = []
    y = []

    #
    for i in range(len(tab)):
        x.append(i+1)
        y.append(tab[i])

    # Calcul de la moyenne, écart type et la droite de regression
    dr = droite_regression(x, y)
    y2 = [dr[1] + dr[0] * x[i] for i in range(len(x))]
    ecart = standardDeviation(tab)
    moy = mean(tab)

    plt.plot(x, y, 'c') # Résu des capteurs
    plt.plot([1, len(tab)], [-2 * ecart + moy, -2 * ecart + moy], 'r')  # Ligne mini acceptable
    plt.plot([1, len(tab)], [2 * ecart + moy, 2 * ecart + moy], 'r')  # Ligne maxi acceptable
    plt.plot([1, len(tab)], [moy, moy], 'k')  # Ligne de la moyenne
    plt.plot(x, y2, 'y')  # Droite de regression

    plt.grid()
    plt.xlabel("Numéro")
    plt.ylabel("Variation")
    if title == "Mesure du capteur 0":
        plt.title("Mesures des capteurs")
    else:
        plt.title(title)
    plt.show()


def showHistogram(tab: list, title="Mesure du capteur 0"):
    x = []
    for i in range(len(tab)):
        x.append(round(tab[i], 2))

    plt.hist(x, rwidth=0.95)

    plt.xlabel("Décalages")
    plt.ylabel("Nombres")
    if title == "Mesure du capteur 0":
        plt.title("Mesures des capteurs")
    else:
        plt.title(title)
    plt.show()

def showHistogram2(tab, title="Mesure du capteur 0"):
    gap = standardDeviation(tab)

    mini = -4 * gap
    maxi = 4 * gap
    bins = np.arange(mini, maxi + gap, gap)
    my_xsticks = ['-4σ', '-3σ', '-2σ', '-1σ', '0', '1σ', '2σ', '3σ', '4σ']
    plt.xticks(np.arange(mini, maxi + gap, gap), my_xsticks)
    counts, edges, patches = plt.hist(tab, bins=bins, align='mid', edgecolor='black')

    for count, edge in zip(counts, edges):
        height = count
        plt.text(edge + gap / 2, height, str(round((int(count) / len(tab)) * 100, 1)) + '%', ha='center', va='bottom')

    plt.ylabel("Nombres de prises")

    if title == "Mesure du capteur 0":
        plt.title("Mesures des capteurs")
    else:
        plt.title(title)

    plt.show()

# Fonction permettant de se connecter à la base de données
def connect():
    return psycopg.connect(
        host="iutinfo-sgbd.uphf.fr",
        dbname="capteurs",
        user="iutinfo313",
        password="uJV5YZBr");


def question1(conn):
    sql = """SELECT (controlvalue-sensorvalue)as différence
    From controlmeasurement join sensormeasurement on sensortimestamp = timestamp
    GROUP BY controlvalue, sensorvalue
    LIMIT 500;"""
    with conn.execute(sql) as cur:
        s = cur.fetchall()
        return s


def question2(conn):
    sql = """
    with valeur_date as(
   SELECT (controlvalue-sensorvalue)as différence, cast(sensortimestamp AS DATE) as d
    From controlmeasurement join sensormeasurement on sensortimestamp = timestamp
    where sensortimestamp >= '2024-03-23' AND sensortimestamp <= '2024-03-31'
    GROUP BY controlvalue, sensorvalue, d
    ORDER BY d ASC
    )
    SELECT avg(valeur_date.différence)
    FROM valeur_date
    group by d;
    """
    with conn.execute(sql) as cur:
        s = cur.fetchall()
        return s


def question4(conn, num):
    sql = """SELECT (controlvalue-sensorvalue)as différence
From controlmeasurement join sensormeasurement on sensortimestamp = timestamp
WHERE sensormeasurement.sensorid = %s
GROUP BY controlvalue, sensorvalue
LIMIT 500;"""
    with conn.execute(sql, [num]) as cur:
        s1 = cur.fetchall()
        return s1


# Fonction pour avoir la moyenne d'une liste
def mean(tab: list) -> float:
    return stat.mean(tab)


# Fonction pour avoir l'ecart type d'une liste
def standardDeviation(tab: list) -> float:
    return stat.stdev(tab)


def extraire(tab: list) -> float:
    newTab = []
    for elt in tab:
        newTab.append(elt[0])
    return newTab


def extractionValeur(conn, rep) -> list:
    if rep == 0:
        tab = extraire(question1(conn))
    else:
        tab = extraire(question4(conn, rep))
    return tab


# Fonction pour compter les points compris entre -σ et +σ
def qStat(tab: list, p) -> int:
    moy = mean(tab)
    ecart = p * standardDeviation(tab)
    cpt = 0
    for x in tab:
        if (moy - ecart) <= x <= (moy + ecart):
            cpt += 1
    return cpt


# Fonction pour vérifier le pourcentage de points dans σ +/-
def verify_percentage(tab: list, p, n):
    total_points = len(tab)
    pts_qStat = qStat(tab, n)
    pourcentage_qStat = (pts_qStat / total_points) * 100
    print(f"Nombre de points dans ", n, "σ +/- : ", pts_qStat, "/", str(total_points))
    print(f"Pourcentage de points dans l'intervalle σ +/- : {pourcentage_qStat}%")
    resu = pourcentage_qStat - p

    print("La différence avec l'intervalle de", p, "% est de :", round(resu, 3), "% \n")


# Test partie 4
def part4(tab: list):
    print("")
    verify_percentage(tab, 68, 1)
    verify_percentage(tab, 95, 2)
    verify_percentage(tab, 99.7, 3)


def droite_regression(x: list, y: list) -> list:
    b1 = round(covariance(x, y) / variance(x), 2)
    b0 = round(mean(y) - b1 * mean(x), 2)
    return [b1, b0]


def variance(x: list) -> float:
    tab = []
    moy = mean(x)
    for elt in x:
        tab.append((elt - moy) ** 2)
    return round(mean(tab), 2)


def covariance(x: list, y: list):
    return round(mean([(x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x))]), 2)


def prob(x: int, n: int, p: float) -> float:
    return 1 - binom.cdf(x - 1, n, p)

# Règle :

def binomial_prob(n, k, p):
    binomial_coefficient = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))

def calculate_p_value(n, k, p):
    probability = 0
    for i in range(k, n + 1):
        probability += binomial_prob(n, i, p)
    return probability

# Paramètres pour la règle pour au moins 8 points sur les 10 derniers du même côté de la moyenne:

def calculate_p_value1():

    # Calcul des p-valeur :
    p_value8 = calculate_p_value(10, 8, 0.5)
    p_value5 = calculate_p_value(7, 5, 0.1)
    p_value7 = calculate_p_value(10, 7, 0.32)


    print(f"La p-valeur pour la règle 'au moins 8 points sur les 10 derniers du même côté de la moyenne' est de : {p_value8:.4f}")
    print(f"La p-valeur pour la règle 'au moins 5 points sur les 7 derniers en dehors de l’intervalle [-2σ, +2σ]' est de : {p_value5:.6f}")
    print(f"La p-valeur pour la règle 'au moins 7 points sur les 10 derniers en dehors de l’intervalle [-σ, +σ]' est de : {p_value7:.5f}")

    # Condition pour afficher le message dans le Notebook
    if p_value8 < 0.05 :
        print("Règle déclenchée : Au moins 8 points sur les 10 derniers du même côté de la moyenne.")

    elif p_value5 < 0.05:
        print("Règle déclenchée : Au moins 5 points sur les 7 derniers en dehors de l’intervalle [-2σ, +2σ].")

    elif p_value8 > 0.05:
        print("Règle déclenchée : Au moins 7 points sur les 10 derniers en dehors de l’intervalle [-σ, +σ].")

    else:
        print("Règle non déclenchée.")

def prog():
    conn = connect()

    rep = int(input("Quel capteur voulez voir ?\nSi tout appuyer sur 0 : "))
    tab = extractionValeur(conn, rep)

    part4(tab)

    rep = "Mesure du capteur " + str(rep)

    showGraph(tab, rep)
    showHistogram(tab, rep)
    showHistogram2(tab)

    calculate_p_value1()


if __name__ == "__main__":
    # print(prob(12, 15, 0.5))

    prog()

    # print("La moyenne μ est : ",mean(tab))
    # print("L'écrat type σ est : ",standardDeviation(tab))
