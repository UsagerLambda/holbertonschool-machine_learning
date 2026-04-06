-- Affiche les temperatures moyenne pour chaque ville dans l'ordre descendant
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
