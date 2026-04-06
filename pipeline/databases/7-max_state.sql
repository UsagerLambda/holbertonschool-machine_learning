-- Affiche la temperature maximale par etat dans l'ordre alphabetique
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state;
