-- Sélectionne les groupes de Glam rock et renvoie leurs durée de vie
SELECT
    band_name,
    CASE
        WHEN split IS NOT NULL THEN split - formed
        ELSE 2020 - formed
    END AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%';
