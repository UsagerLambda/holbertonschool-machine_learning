-- Fonction qui divise a par b et retourne le résultat (ou 0 si b == 0)
DROP FUNCTION IF EXISTS SafeDiv;

DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    END IF;
    RETURN a / b;
END//
DELIMITER ;
