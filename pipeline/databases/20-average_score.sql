-- Stored procedure ComputeAverageScoreForUser: Calcule la moyenne des notes d'un elève sur tous les projet
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN p_user_id INT
)
BEGIN
    DECLARE v_average FLOAT;

    -- Récupère l'id du projet si il existe
    SELECT AVG(score) INTO v_average FROM corrections WHERE user_id = p_user_id;

    UPDATE users SET average_score = v_average WHERE id = p_user_id;
END//
DELIMITER ;
