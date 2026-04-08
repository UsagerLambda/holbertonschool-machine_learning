-- Stored procedure AddBonus: ajoute une correction pour un étudiant
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
    -- variable pour stocker l'id du projet trouvé ou crée
    DECLARE v_project_id INT;

    -- Récupère l'id du projet si il existe
    SELECT id INTO v_project_id FROM projects WHERE name = p_project_name LIMIT 1;

    -- Crée le projet s'il n'existe pas
    IF v_project_id IS NULL THEN
        INSERT INTO projects (name) VALUES (p_project_name);
        SET v_project_id = LAST_INSERT_ID();
    END IF;

    -- Insère la correction
    INSERT INTO corrections (user_id, project_id, score) VALUES (p_user_id, v_project_id, p_score);
END//
DELIMITER ;
