-- Trigger qui reset valid_email quand email est changé
DELIMITER //

CREATE TRIGGER validation
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email <> OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END//

DELIMITER ;
