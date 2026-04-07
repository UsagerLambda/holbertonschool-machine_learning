-- Trigger qui décrémente la table items par rapport aux orders
DELIMITER //

CREATE TRIGGER buy
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
END//

DELIMITER ;
