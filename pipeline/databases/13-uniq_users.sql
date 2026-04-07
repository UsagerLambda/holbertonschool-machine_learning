-- Créer une table users avec des champs contraignants
CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(256),
    email VARCHAR(256) NOT NULL UNIQUE
);
